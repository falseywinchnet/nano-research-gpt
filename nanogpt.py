#this file contains an experimental, somewhat slow to train, gpt model containing some experimental features
#copyright joshuah rainstar 2025
#licensed under christian freeware license
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import math
import requests

        
# ====================================================
# Experiment Toggles
# ====================================================
USE_STAN = False          # If True, use Self-Scalable Tanh (STAN) in transformer MLPs; else use GELU.
USE_DIFF_XOR = False      # If True, inject a differentiable XOR module into each transformer block.
USE_HLOSS = True         # If True, use harmonic loss (as defined in Baek et al.) in place of standard cross-entropy.
EPS = 1e-8               # Small constant for numerical stability.

# For reproducibility.
torch.manual_seed(42)

# ====================================================
# Self-Scalable Sinh + Tanh (Tammy and Steve)
#2204.12589v2.pdf + my own changes
# ====================================================
class SelfScalableTanh(nn.Module):
    def __init__(self, init_scale=0.1, max_scale=0.12):  # Safe upper bound
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))

    def forward(self, x):
        return torch.tanh(x) + self.scale * torch.tanh(x)


# ====================================================
# Differentiable XOR Module
# ====================================================
# This module splits the input tensor along the feature dimension (which must be even),
# applies a sigmoid to each half (to constrain values to [0, 1]),
# computes a soft-XOR via a + b - 2 * a * b,
# and projects the result back to the original embedding dimension.
class DifferentiableXORLayer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        assert embed_dim % 2 == 0, "Embedding dimension must be even for the XOR module."
        self.embed_dim = embed_dim
        self.proj = nn.Linear(embed_dim // 2, embed_dim)


    def forward(self, x):
        d = self.embed_dim // 2
        x1, x2 = x[..., :d], x[..., d:]
        a = torch.sigmoid(x1)
        b = torch.sigmoid(x2)
        xor_out = 0.5 * (a + b - 2 * a * b)  # Reduce strength by scaling 0.5
        out = self.proj(xor_out)
        return out


# ====================================================
# Transformer Block
# ====================================================
# A minimal transformer block that applies self-attention, layer normalization, and an MLP.
# The MLP uses either STAN (if toggled) or GELU.
# If USE_DIFF_XOR is enabled, a differentiable XOR module is injected after the MLP.
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, res_scale=1.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        activation = SelfScalableTanh() if USE_STAN else nn.GELU()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            activation,
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.res_scale = res_scale
        self.diff_xor = DifferentiableXORLayer(embed_dim) if USE_DIFF_XOR else None

    def forward(self, x):
        # x: (B, seq, embed_dim)
        x_t = x.transpose(0, 1)  # (seq, B, embed_dim)
        attn_out, _ = self.attn(x_t, x_t, x_t)
        # Apply residual scaling for the attention output.
        x_t = self.ln1(x_t + self.res_scale * attn_out)
        x = x_t.transpose(0, 1)  # (B, seq, embed_dim)
        mlp_out = self.mlp(x)
        if self.diff_xor is not None:
            xor_features = self.diff_xor(mlp_out)
            mlp_out = mlp_out + xor_features
        # Apply residual scaling for the MLP output.
        x = self.ln2(x + self.res_scale * mlp_out)
        return x
def man_torch(arr):
    nonzero_arr = arr[arr > 0]  # Remove zero values
    med = torch.nanmedian(nonzero_arr)  # Median of nonzero values
    return torch.nanmedian(torch.abs(nonzero_arr - med))  # MAD from median
def atd_torch(arr):
    x = torch.square(torch.abs(arr - man_torch(arr)))
    return torch.sqrt(torch.nanmean(x))

# ====================================================
# NanoGPT-like Model with Harmonic Loss Option
# ====================================================
# If USE_HLOSS is True, we replace the final LM head with a harmonic unembedding.
#2502.01628v1.pdf
# Instead of computing logits via an inner product, we compute L2 distances between the penultimate representation
# and class centers (stored in self.unembedding). The probability for class i is defined as:
#   p_i = (1 / d_i^n) / sum_j (1 / d_j^n)
# where d_i = ||w_i - x||_2 and n is the harmonic exponent.
class NanoGPTModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, num_layers=10, num_heads=1, seq_len=128, use_latent_stretch=True, latent_alpha=2.0, placeholder_idx=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.use_latent_stretch = use_latent_stretch
        self.placeholder_idx = placeholder_idx  # Store placeholder index in model

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, embed_dim))

        # Transformer layers
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, res_scale=1/math.sqrt(num_layers))
             for _ in range(num_layers)]
        )

        if USE_HLOSS:
            self.unembedding = nn.Parameter(torch.randn(embed_dim, vocab_size))
            nn.init.kaiming_uniform_(self.unembedding, a=math.sqrt(5))
        else:
            self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.size()
        token_emb = self.token_embedding(x)  # (B, seq, embed_dim)
        pos_emb = self.pos_embedding[:, :seq_len, :]  # (1, seq, embed_dim)
        h = token_emb + pos_emb

        for block in self.blocks:
            h = block(h)

        if USE_HLOSS:
            # Compute variance per embedding dimension
            var_x = torch.var(h, dim=[0, 1], keepdim=True) + EPS  # variance across batch & sequence

            # Compute Mahalanobis distance using diagonal covariance approximation
            h_exp = h.unsqueeze(2)  # (B, seq, 1, D)
            W_T_exp = self.unembedding.t().unsqueeze(0).unsqueeze(0)  # (1, 1, vocab_size, D)
            delta = h_exp - W_T_exp  # (B, seq, vocab_size, D)
            mahalanobis_d = torch.sqrt(torch.sum(delta**2 / var_x, dim=-1))

            # === Adaptive Placeholder Distance Handling ===
            # === Scaling and Probability Computation ===
            scale_factor = mahalanobis_d.mean().item()
            scale_factor = max(scale_factor, EPS)  # Prevent division by zero
            mahalanobis_d_scaled = mahalanobis_d / scale_factor
            mahalanobis_d_clamped = torch.clamp(mahalanobis_d_scaled, min=1e-6)

            harmonic_exponent = int(math.sqrt(self.embed_dim))
            log_inv_dn = -harmonic_exponent * torch.log(mahalanobis_d_clamped + EPS)
            log_sum = torch.logsumexp(log_inv_dn, dim=-1, keepdim=True)
            log_p = log_inv_dn - log_sum
            p = torch.exp(log_p)  # Convert log probabilities to final probability distribution

            # === Debugging: Track Placeholder Probability ===
            return p
        else:
            # Standard LM head path
            logits = self.lm_head(h)  # (B, seq, vocab_size)

            return logits


# ====================================================
# Data Preparation (Shakespeare)
# ====================================================
def load_shakespeare_text():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url).text
    return text

text = load_shakespeare_text()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

def get_batch(batch_size, seq_len):
    ix = torch.randint(0, data.size(0) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x, y

# ====================================================
# Training Setup
# ====================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NanoGPTModel(vocab_size, embed_dim=192, num_layers=8, num_heads=8, seq_len=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=6e-4)

scaler = GradScaler()

# For standard loss, we use cross-entropy; for harmonic loss we compute negative log probability manually.
criterion_ce = nn.CrossEntropyLoss()

num_epochs = 10
batch_size = 10
seq_len = 128


    
# In your generation loop:

# ====================================================
# Training Loop
# ====================================================            if USE_HLOSS:
losses = []
for epoch in range(num_epochs):
    
    model.train()

    total_loss = 0.0
    for step in range(10):  # Adjust the number of steps as needed.
        x_batch, y_batch = get_batch(batch_size, seq_len)
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        with autocast():
            p = model(x_batch)
            if USE_HLOSS:
                    probs_flat = p.view(-1, vocab_size)
                    y_flat = y_batch.view(-1)
                    
                    #probs = apply_logistic_scaling_with_placeholder(p, placeholder_idx, points=1000, temperature=1.0)
                    #probs_flat = probs.view(-1, vocab_size)  # Now shape is (B*seq_len, vocab_size)
                    per_token_loss = -torch.log(probs_flat[torch.arange(probs_flat.size(0)), y_flat] + EPS)  # Avoid log(0)
                    loss = per_token_loss.mean()
            else:
                # Generic GPT (non-harmonic) loss path.
                logits = model(x_batch)  # (B, seq, vocab_size)
                logits_flat = logits.view(-1, vocab_size)
                y_flat = y_batch.view(-1)
                loss = criterion_ce(logits_flat, y_flat)

                
            main_loss = loss.detach()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += main_loss
        losses.append(main_loss.cpu())
        if step % 200 == 0:
            print(f"Epoch {epoch+1}, Step {step}, Loss: {main_loss:.4f}")

    print(f"Epoch {epoch+1} Average Loss: {total_loss/10:.4f}")

# ====================================================
# Evaluation: Text Generation
# ====================================================

    # Decay rate (tune this to control how fast the bonus decays)
model.eval()
with torch.no_grad():
    prompt = text[:50]
    context = torch.tensor(encode(prompt), dtype=torch.long)[None, :].to(device)
    generated = context
    for _ in range(200):  # Generate 200 tokens.
        inp = generated[:, -seq_len:]
        if USE_HLOSS:
            p = model(inp)  # p: (B, seq, vocab_size)

            probs = p[:, -1, :]

            #probs = apply_logistic_scaling_with_placeholder(probs, placeholder_idx)           
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            logits = model(inp)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)
    sample = decode(generated[0].cpu().tolist())
    print("Generated Sample:\n", sample)

