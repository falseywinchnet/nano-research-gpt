#this file contains an experimental, somewhat fast to train, model using a single RNN to guide and pool outputs from transformers
#copyright joshuah rainstar 2025
#licensed under christian freeware license
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from torch.cuda.amp import autocast, GradScaler

# -------------------------------------------------------------------
# Experiment Toggles and Numerical Constants
# -------------------------------------------------------------------
USE_STAN = True         # Toggle self-scalable tanh in activations.
USE_DIFF_XOR = True     # Toggle differentiable XOR.
USE_HLOSS = True         # Use harmonic loss (via unembedding) instead of CE.
EPS = 1e-8               # For numerical stability.

# -------------------------------------------------------------------
# Auxiliary Modules (from the provided code bases)
# -------------------------------------------------------------------

# Self-Scalable Tanh (STAN)
class SelfScalableTanh(nn.Module):
    def __init__(self, init_scale=0.1, max_scale=0.12):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
    def forward(self, x):
        return torch.tanh(x) + self.scale * torch.tanh(x)

# Differentiable XOR Layer
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
        xor_out = 0.5 * (a + b - 2 * a * b)
        out = self.proj(xor_out)
        return out

# Minimal Transformer Block
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
        #self.diff_xor = DifferentiableXORLayer(embed_dim) if USE_DIFF_XOR else None

    def forward(self, x):
        # x: (batch, seq, embed_dim) -> (seq, batch, embed_dim)
        x_t = x.transpose(0, 1)
        attn_out, _ = self.attn(x_t, x_t, x_t)
        x_t = self.ln1(x_t + self.res_scale * attn_out)
        x = x_t.transpose(0, 1)
        mlp_out = self.mlp(x)
        #if self.diff_xor is not None:
         #   xor_features = self.diff_xor(mlp_out)
         #   mlp_out = mlp_out + xor_features
        x = self.ln2(x + self.res_scale * mlp_out)
        return x

# -------------------------------------------------------------------
# Option 2 Synthesis: Transformer-to-RNN Integration
# -------------------------------------------------------------------
# Here we define an RNN cell that takes both a token embedding and a
# transformer output (from the attention heads) to update a single global
# hidden state.

class TransformerRNNCell(nn.Module):
    def __init__(self, token_embed_dim, transformer_dim, hidden_size):
        """
        token_embed_dim: Dimensionality of the token embedding.
        transformer_dim: Dimensionality of the transformer output.
        hidden_size: Dimensionality of the RNN hidden state.
        """
        super().__init__()
        # Combine token embedding and transformer output:
        self.input_proj = nn.Linear(token_embed_dim + transformer_dim, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.activation = SelfScalableTanh() if USE_STAN else nn.Tanh()
        if USE_DIFF_XOR:
            self.diff_xor = DifferentiableXORLayer(hidden_size)
        else:
            self.diff_xor = None

    def forward(self, token_emb, transformer_out, h_prev):
        # Concatenate the token embedding with the transformer output.
        combined = torch.cat([token_emb, transformer_out], dim=-1)
        # Compute a candidate update:
        h_candidate = self.input_proj(combined) + self.h2h(h_prev)
        h_new = self.activation(h_candidate)
        if self.diff_xor is not None:
            h_new = h_new + self.diff_xor(h_new)
        return h_new

# Now we define the overall model that integrates the transformer and the RNN.
class TransformerRNNModel(nn.Module):
    def __init__(self, vocab_size, 
                 embed_dim=128, 
                 transformer_layers=4, 
                 transformer_heads=4, 
                 rnn_hidden_size=256, 
                 seq_len=128, 
                 use_hloss=USE_HLOSS):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.use_hloss = use_hloss

        # Token embedding layer.
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # Positional embedding for the transformer.
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        # A stack of transformer blocks.
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, transformer_heads, dropout=0.0, res_scale=1/math.sqrt(transformer_layers))
            for _ in range(transformer_layers)
        ])
        # A single RNN cell that will be updated sequentially.
        self.rnn_cell = TransformerRNNCell(embed_dim, embed_dim, rnn_hidden_size)

        # Output head: either a harmonic unembedding (if USE_HLOSS) or a standard linear layer.
        if self.use_hloss:
            self.unembedding = nn.Parameter(torch.randn(rnn_hidden_size, vocab_size))
            nn.init.kaiming_uniform_(self.unembedding, a=math.sqrt(5))
        else:
            self.fc = nn.Linear(rnn_hidden_size, vocab_size)

    def forward(self, x, hidden_state=None):
        """
        x: LongTensor of shape (batch, seq_len)
        hidden_state: (batch, rnn_hidden_size) or None (will be initialized to zeros).
        """
        batch_size, seq_len = x.size()
        # Obtain token embeddings.
        token_emb = self.token_embedding(x)  # (batch, seq_len, embed_dim)
        pos_emb = self.pos_embedding[:, :seq_len, :]  # (1, seq_len, embed_dim)
        transformer_input = token_emb + pos_emb

        # Pass through the transformer blocks to get global (attention-based) representations.
        h_transformer = transformer_input
        for block in self.transformer_blocks:
            h_transformer = block(h_transformer)  # (batch, seq_len, embed_dim)

        # Initialize the RNN hidden state if needed.
        if hidden_state is None:
            hidden_state = torch.zeros(batch_size, self.rnn_cell.h2h.out_features, device=x.device)
        
        outputs = []
        # For each timestep, update the RNN hidden state by combining:
        #   - The token embedding at that timestep (local pattern)
        #   - The transformer output at that timestep (global context)
        for t in range(seq_len):
            token_t = token_emb[:, t, :]       # (batch, embed_dim)
            trans_t = h_transformer[:, t, :]     # (batch, embed_dim)
            hidden_state = self.rnn_cell(token_t, trans_t, hidden_state)
            outputs.append(hidden_state)
        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, rnn_hidden_size)

        # Output computation: either via harmonic loss or a standard LM head.
        if self.use_hloss:
            # Compute the Mahalanobis distance–based probabilities.
            var_x = torch.var(outputs, dim=[0, 1], keepdim=True) + EPS
            outputs_exp = outputs.unsqueeze(2)  # (B, seq_len, 1, hidden_size)
            W_T_exp = self.unembedding.t().unsqueeze(0).unsqueeze(0)  # (1, 1, vocab_size, hidden_size)
            delta = outputs_exp - W_T_exp
            mahalanobis_d = torch.sqrt(torch.sum(delta**2 / var_x, dim=-1))
            scale_factor = mahalanobis_d.mean().item()
            scale_factor = max(scale_factor, EPS)
            mahalanobis_d_scaled = mahalanobis_d / scale_factor
            mahalanobis_d_clamped = torch.clamp(mahalanobis_d_scaled, min=1e-6)
            harmonic_exponent = int(math.sqrt(self.rnn_cell.h2h.out_features))
            log_inv_dn = -harmonic_exponent * torch.log(mahalanobis_d_clamped + EPS)
            log_sum = torch.logsumexp(log_inv_dn, dim=-1, keepdim=True)
            log_p = log_inv_dn - log_sum
            p = torch.exp(log_p)
            return p, hidden_state
        else:
            logits = self.fc(outputs)  # (batch, seq_len, vocab_size)
            return logits, hidden_state

# -------------------------------------------------------------------
# (For example) Data Preparation & Training Stub (using Shakespeare)
# -------------------------------------------------------------------
import requests

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

# Example device and model initialization.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerRNNModel(vocab_size, embed_dim=128, transformer_layers=4, transformer_heads=4, 
                            rnn_hidden_size=256, seq_len=128, use_hloss=USE_HLOSS).to(device)
optimizer = optim.Adam(model.parameters(), lr=6e-4)
scaler = GradScaler()

# A simple training loop (using harmonic loss if toggled)
num_epochs = 2
batch_size = 16
seq_len = 128
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for step in range(50):  # Adjust number of steps as needed.
        x_batch, y_batch = get_batch(batch_size, seq_len)
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        with autocast():
            if USE_HLOSS:
                p, _ = model(x_batch)
                # p: (B, seq_len, vocab_size)
                probs_flat = p.view(-1, vocab_size)
                y_flat = y_batch.view(-1)
                loss = -torch.log(probs_flat[torch.arange(probs_flat.size(0)), y_flat] + EPS).mean()
            else:
                logits, _ = model(x_batch)
                logits_flat = logits.view(-1, vocab_size)
                y_flat = y_batch.view(-1)
                loss = F.cross_entropy(logits_flat, y_flat)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        if step % 10 == 0:
            print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")
    print(f"Epoch {epoch+1} Average Loss: {total_loss/50:.4f}")

# -------------------------------------------------------------------
# Generation Example
# -------------------------------------------------------------------
model.eval()
with torch.no_grad():
    prompt = text[:50]
    context = torch.tensor(encode(prompt), dtype=torch.long)[None, :].to(device)
    generated = context
    hidden_state = None
    for _ in range(200):  # Generate 200 tokens.
        inp = generated[:, -seq_len:]
        if USE_HLOSS:
            p, hidden_state = model(inp, hidden_state)
            probs = p[:, -1, :]
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            logits, hidden_state = model(inp, hidden_state)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)
    sample = decode(generated[0].cpu().tolist())
    print("Generated Sample:\n", sample)
