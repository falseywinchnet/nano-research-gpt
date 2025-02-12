#this file contains an experimental, somewhat slow to train, model using individual RNN cells to guide transformer inputs and a shared transformer to pool outputs.
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
USE_HLOSS = True        # Use harmonic loss (via unembedding) instead of CE.
EPS = 1e-8              # For numerical stability.

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
    def forward(self, x):
        # x: (batch, seq, embed_dim) -> (seq, batch, embed_dim)
        x_t = x.transpose(0, 1)
        attn_out, _ = self.attn(x_t, x_t, x_t)
        x_t = self.ln1(x_t + self.res_scale * attn_out)
        x = x_t.transpose(0, 1)
        mlp_out = self.mlp(x)
        x = self.ln2(x + self.res_scale * mlp_out)
        return x

# -------------------------------------------------------------------
# Mini-RNN Cell for Each Transformer Head
# -------------------------------------------------------------------
# This cell is similar to the previous TransformerRNNCell but expects a 
# concatenated input of (token slice, transformer slice) where each slice 
# has dimension (embed_dim_per_head).
class MiniRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        input_dim: Dimension of the concatenated input (2 * (embed_dim_per_head))
        hidden_dim: Dimension of this mini-RNN's hidden state.
        """
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.h2h = nn.Linear(hidden_dim, hidden_dim)
        self.activation = SelfScalableTanh() if USE_STAN else nn.Tanh()
        if USE_DIFF_XOR:
            self.diff_xor = DifferentiableXORLayer(hidden_dim)
        else:
            self.diff_xor = None

    def forward(self, token_slice, transformer_slice, h_prev):
        # token_slice and transformer_slice: (B, embed_dim_per_head)
        # Concatenate the two slices.
        combined = torch.cat([token_slice, transformer_slice], dim=-1)
        h_candidate = self.input_proj(combined) + self.h2h(h_prev)
        h_new = self.activation(h_candidate)
        if self.diff_xor is not None:
            h_new = h_new + self.diff_xor(h_new)
        return h_new

# -------------------------------------------------------------------
# New Model: Multi-Head Mini-RNN with Second Transformer Stage
# -------------------------------------------------------------------
class MultiHeadMiniRNNTransformerModel(nn.Module):
    def __init__(self, vocab_size, 
                 embed_dim=128, 
                 transformer_layers=4, 
                 transformer_heads=4, 
                 rnn_hidden_size=256, 
                 second_transformer_layers=2,
                 seq_len=128, 
                 use_hloss=USE_HLOSS):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim          # Total embedding dimension
        self.seq_len = seq_len
        self.use_hloss = use_hloss
        self.num_heads = transformer_heads

        assert embed_dim % transformer_heads == 0, "embed_dim must be divisible by transformer_heads"
        # We'll split the embed_dim into num_heads segments.
        self.embed_dim_per_head = embed_dim // transformer_heads

        # For the mini-RNNs, we assume that the overall RNN hidden size is divisible by num_heads.
        assert rnn_hidden_size % transformer_heads == 0, "rnn_hidden_size must be divisible by transformer_heads"
        self.mini_rnn_hidden_dim = rnn_hidden_size // transformer_heads

        # Token embedding and positional embedding.
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, embed_dim))

        # First transformer stack (global context).
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, transformer_heads, dropout=0.0, res_scale=1/math.sqrt(transformer_layers))
            for _ in range(transformer_layers)
        ])

        # Create a mini-RNN cell for each head.
        self.mini_rnn_cells = nn.ModuleList([
            MiniRNNCell(input_dim=2 * self.embed_dim_per_head, hidden_dim=self.mini_rnn_hidden_dim)
            for _ in range(transformer_heads)
        ])

        # Second transformer stack to combine the concatenated mini-RNN outputs.
        # This transformer operates on vectors of size rnn_hidden_size.
        self.second_transformer_blocks = nn.ModuleList([
            TransformerBlock(rnn_hidden_size, transformer_heads, dropout=0.0, res_scale=1/math.sqrt(second_transformer_layers))
            for _ in range(second_transformer_layers)
        ])

        # Output head: either harmonic unembedding or a standard linear layer.
        if self.use_hloss:
            self.unembedding = nn.Parameter(torch.randn(rnn_hidden_size, vocab_size))
            nn.init.kaiming_uniform_(self.unembedding, a=math.sqrt(5))
        else:
            self.fc = nn.Linear(rnn_hidden_size, vocab_size)

    def forward(self, x, hidden_states=None):
        """
        x: LongTensor of shape (B, seq_len)
        hidden_states: list of length num_heads, each of shape (B, mini_rnn_hidden_dim);
                        if None, they are initialized to zeros.
        """
        B, T = x.size()
        # Embed tokens and add positional embeddings.
        token_emb = self.token_embedding(x)           # (B, T, embed_dim)
        pos_emb = self.pos_embedding[:, :T, :]           # (1, T, embed_dim)
        transformer_input = token_emb + pos_emb          # (B, T, embed_dim)

        # Pass through the first transformer stack.
        h_transformer = transformer_input
        for block in self.transformer_blocks:
            h_transformer = block(h_transformer)         # (B, T, embed_dim)

        # Initialize mini-RNN hidden states for each head if not provided.
        if hidden_states is None:
            hidden_states = [torch.zeros(B, self.mini_rnn_hidden_dim, device=x.device) 
                             for _ in range(self.num_heads)]

        # Process each timestep with the mini-RNN cells.
        mini_rnn_outputs = []
        for t in range(T):
            # For each head, take the corresponding slice from the token and transformer outputs.
            head_outputs = []
            for head in range(self.num_heads):
                start = head * self.embed_dim_per_head
                end = (head + 1) * self.embed_dim_per_head
                token_slice = token_emb[:, t, start:end]       # (B, embed_dim_per_head)
                trans_slice = h_transformer[:, t, start:end]     # (B, embed_dim_per_head)
                # Update the mini-RNN cell for this head.
                h_new = self.mini_rnn_cells[head](token_slice, trans_slice, hidden_states[head])
                hidden_states[head] = h_new
                head_outputs.append(h_new)
            # Concatenate the outputs from all mini-RNNs along the feature dimension.
            combined = torch.cat(head_outputs, dim=-1)         # (B, rnn_hidden_size)
            mini_rnn_outputs.append(combined)
        # Stack outputs along time dimension.
        mini_rnn_outputs = torch.stack(mini_rnn_outputs, dim=1)  # (B, T, rnn_hidden_size)

        # Pass the combined mini-RNN outputs through the second transformer stack.
        h_second = mini_rnn_outputs
        for block in self.second_transformer_blocks:
            h_second = block(h_second)  # (B, T, rnn_hidden_size)

        # Output head.
        if self.use_hloss:
            # Compute harmonic loss probabilities.
            var_x = torch.var(h_second, dim=[0, 1], keepdim=True) + EPS
            outputs_exp = h_second.unsqueeze(2)  # (B, T, 1, rnn_hidden_size)
            W_T_exp = self.unembedding.t().unsqueeze(0).unsqueeze(0)  # (1, 1, vocab_size, rnn_hidden_size)
            delta = outputs_exp - W_T_exp
            mahalanobis_d = torch.sqrt(torch.sum(delta**2 / var_x, dim=-1))
            scale_factor = mahalanobis_d.mean().item()
            scale_factor = max(scale_factor, EPS)
            mahalanobis_d_scaled = mahalanobis_d / scale_factor
            mahalanobis_d_clamped = torch.clamp(mahalanobis_d_scaled, min=1e-6)
            harmonic_exponent = int(math.sqrt(h_second.size(-1)))
            log_inv_dn = -harmonic_exponent * torch.log(mahalanobis_d_clamped + EPS)
            log_sum = torch.logsumexp(log_inv_dn, dim=-1, keepdim=True)
            log_p = log_inv_dn - log_sum
            p = torch.exp(log_p)
            return p, hidden_states
        else:
            logits = self.fc(h_second)  # (B, T, vocab_size)
            return logits, hidden_states

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
model = MultiHeadMiniRNNTransformerModel(vocab_size, embed_dim=160, transformer_layers=8, transformer_heads=8, 
                                         rnn_hidden_size=256, second_transformer_layers=2, seq_len=128, 
                                         use_hloss=USE_HLOSS).to(device)
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
                # p: (B, T, vocab_size)
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
    hidden_states = None
    for _ in range(200):  # Generate 200 tokens.
        inp = generated[:, -seq_len:]
        if USE_HLOSS:
            p, hidden_states = model(inp, hidden_states)
            probs = p[:, -1, :]
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            logits, hidden_states = model(inp, hidden_states)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)
    sample = decode(generated[0].cpu().tolist())
    print("Generated Sample:\n", sample)
