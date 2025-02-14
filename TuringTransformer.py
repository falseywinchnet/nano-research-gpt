#copyright joshuah rainstar 2025
#licensed under christian freeware license
#this version adds a  cache because generation on previous version is very slow
#but doesnt seem to be faster

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import math
import requests

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------------------------------------------------
# Custom Activation
# ---------------------------------------------------
class SelfScalableTanh(nn.Module):
    def __init__(self, init_scale=0.1, max_scale=0.12):
        super().__init__()
        # Learned scale parameter
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))

    def forward(self, x):
        # "Scaled Tanh"
        return torch.tanh(x) + self.scale * torch.tanh(x)


# ---------------------------------------------------
# Differentiable XOR
# ---------------------------------------------------
class DifferentiableXORLayer(nn.Module):
    """
    Splits the incoming embedding in half, and does a
    sigmoid-based XOR-like transformation.
    """
    def __init__(self, embed_dim):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even for XOR."
        self.embed_dim = embed_dim
        self.proj = nn.Linear(embed_dim // 2, embed_dim)

    def forward(self, x):
        d = self.embed_dim // 2
        x1, x2 = x[..., :d], x[..., d:]
        a = torch.sigmoid(x1)
        b = torch.sigmoid(x2)
        # approximate XOR = a + b - 2ab
        xor_out = 0.5 * (a + b - 2 * a * b)  # scaled by 0.5
        out = self.proj(xor_out)
        return out


# ---------------------------------------------------
# Harmonic Distance => Probability
# ---------------------------------------------------
def harmonic_unembedding(hidden_states, unembedding, eps=1e-8):
    """
    hidden_states: (B, seq_len, D)
    unembedding:   (D, vocab_size)  learnable parameter
    returns: p of shape (B, seq_len, vocab_size)

    You had done something like:
      distances = sqrt(sum((x - w)^2))
      log_inv_dn = - H * log(distances)
      log_p = log_inv_dn - logsumexp(...)
      p = exp(log_p)
    Where H might be int(sqrt(D)).
    """
    B, S, D = hidden_states.shape
    vocab_size = unembedding.shape[1]

    # Expand hidden => (B, S, 1, D)
    x_exp = hidden_states.unsqueeze(2)
    # Expand unembedding => (1,1,vocab_size,D)
    w_exp = unembedding.t().unsqueeze(0).unsqueeze(0)  # (1,1,V,D)
    # L2 distance
    distances = torch.sqrt(torch.sum((x_exp - w_exp)**2, dim=-1) + eps)
    harmonic_exponent = int(math.sqrt(D))

    log_inv_dn = -harmonic_exponent * torch.log(distances + eps)
    log_sum = torch.logsumexp(log_inv_dn, dim=-1, keepdim=True)
    log_p = log_inv_dn - log_sum
    p = torch.exp(log_p)
    return p

class TapeHead(nn.Module):
    """
    A single head that attends over chunked embeddings of size chunk_size.
    """
    def __init__(self, embed_dim, chunk_size=2, num_heads=1, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.chunk_size = chunk_size
        self.num_heads = num_heads
        # Project c * D => D to build a chunk embedding.
        self.chunk_proj = nn.Linear(chunk_size * embed_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, token_emb, cache=None):
        """
        token_emb: (B, S, D)
        cache: optional dict containing cached computations.
           Expected keys:
              - "chunk_tensor": previously computed chunk embeddings (B, L, D)
              - "cached_length": integer L, the number of tokens previously computed.
        Returns (B, S, D)
        """
        B, S, D = token_emb.shape
        c = self.chunk_size

        if cache is None or "chunk_tensor" not in cache:
            # No cache: compute the entire sequence.
            chunk_embs = []
            for i in range(S):
                start = i
                end = min(i + c, S)
                window = token_emb[:, start:end, :]  # (B, <= c, D)
                needed = c - (end - start)
                if needed > 0:
                    pad = torch.zeros((B, needed, D), device=token_emb.device)
                    window = torch.cat([window, pad], dim=1)
                window_flat = window.view(B, c * D)
                chunk_repr = self.chunk_proj(window_flat)  # (B, D)
                chunk_embs.append(chunk_repr.unsqueeze(1))
            chunk_tensor = torch.cat(chunk_embs, dim=1)
        else:
            # Use the cached chunk embeddings.
            cached_length = cache["cached_length"]  # previously computed sequence length
            # Determine from where we need to re-compute.
            # We must recompute for tokens whose chunk window could have changed.
            start_idx = max(cached_length - (c - 1), 0)
            new_chunk_embs = []
            for i in range(start_idx, S):
                start = i
                end = min(i + c, S)
                window = token_emb[:, start:end, :]  # (B, <= c, D)
                needed = c - (end - start)
                if needed > 0:
                    pad = torch.zeros((B, needed, D), device=token_emb.device)
                    window = torch.cat([window, pad], dim=1)
                window_flat = window.view(B, c * D)
                chunk_repr = self.chunk_proj(window_flat)  # (B, D)
                new_chunk_embs.append(chunk_repr.unsqueeze(1))
            new_chunk_tensor = torch.cat(new_chunk_embs, dim=1)
            if start_idx > 0:
                # Keep cached values for positions that are not affected.
                chunk_tensor = torch.cat([cache["chunk_tensor"][:, :start_idx, :], new_chunk_tensor], dim=1)
            else:
                chunk_tensor = new_chunk_tensor
            # Update the cache.
            cache["chunk_tensor"] = chunk_tensor.detach()  # detach to prevent backprop through cache
            cache["cached_length"] = S

        # Self-attention among chunk embeddings.
        out, _ = self.attn(chunk_tensor, chunk_tensor, chunk_tensor)
        out = self.ln(chunk_tensor + out)
        return out


class MultiScaleTapeAttention(nn.Module):
    """
    Combines multiple TapeHeads of different chunk sizes (including c=1 for token-level).
    """
    def __init__(self, embed_dim, chunk_sizes=(1, 2, 4), num_heads=2, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList([
            TapeHead(embed_dim, c, num_heads=num_heads, dropout=dropout)
            for c in chunk_sizes
        ])
        # Fuse the outputs from each head.
        total_dim = len(chunk_sizes) * embed_dim
        self.fuse = nn.Linear(total_dim, embed_dim)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x, cache=None):
        # x: (B, S, D)
        out_heads = []
        # If cache is provided, assume a separate cache per head.
        if cache is None:
            head_caches = [None] * len(self.heads)
        else:
            head_caches = [cache.get(f"head_{i}", None) for i in range(len(self.heads))]
        for i, head in enumerate(self.heads):
            head_out = head(x, cache=head_caches[i])
            out_heads.append(head_out)
            # Update the cache for each head.
            if cache is not None:
                cache[f"head_{i}"] = head_caches[i]
        cat_out = torch.cat(out_heads, dim=-1)  # (B, S, total_dim)
        fused = self.fuse(cat_out)
        fused = self.ln(fused)
        return fused


class MultiScaleXORTransformerBlock(nn.Module):
    """
    A single block that:
      - Applies multi-scale "Tape" self-attention
      - Then an MLP with SelfScalableTanh
      - Then a DifferentiableXOR gating
      - Then LN + residual
    """
    def __init__(self, embed_dim, chunk_sizes=(1, 2, 4), num_heads=2, dropout=0.1, res_scale=1.0):
        super().__init__()
        self.attn = MultiScaleTapeAttention(
            embed_dim, chunk_sizes=chunk_sizes, num_heads=num_heads, dropout=dropout
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.activation = SelfScalableTanh()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            self.activation,
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.diff_xor = DifferentiableXORLayer(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.res_scale = res_scale

    def forward(self, x, cache=None):
        # x: (B, S, D)
        attn_cache = None if cache is None else cache.get("attn", None)
        attn_out = self.attn(x, cache=attn_cache)
        if cache is not None:
            cache["attn"] = attn_cache
        x = self.ln1(x + self.res_scale * attn_out)
        mlp_out = self.mlp(x)
        xor_features = self.diff_xor(mlp_out)
        mlp_out = mlp_out + xor_features
        x = self.ln2(x + self.res_scale * mlp_out)
        return x


class MultiScaleTapeModel(nn.Module):
    """
    End-to-end model:
      - token + positional embeddings
      - N "MultiScaleXORTransformerBlock" layers
      - final harmonic unembedding to produce p(logits)
    """
    def __init__(self,
                 vocab_size,
                 seq_len=128,
                 embed_dim=128,
                 num_layers=4,
                 chunk_sizes=(1, 2, 4),
                 num_heads=2,
                 dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        self.blocks = nn.ModuleList([
            MultiScaleXORTransformerBlock(
                embed_dim=embed_dim,
                chunk_sizes=chunk_sizes,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        # Final harmonic unembedding.
        self.unembeddings = nn.ParameterList([
            nn.Parameter(torch.randn(embed_dim, vocab_size))
            for _ in range(num_heads)
        ])
        for unembedding in self.unembeddings:
            nn.init.kaiming_uniform_(unembedding, a=math.sqrt(5))

    def forward(self, x, cache=None):
        """
        x: (B, S)
        cache: optional dict to hold cached activations for incremental generation.
               We assume keys like "block_0", "block_1", ..., for each transformer block.
        """
        B, S = x.shape
        assert S <= self.seq_len, "Input seq too long for pos_emb"
        tok_emb = self.token_emb(x)          # (B, S, D)
        pos_slice = self.pos_emb[:, :S, :]     # (1, S, D)
        h = tok_emb + pos_slice                # (B, S, D)
        if cache is None:
            block_caches = [None] * len(self.blocks)
        else:
            block_caches = [cache.get(f"block_{i}", None) for i in range(len(self.blocks))]
        for i, block in enumerate(self.blocks):
            h = block(h, cache=block_caches[i])
            if cache is not None:
                cache[f"block_{i}"] = block_caches[i]
        # Unembedding: aggregate outputs from each unembedding head.
        p_all = []
        for unembedding in self.unembeddings:
            p_all.append(harmonic_unembedding(h, unembedding))  # (B, S, V)
        p = torch.stack(p_all, dim=0).mean(dim=0)  # (B, S, V)
        return p

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiScaleTapeModel(
    vocab_size=vocab_size,  # example
    seq_len=200,
    embed_dim=64,
    num_layers=8,
    chunk_sizes=(1,2,4,6),
    num_heads=2,
    dropout=0.0
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=6e-4)

scaler = GradScaler()

# For standard loss, we use cross-entropy; for harmonic loss we compute negative log probability manually.

num_epochs = 10
batch_size = 10
seq_len = 128


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
            loss = -torch.log(torch.gather(p, -1, y_batch.unsqueeze(-1)) + 1e-8).squeeze(-1).mean()

        main_loss = loss.detach()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += main_loss
        losses.append(main_loss.cpu())
        if step % 1 == 0:
            print(f"Epoch {epoch+1}, Step {step}, Loss: {main_loss:.4f}")

    print(f"Epoch {epoch+1} Average Loss: {total_loss/10:.4f}")

# ====================================================
# Evaluation: Text Generation
# ====================================================

    # Decay rate (tune this to control how fast the bonus decays)
model.eval()
global_cache = {}  # persistent cache across generation steps
with torch.no_grad():
    prompt = text[:50]
    context = torch.tensor(encode(prompt), dtype=torch.long)[None, :].to(device)
    generated = context
    for _ in range(2000):  # e.g. generate 2000 tokens incrementally
        # Only feed in the last seq_len tokens.
        inp = generated[:, -seq_len:]
        # Pass in the cache; the model will update it in-place.
        p = model(inp, cache=global_cache)
        last_token_probs = p[:, -1, :]  # (B, vocab_size)
        next_token = torch.multinomial(last_token_probs, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)
    sample = decode(generated[0].cpu().tolist())
    print("Generated Sample:\n", sample)






