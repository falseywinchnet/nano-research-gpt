#this file contains an experimental, somewhat fast to train, but slow to inference char rnn model containing some experimental features
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
USE_STAN = True          # If True, use Self-Scalable Tanh in the RNN cell; else use standard Tanh.
USE_DIFF_XOR = True      # If True, inject a differentiable XOR module into each RNN cell.
USE_HLOSS = True         # If True, use harmonic loss (via an unembedding layer) instead of standard cross-entropy.
apply_logistic_scaling = True   # If True, stretch the logits for more learnin
EPS = 1e-8                # Small constant for numerical stability.

# For reproducibility.
torch.manual_seed(42)

# ====================================================
# Self-Scalable Tanh (STAN)
# ====================================================
class SelfScalableTanh(nn.Module):
    def __init__(self, init_scale=0.1, max_scale=0.12):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))

    def forward(self, x):
        return torch.tanh(x) + self.scale * torch.tanh(x)

# ====================================================
# Differentiable XOR Module
# ====================================================
class DifferentiableXORLayer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        assert embed_dim % 2 == 0, "Embedding dimension must be even for the XOR module."
        self.embed_dim = embed_dim
        self.proj = nn.Linear(embed_dim // 2, embed_dim)

    def forward(self, x):
        # Split the last dimension in half.
        d = self.embed_dim // 2
        x1, x2 = x[..., :d], x[..., d:]
        a = torch.sigmoid(x1)
        b = torch.sigmoid(x2)
        # Soft-XOR: note the scaling by 0.5 (to reduce the overall magnitude)
        xor_out = 0.5 * (a + b - 2 * a * b)
        out = self.proj(xor_out)
        return out

# ====================================================
# Custom Char-RNN Cell
# ====================================================
class CharRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Compute candidate hidden state from input and previous hidden state.
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        # Choose activation: either SelfScalableTanh or standard Tanh.
        if USE_STAN:
            self.activation = SelfScalableTanh()
        else:
            self.activation = nn.Tanh()
        # Optionally inject differentiable XOR.
        if USE_DIFF_XOR:
            # hidden_size must be even.
            assert hidden_size % 2 == 0, "hidden_size must be even for differentiable XOR."
            self.diff_xor = DifferentiableXORLayer(hidden_size)
        else:
            self.diff_xor = None

    def forward(self, x, h_prev):
        # x: (batch, input_size)
        # h_prev: (batch, hidden_size)
        h_candidate = self.i2h(x) + self.h2h(h_prev)
        h_new = self.activation(h_candidate)
        if self.diff_xor is not None:
            h_new = h_new + self.diff_xor(h_new)
        return h_new

# ====================================================
# Char-RNN Model
# ====================================================
class CharRNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_size=256, num_layers=2, placeholder_idx=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.placeholder_idx = placeholder_idx

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Create a stack of RNN cells.
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            input_size = embed_dim if layer == 0 else hidden_size
            self.cells.append(CharRNNCell(input_size, hidden_size))

        # Final projection: choose between harmonic unembedding vs. a standard linear layer.
        if USE_HLOSS:
            self.unembedding = nn.Parameter(torch.randn(hidden_size, vocab_size))
            nn.init.kaiming_uniform_(self.unembedding, a=math.sqrt(5))
        else:
            self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden_state=None):
        """
        x: LongTensor of shape (batch, seq_len)
        hidden_state: list of hidden states (one per layer); if None, they are initialized to zeros.
        Returns:
          - logits (or probabilities) of shape (batch, seq_len, vocab_size)
          - the final hidden_state (list of tensors, one per layer)
        """
        batch_size, seq_len = x.size()
        x_embed = self.embedding(x)  # (batch, seq_len, embed_dim)
        
        # Initialize hidden states for each layer if not provided.
        if hidden_state is None:
            hidden_state = [torch.zeros(batch_size, self.hidden_size, device=x.device)
                            for _ in range(self.num_layers)]
        
        outputs = []
        # Process the input sequence one timestep at a time.
        for t in range(seq_len):
            input_t = x_embed[:, t, :]  # (batch, embed_dim) for first layer.
            new_hidden = []
            for layer in range(self.num_layers):
                cell = self.cells[layer]
                h_prev = hidden_state[layer]
                h_new = cell(input_t, h_prev)
                new_hidden.append(h_new)
                input_t = h_new  # Input for next layer.
            hidden_state = new_hidden
            outputs.append(input_t)
        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden_size)

        if USE_HLOSS:
            # --- Harmonic Loss Computation ---
            var_x = torch.var(outputs, dim=[0, 1], keepdim=True) + EPS  # variance across batch and time.
            outputs_exp = outputs.unsqueeze(2)  # (B, seq_len, 1, hidden_size)
            W_T_exp = self.unembedding.t().unsqueeze(0).unsqueeze(0)  # (1, 1, vocab_size, hidden_size)
            delta = outputs_exp - W_T_exp
            mahalanobis_d = torch.sqrt(torch.sum(delta**2 / var_x, dim=-1))
            
            scale_factor = mahalanobis_d.mean().item()
            scale_factor = max(scale_factor, EPS)
            mahalanobis_d_scaled = mahalanobis_d / scale_factor
            mahalanobis_d_clamped = torch.clamp(mahalanobis_d_scaled, min=1e-6)
            harmonic_exponent = int(math.sqrt(self.hidden_size))
            log_inv_dn = -harmonic_exponent * torch.log(mahalanobis_d_clamped + EPS)
            log_sum = torch.logsumexp(log_inv_dn, dim=-1, keepdim=True)
            log_p = log_inv_dn - log_sum
            p = torch.exp(log_p)
            return p, hidden_state
        else:
            # --- Standard LM head ---
            logits = self.fc(outputs)  # (batch, seq_len, vocab_size)
            return logits, hidden_state

# ====================================================
# Data Preparation (Shakespeare)
# ====================================================
def load_shakespeare_text():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url).text
    return text

text = load_shakespeare_text()
chars = sorted(list(set(text)))
if USE_PLACEHOLDER:
    placeholder_token = "<PLH>"
    chars.append(placeholder_token)
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

def get_batch(batch_size, seq_len):
    # Randomly sample starting indices.
    ix = torch.randint(0, data.size(0) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x, y

# ====================================================
# Training Setup
# ====================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CharRNNModel(vocab_size, embed_dim=128, hidden_size=256, num_layers=3).to(device)
optimizer = Wolf(model.parameters(), lr=0.6)
scaler = GradScaler()

# For standard loss we use cross-entropy.
criterion_ce = nn.CrossEntropyLoss()

num_epochs = 2
batch_size = 16
seq_len = 150

# ====================================================
# Training Loop
# ====================================================
losses = []
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for step in range(100):  # Adjust number of steps per epoch as needed.
        x_batch, y_batch = get_batch(batch_size, seq_len)
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        with autocast():
            if USE_HLOSS:
                # The model returns probabilities (via harmonic loss).
                p, _ = model(x_batch)
                # p: (batch, seq_len, vocab_size)
                probs_flat = p.view(-1, vocab_size)

                y_flat = y_batch.view(-1)
                per_token_loss = -torch.log(probs_flat[torch.arange(probs_flat.size(0)), y_flat] + EPS)
                loss = per_token_loss.mean()
            else:
                # Standard LM head path.
                logits, _ = model(x_batch)
                logits_flat = logits.view(-1, vocab_size)
                y_flat = y_batch.view(-1)
                loss = criterion_ce(logits_flat, y_flat)

            
            main_loss = loss.detach()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += main_loss.item()
        losses.append(main_loss.cpu().item())
        if step % 20 == 0:
            print(f"Epoch {epoch+1}, Step {step}, Loss: {main_loss.item():.4f}")
    print(f"Epoch {epoch+1} Average Loss: {total_loss/100:.4f}")

# ====================================================
# Evaluation: Text Generation
# ====================================================
model.eval()
with torch.no_grad():
    prompt = text[:50]
    context = torch.tensor(encode(prompt), dtype=torch.long)[None, :].to(device)
    generated = context

    for i in range(200):  # Generate 200 tokens.
        inp = generated[:, -seq_len:]
        if USE_HLOSS:
            # When using harmonic loss, the model returns (p, hidden_state).
            p, _ = model(inp)
            # Extract the probabilities for the last token.
            p = p[:, -1, :]

        else:
            logits, _ = model(inp)
            logits = logits[:, -1, :]
            p = F.softmax(logits, dim=-1)

        next_token = torch.multinomial(p, num_samples=1)            
        generated = torch.cat((generated, next_token), dim=1)

    sample = decode(generated[0].cpu().tolist())
    print("Generated Sample:\n", sample)



