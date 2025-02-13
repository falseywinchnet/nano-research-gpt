#this file contains an experimental, somewhat fast to train, but not yet generating meaningful outputs hybrid
#gpt and rnn model containing some experimental features
#copyright joshuah rainstar 2025
#licensed under christian freeware license
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import math
import requests
# -------------------------------------------------------------------
# Constants and toggles
# -------------------------------------------------------------------
EPS = 1e-8
USE_HLOSS = True
USE_STAN = True
USE_DIFF_XOR = True

# -------------------------------------------------------------------
# Auxiliary Modules
# -------------------------------------------------------------------

class SelfScalableTanh(nn.Module):
    def __init__(self, init_scale=0.1, max_scale=0.12):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        self.max_scale = max_scale

    def forward(self, x):
        # Hard-clamp the learnable scale so it doesn't exceed max_scale
        clipped_scale = torch.clamp(self.scale, 0.0, self.max_scale)
        return torch.tanh(x) + clipped_scale * torch.tanh(x)

class DifferentiableXORLayer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        assert embed_dim % 2 == 0, "Embedding dimension must be even for XOR"
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

# A custom RNN cell for each head that integrates STAN and differentiable XOR.
class HeadRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)  # ✅ Ensure input is projected to hidden_dim
        self.h2h = nn.Linear(hidden_dim, hidden_dim)  # ✅ Keep consistent hidden state
        self.activation = SelfScalableTanh() if USE_STAN else nn.Tanh()
        if USE_DIFF_XOR:
            self.diff_xor = DifferentiableXORLayer(hidden_dim)
        else:
            self.diff_xor = None
    def forward(self, inp, h_prev):
        h_prev = self.h2h(h_prev)  # ✅ Ensure `h_prev` matches the projected `inp`
        h_candidate = self.input_proj(inp) + h_prev  # ✅ Now they match
        h_new = self.activation(h_candidate)
        if self.diff_xor is not None:
            h_new = h_new + self.diff_xor(h_new)
        return h_new


class FastRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.activation = SelfScalableTanh()
        self.diff_xor = DifferentiableXORLayer(hidden_dim) if hidden_dim % 2 == 0 else None
        self.hidden_dim = hidden_dim
        
        # Persistent state: Learned hidden state stored during training
        self.persistent_state = nn.Parameter(torch.zeros(1, hidden_dim), requires_grad=True)
        self.internal_state = nn.Parameter(torch.zeros(1, hidden_dim), requires_grad=True)
        
        self.etcerta = 0.367879441  # Constant used in update rule
        self.et = 1 - self.etcerta
    
    def forward(self, inp, h_prev=None):
        if h_prev is None:
            h_prev = self.persistent_state.expand(inp.size(0), -1)  # Use learned persistent state

        h_candidate = self.input_proj(inp)
        h_prev_transformed = self.hidden_proj(h_prev)
        h_new = self.activation(h_candidate + h_prev_transformed)
        if self.diff_xor is not None:
            h_new = h_new + self.diff_xor(h_new)
        
        # Apply internal state update rule
        update = self.internal_state * self.et + h_new * self.etcerta
        self.internal_state.data = self.internal_state * self.et + update * self.etcerta

        # Update persistent state only during evaluation
        if not self.training:
            self.persistent_state.data = torch.where(
                torch.sign(update) * torch.sign(h_new) > 0,
                self.persistent_state.data - update,
                self.persistent_state.data
            )
        
        return h_new




class GatedTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)  # ✅ Rename to `self_attn`
        self.ln1 = nn.LayerNorm(embed_dim)

        # Gating mechanism for feature selection
        self.gate = nn.Linear(embed_dim, embed_dim)
        self.sigmoid = nn.Sigmoid()

        # MLP with STAN activation
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            SelfScalableTanh(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # Pass `is_causal` to `self_attn` if supported by PyTorch version
        attn_out, _ = self.self_attn(x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)  
        x = self.ln1(x + attn_out)
    
        # Gating mechanism (adaptive feature selection)
        gate_values = self.sigmoid(self.gate(x))  # (B, seq_len, embed_dim)
        gated_mlp_out = gate_values * self.mlp(x)  # Element-wise feature selection
    
        # Residual connection with LayerNorm
        x = self.ln2(x + gated_mlp_out)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfScalableTanh(nn.Module):
    def __init__(self, init_scale=0.1, max_scale=0.12):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        self.max_scale = max_scale

    def forward(self, x):
        clipped_scale = torch.clamp(self.scale, 0.0, self.max_scale)
        return torch.tanh(x) + clipped_scale * torch.tanh(x)

class FastRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.diff_xor = DifferentiableXORLayer(hidden_dim) if USE_DIFF_XOR else None

        self.activation = SelfScalableTanh()
        self.hidden_dim = hidden_dim
        
        # Persistent and internal state
        self.persistent_state = nn.Parameter(torch.zeros(1, hidden_dim), requires_grad=True)

        
        self.etcerta = 0.367879441  # Decay constant
        self.et = 1 - self.etcerta
    
    def forward(self, inp, h_prev=None):
        if h_prev is None:
            h_prev = self.persistent_state.to(inp.device).expand(inp.size(0), -1)
    
        h_candidate = self.input_proj(inp)
        h_prev_transformed = self.hidden_proj(h_prev)
        h_new = self.activation(h_candidate + h_prev_transformed)
        
        if self.diff_xor is not None:
            h_new = h_new + self.diff_xor(h_new)
        
        # Ensure internal state is on the same device
        update = self.persistent_state.data * self.et + h_new * self.etcerta
        self.persistent_state.data = self.persistent_state.to(inp.device) * self.et + update * self.etcerta
    
        return h_new

        

class GatedTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.gate = nn.Linear(embed_dim, embed_dim)
        self.sigmoid = nn.Sigmoid()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            SelfScalableTanh(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x, src_mask=None, src_key_padding_mask=None, is_causal=None):
        attn_out, _ = self.self_attn(x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        x = self.ln1(x + attn_out)
        gate_values = self.sigmoid(self.gate(x))
        gated_mlp_out = gate_values * self.mlp(x)
        x = self.ln2(x + gated_mlp_out)
        return x


class CustomTransformAndRememberHead(nn.Module):
    def __init__(self, embed_dim, num_layers=4, tape_length=128):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            GatedTransformerEncoderLayer(embed_dim, num_heads=2, dropout=0.1),
            num_layers=num_layers
        )
        self.rnn = FastRNNCell(embed_dim, embed_dim)
        self.tape_length = tape_length
        self.tape = torch.zeros(tape_length, embed_dim)
        self.pointer = 0
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape  # ✅ Explicitly track shape
    
        # Ensure tape is on the correct device
        self.tape = self.tape.to(x.device)
    
        # Transformer processing (sequence length must be preserved)
        x = self.transformer(x)  # ✅ Maintains shape [batch, seq_len, embed_dim]
    
        # Retrieve the previous state from tape
        prev_state = self.tape[(self.pointer - 1) % self.tape_length].to(x.device)  # ✅ Move to same device
        prev_state = prev_state.unsqueeze(0).expand(batch_size, -1)  # ✅ Ensure batch dimension is correct
    
        # Apply RNN cell to the last token in the sequence
        new_state = self.rnn(x[:, -1, :], prev_state)  # ✅ Uses last timestep as input
    
        # Compute batch mean and store in tape (detach to prevent gradient tracking)
        batch_mean_state = new_state.mean(dim=0).detach()
        self.tape[self.pointer % self.tape_length] = batch_mean_state.to(x.device)  # ✅ Store on correct device
        self.pointer = (self.pointer + 1) % self.tape_length
    
        return x, self.tape  # ✅ Returns full sequence, keeping sequence length intact




class MultiHeadTapeTransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, tape_length=128, final_transformer_layers=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 128, embed_dim))
        self.internal_var = None
        
        self.etcerta = 0.367879441  # Decay constant
        self.et = 1 - self.etcerta
        self.heads = nn.ModuleList([CustomTransformAndRememberHead(embed_dim, num_layers=4, tape_length=tape_length) for _ in range(num_heads)])
        
        self.final_transformer = nn.TransformerEncoder(
            GatedTransformerEncoderLayer(embed_dim * num_heads, num_heads=4, dropout=0.1),
            num_layers=final_transformer_layers
        )
        
        self.unembedding = nn.Parameter(torch.randn(embed_dim * num_heads, vocab_size))
        nn.init.kaiming_uniform_(self.unembedding, a=math.sqrt(5))
    
    def forward(self, x):

            batch_size, seq_len = x.shape  # ✅ Explicitly extract sequence length
        
            # Token embedding + position embedding
            x = self.token_embedding(x) + self.pos_embedding[:, :seq_len, :]
        
            # Process each head separately
            head_outputs = []
            for head in self.heads:
                head_out, _ = head(x)  # ✅ Each head should output [batch, seq_len, embed_dim]
                head_outputs.append(head_out)
        
            # Concatenate head outputs explicitly along embedding dim
            integrated_latent = torch.cat(head_outputs, dim=-1)
        
            # Pass through final transformer
            integrated_latent = self.final_transformer(integrated_latent)
        
            # Compute Mahalanobis distance **per token** (tracking seq_len explicitly)
            # Compute variance in training mode
            if self.training:
                if self.internal_var is not None:
                    var_x = torch.var(integrated_latent, dim=0, keepdim=True).detach() 
                    update = self.internal_var * self.et + var_x * self.etcerta
                    self.internal_var = self.internal_var * self.et + update * self.etcerta
                else:
                    self.internal_var =  torch.var(integrated_latent, dim=0, keepdim=True).detach() 

            if not self.training:
                if self.internal_var is None:
                    var_x = self.internal_var  # ✅ Use stored variance instead of forcing 1e-8
                else:
                    var_x = torch.ones_like(integrated_latent, device=integrated_latent.device) * 1e-8  # Fallback

            delta = integrated_latent.unsqueeze(2) - self.unembedding.t().unsqueeze(0).unsqueeze(0)

            if integrated_latent.shape[0] == 1:  # Single-sequence mode
                var_x = torch.ones_like(integrated_latent, device=integrated_latent.device) * 1e-8  # ✅ Use small fixed variance
            else:
                var_x = torch.var(integrated_latent, dim=0, keepdim=True) + 1e-8  # ✅ Normal variance

            # Correct the variance shape for broadcasting
            var_x = var_x.unsqueeze(2)  # ✅ Now [1, 128, 1, 320]
            
            # Compute Mahalanobis distance correctly
            mahalanobis_d = torch.sqrt(torch.sum(delta ** 2 / var_x, dim=-1))  # ✅ Now properly matches dimensions
        
            # Normalize and compute harmonic loss probabilities
            scale_factor = max(mahalanobis_d.mean().item(), 1e-8)
            mahalanobis_d_scaled = mahalanobis_d / scale_factor
            mahalanobis_d_clamped = torch.clamp(mahalanobis_d_scaled, min=1e-6)
        
            harmonic_exponent = int(math.sqrt(integrated_latent.size(-1)))
            log_inv_dn = -harmonic_exponent * torch.log(mahalanobis_d_clamped + EPS)
        
            log_sum = torch.logsumexp(log_inv_dn, dim=-1, keepdim=True)
        
            log_p = log_inv_dn - log_sum
        
            p = torch.exp(log_p)
        
            return p






# Assuming get_batch, encode, decode, text, device, optimizer, etc. are defined.
# Also assuming our model (MultiHeadTapeTransformerModel) is already created and on device.
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
model =  MultiHeadTapeTransformerModel(vocab_size, 
                 embed_dim=160, 
                 num_heads=2, 
                 tape_length=128, 
                 final_transformer_layers=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=6e-4)
scaler = GradScaler()

num_epochs = 2
batch_size = 16
seq_len = 128

losses = []
for epoch in range(num_epochs):
    
    model.train()
    torch.autograd.set_detect_anomaly(True)

    total_loss = 0.0
    for step in range(10):  # Adjust the number of steps as needed.
        x_batch, y_batch = get_batch(batch_size, seq_len)
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        with autocast():

            p = model(x_batch)
            probs_flat = p.view(-1, vocab_size)
            y_flat = y_batch.view(-1)
            per_token_loss = -torch.log(probs_flat[torch.arange(probs_flat.size(0)), y_flat] + EPS)  # Avoid log(0)
            loss = per_token_loss.mean()
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

model.eval()
with torch.no_grad():
    prompt = text[:50]
    context = torch.tensor(encode(prompt), dtype=torch.long)[None, :].to(device)
    generated = context
    for _ in range(200):  # Generate 200 tokens.
        inp = generated[:, -seq_len:]
        p = model(inp)  # p: (B, seq, vocab_size)
        probs = p[:, -1, :]        
        probs = torch.clamp(probs, min=1e-6, max=1.0)  # Remove negatives
        probs = probs / probs.sum(dim=-1, keepdim=True)  # Normalize
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)
    sample = decode(generated[0].cpu().tolist())
    print("Generated Sample:\n", sample)
