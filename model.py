import torch
from torch import nn
import math
from einops import rearrange, einsum

class Linear(nn.Module):
    def __init__ (self, in_features : int, out_features : int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        tensor = torch.empty((out_features, in_features), device = device, dtype = dtype)
        self.weight = nn.Parameter(tensor)
        scale = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean = 0.0, std = scale, a = -3 * scale, b = 3 * scale)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

class Embedding(nn.Module):
    def __init__ (self, num_embeddings : int, embedding_dim : int, device : torch.device | None = None, dtype : torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        tensor = torch.empty((num_embeddings, embedding_dim), device = device, dtype = dtype)
        self.weight = nn.Parameter(tensor)
        nn.init.trunc_normal_(self.weight, mean = 0.0, std = 1, a = -3, b = 3)

    def forward(self, token_ids : torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model : int, eps : float = 1e-5, device = None, dtype = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device = device, dtype = dtype))

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        x_float = x.to(torch.float32)
        # mean_square = mean(x^2)
        mean_square = x_float.pow(2).mean(dim = -1, keepdim = True)
        # 1 / sqrt(mean_square + eps)
        inverse_rms = torch.rsqrt(mean_square + self.eps)
        # x / sqrt(mean_square + eps)
        normalized = x_float * inverse_rms
        # x / sqrt(mean_square + eps) * g
        return self.weight * normalized.to(original_dtype)

class SwiGLU(nn.Module):
    def __init__ (self, d_model, d_ff : int | None = None, device = None, dtype = None):
        super().__init__()
        if d_ff is None:
            d_ff = math.ceil((8 * d_model / 3) / 64) * 64
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        first_branch = self.w1(x)
        # SiLu = x * sigmoid(x)
        silu = torch.sigmoid(first_branch) * first_branch
        third_branch = self.w3(x)
        # Element-wise multiplication
        result = silu * third_branch
        second_branch = self.w2(result)
        return second_branch

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta : float, d_k : int, max_seq_len : int, device = None):
        super().__init__()
        assert d_k % 2 == 0
        # 2i / d_k
        freq_idx = torch.arange(0, d_k, 2, device = device) / d_k
        # θ^(-2i/d_k)
        inv_freq = theta ** (-freq_idx)
        positions = torch.arange(max_seq_len, device = device)
        # m * θ^(-2i/d_k)
        angles = positions[:, None] * inv_freq[None, :]
        self.register_buffer("cos_cache", torch.cos(angles), persistent = False)
        self.register_buffer("sin_cache", torch.sin(angles), persistent = False)

    def forward(self, x : torch.Tensor, token_positions : torch.Tensor) -> torch.Tensor:
        cos = self.cos_cache[token_positions].to(dtype = x.dtype)
        sin = self.sin_cache[token_positions].to(dtype = x.dtype)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)

def scaled_dot_product_attention(Q, K, V, mask = None):
    K_value = rearrange(K, "... keys d_k -> ... d_k keys")
    scores = Q @ K_value / math.sqrt(Q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    probabilities = torch.softmax(scores, dim = -1)

    combined = einsum(
        probabilities, V,
        "... query key, ... key width -> ... query width"
    )
    return combined

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model : int, num_heads : int, theta : float | None = None, max_seq_len : int | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.d_head = d_model // num_heads
        self.rope = None
        if theta is not None:
            assert max_seq_len is not None
            self.rope = RotaryPositionalEmbedding(theta, self.d_head, max_seq_len)
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)

    def forward(self, x, token_positions : torch.Tensor | None = None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # d_model(hidden size) = head * dim
        q = rearrange(
            q, "... seq (heads dim) -> ... heads seq dim",
            heads = self.num_heads
            )
        k = rearrange(
            k, "... seq (heads dim) -> ... heads seq dim",
            heads = self.num_heads
            )
        v = rearrange(
            v, "... seq (heads dim) -> ... heads seq dim",
            heads = self.num_heads
            )
        seq_len = x.shape[-2]
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device = x.device)
            rope_positions = token_positions.unsqueeze(-2)
            q = self.rope(q, rope_positions)
            k = self.rope(k, rope_positions)
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype = torch.bool, device = x.device)
        )
        attention_output = scaled_dot_product_attention(q, k, v, causal_mask)
        attention_output = rearrange(
            attention_output, "... heads seq dim -> ... seq (heads dim)"
        )
        output = self.o_proj(attention_output)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x):
        y = x + self.attn(self.ln1(x))
        output = y + self.ffn(self.ln2(y))
        return output
