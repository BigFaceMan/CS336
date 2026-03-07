import torch
import math
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import functional as F, init
from tools.test_frame import test_log


# func -----------------------------------------------------------------------------------------------------------------------------
def silu(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x) * x


def softmax(x: torch.Tensor) -> torch.Tensor:
    x = x - x.max(dim=-1, keepdim=True).values
    exp_x = torch.exp(x)
    inv = exp_x.sum(dim=-1, keepdim=True)
    return exp_x / inv


def scaled_dot_product_attention(q, k, v, mask=None):
    d_model = q.shape[-1]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_model)
    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.inf)
    attn = softmax(scores)
    return attn @ v


# model -----------------------------------------------------------------------------------------------------------------------------
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **self.factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        sigma = 2 / (self.in_features + self.out_features)
        std = math.sqrt(sigma)
        init.trunc_normal_(self.weight, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, **self.factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        sigma = 1
        std = math.sqrt(sigma)
        init.trunc_normal_(self.weight, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedding = self.weight[token_ids]
        return embedding


class RMSNorm(nn.Module):
    """
    rmsnorm(xi) = xi / rms(x) * gi
    rms(x) 均方根
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.ones(d_model, **self.factory_kwargs))

    def _rms(self, x):
        """
        x: (bs, seq_len, d_model)

        """
        return ((x**2).mean(dim=-1) + self.eps).sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / self._rms(x).unsqueeze(-1) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.w1 = Linear(d_model, d_ff, **self.factory_kwargs)
        self.w2 = Linear(d_ff, d_model, **self.factory_kwargs)
        self.w3 = Linear(d_model, d_ff, **self.factory_kwargs)

    def reset_parameters(self):
        self.w1.reset_parameters()
        self.w2.reset_parameters()
        self.w3.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        silu_out = silu(self.w1(x))
        w3_out = self.w3(x)
        element_mul = silu_out * w3_out
        return self.w2(element_mul)


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        device = device if device is not None else torch.device("cpu")
        self.factory_kwargs = {"device": device}
        self.theta = theta
        self.d_k = d_k
        inv_freq = theta ** (-2 * torch.arange(0, d_k // 2, device=device) / d_k)
        pos = torch.arange(0, max_seq_len, device=device).unsqueeze(1)
        freqs = pos * inv_freq
        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        R@X.T.T
        X.T@R.T
        (seq_len, d_model, de_model)
        (bs, seq_len, d_model, 1)
        """
        cos_val = self.cos[token_positions]
        sin_val = self.sin[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        out = torch.empty_like(x)

        out[..., 0::2] = x_even * cos_val - x_odd * sin_val
        out[..., 1::2] = x_odd * cos_val + x_even * sin_val

        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RoPE = None, device=None, dtype=None):
        super().__init__()
        if d_model % num_heads != 0:
            raise Exception("维度不匹配")

        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.q_proj = Linear(d_model, d_model, **self.factory_kwargs)
        self.k_proj = Linear(d_model, d_model, **self.factory_kwargs)
        self.v_proj = Linear(d_model, d_model, **self.factory_kwargs)
        self.output_proj = Linear(d_model, d_model, **self.factory_kwargs)
        self.rope = rope
        self.d_model = d_model
        self.num_heads = num_heads
        self.hdim = d_model // num_heads
        self.register_buffer("causal_mask", torch.tril(torch.ones(4096, 4096, dtype=torch.bool)), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        """
        x (bs, seq_len, d_model)
        (bs, seq_len, num_heads, hdim)
        transpose -> (bs, num_heads, seq_len, hdim)

        """
        bs, seq_len, d_model = x.shape

        q: torch.Tensor = self.q_proj(x)
        k: torch.Tensor = self.k_proj(x)
        v: torch.Tensor = self.v_proj(x)
        q = q.view(bs, seq_len, self.num_heads, self.hdim).transpose(1, 2)
        k = k.view(bs, seq_len, self.num_heads, self.hdim).transpose(1, 2)
        v = v.view(bs, seq_len, self.num_heads, self.hdim).transpose(1, 2)

        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        mask = self.causal_mask[:seq_len, :seq_len]
        attn = scaled_dot_product_attention(q, k, v, mask)
        attn = attn.transpose(1, 2).contiguous().view(bs, seq_len, d_model)
        out = self.output_proj(attn)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len, rope=None, device=None, dtype=None):
        """
        x -> rmsnorm -> mha -> rmsnorm -> swiglu
        """
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.rope = rope
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope=self.rope, **self.factory_kwargs)
        self.ffn = SwiGLU(d_model, d_ff, **self.factory_kwargs)
        self.ln1 = RMSNorm(d_model, **self.factory_kwargs)
        self.ln2 = RMSNorm(d_model, **self.factory_kwargs)

    def forward(self, x: Tensor, token_positions: Tensor = None):
        if token_positions is None:
            token_positions = torch.arange(x.shape[1], device=x.device, dtype=torch.long)
        x_norm = self.ln1(x)
        mha_out = self.attn(x_norm, token_positions) + x
        mha_out_norm = self.ln2(mha_out)
        ffn_out = self.ffn(mha_out_norm) + mha_out
        return ffn_out


class TransformerLM(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_seq_len,
        theta,
        vocab_size,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.vocab_size = vocab_size
        self.rope = RoPE(theta, d_model // num_heads, max_seq_len, device)
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, max_seq_len, self.rope, **self.factory_kwargs)
                for i in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, **self.factory_kwargs)
        self.lm_head = Linear(d_model, vocab_size, **self.factory_kwargs)

    def forward(self, x: torch.Tensor):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x_norm = self.ln_final(x)
        out = self.lm_head(x_norm)
        # out = softmax(out)
        return out


# R@X
# def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
#     """
#     R@X.T.T
#     X.T@R.T
#     (seq_len, d_model, de_model)
#     (bs, seq_len, d_model, 1)
#     """
#     cos_val = self.cos[token_positions]
#     sin_val = self.sin[token_positions]
#     bs, seq_len = x.shape[:2]

#     R = torch.zeros(bs, seq_len, self.d_k, self.d_k, device=x.device, dtype=x.dtype)
#     idx = torch.arange(self.d_k // 2, device=R.device) * 2
#     R[:, :, idx, idx] = cos_val
#     R[:, :, idx + 1, idx + 1] = cos_val
#     R[:, :, idx, idx + 1] = -sin_val
#     R[:, :, idx + 1, idx] = sin_val
#     RT = R.transpose(-2, -1)
#     x = x.unsqueeze(dim=-2)
#     return (x @ RT).transpose(-2, -1).squeeze(-1)


# test part --------------------------------------------------------------------------------------------------------------------------




@test_log("Linear")
def test_linear():
    in_features = 3
    out_features = 4
    factory_kwargs = {"device": "cpu", "dtype": torch.float32}
    linear_layer = Linear(in_features, out_features, **factory_kwargs)
    inputs = torch.randn(3)
    out = linear_layer(inputs)
    print(linear_layer.weight.data)
    print(out.shape)


@test_log("Embedding")
def test_embedding():
    num_embeddings = 100
    embedding_dim = 300
    factory_kwargs = {"device": "cpu", "dtype": torch.float32}
    inputs = torch.randint(0, 100, (30, 10))
    print(f"input shape {inputs.shape}")
    embedding_layer = Embedding(num_embeddings, embedding_dim, **factory_kwargs)
    embed = embedding_layer(inputs)
    print(f"after embedding shape is : {embed.shape}")


@test_log("RMSNorm")
def test_rmsnorm():
    d_model = 4
    rmsnrom_layzer = RMSNorm(d_model)
    inputs = torch.randn((10, 20, 4))
    outputs = rmsnrom_layzer(inputs)
    print(f"after norm shape is : {outputs.shape}")


@test_log("SwiGLU")
def test_swiglu():
    d_model = 3
    d_ff = 10
    swiglu_layzer = SwiGLU(d_model, d_ff)
    inputs = torch.randn((10, 20, 3))
    outputs = swiglu_layzer(inputs)
    print(f"after swiglu shape is : {outputs.shape}")


@test_log("RoPE")
def test_RoPE():
    theat = 10000
    d_k = 512
    max_seq_len = 100
    rope = RoPE(theat, d_k, max_seq_len)
    inputs = torch.ones(100, 10, 512)
    token_position = torch.arange(0, 10)
    outputs = rope(inputs, token_position)
    print(f"after rope shape is {outputs.shape}")
    assert inputs.shape == outputs.shape


@test_log("softmax")
def test_softmax():
    inputs = torch.ones((1, 2, 10), dtype=torch.float32)
    norm_output = softmax(inputs)
    print(norm_output)


@test_log("scaled_dot_product_attention")
def test_scaled_dot_product_attention():
    q = torch.ones((1, 2, 10), dtype=torch.float32)
    k = torch.ones((1, 2, 10), dtype=torch.float32)
    v = torch.ones((1, 2, 10), dtype=torch.float32)
    mask = torch.rand(1, 2, 2)
    mask = mask > 0.5
    print("mask : ", mask)
    inf_val = torch.fill(torch.empty(1, 2, 2), -torch.inf)
    print(f"before inf value", inf_val)
    inf_val[mask] = 0
    print(f"after inf value", inf_val)

    norm_output = scaled_dot_product_attention(q, k, v, mask)
    print(norm_output.shape)


@test_log("multi_head_attention")
def test_mha():
    d_model = 512
    num_heads = 8
    seq_len = 10

    mha_layer = MultiHeadSelfAttention(d_model, num_heads)

    inputs = torch.ones(64, 100, 512)
    token_positions = torch.arange(100)

    mha_layer(inputs, token_positions)
    mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=0).bool()
    print(mask)


@test_log("TransformerBlock")
def test_transformer_block():
    d_model = 512
    num_heads = 8
    d_ff = 2048
    max_seq_len = 100
    theta = 10000

    rope = RoPE(theta, d_model // num_heads, max_seq_len)
    block = TransformerBlock(d_model, num_heads, d_ff, max_seq_len, rope)

    bs, seq_len = 2, 10
    inputs = torch.randn(bs, seq_len, d_model)
    outputs = block(inputs)

    assert outputs.shape == inputs.shape, f"Expected {inputs.shape}, got {outputs.shape}"
    print(f"input shape: {inputs.shape}, output shape: {outputs.shape}")

    for name, param in block.named_parameters():
        print(name)


@test_log("Transformer")
def test_transformer():
    d_model = 512
    num_heads = 8
    d_ff = 2048
    max_seq_len = 100
    theta = 10000
    num_layers = 6
    vocab_size = 10000

    transformer = TransformerLM(d_model, num_heads, num_layers, d_ff, max_seq_len, theta, vocab_size)

    bs, seq_len = 2, 10
    inputs = torch.randint(0, vocab_size, (bs, seq_len))
    outputs = transformer(inputs)

    assert outputs.shape == (bs, seq_len, vocab_size), f"Expected {(bs, seq_len, vocab_size)}, got {outputs.shape}"
    print(f"input shape: {inputs.shape}, output shape: {outputs.shape}")

    for name, param in transformer.named_parameters():
        print(name)


if __name__ == "__main__":
    test_linear()
    test_embedding()
    test_rmsnorm()
    test_swiglu()
    test_RoPE()
    test_softmax()
    test_scaled_dot_product_attention()
    test_mha()
    test_transformer_block()
    test_transformer()
