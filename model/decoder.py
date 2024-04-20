from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Module, Parameter, Sequential

EPS = torch.finfo(torch.float32).eps


class CosinePositionalEncoding(Module):
    """
    Cosine positional encoding module.

    Args:
        seq_len (int): Length of the input sequence.
        dim_emb (int): Dimension of the input embeddings.
        base (int, optional): Base value for positional encoding. Defaults to 10_000.
        eps (float, optional): Small value to avoid division by zero. Defaults to EPS.
    """

    def __init__(self, seq_len: int, dim_emb: int, base: int = 10_000, eps: float = EPS) -> None:
        super().__init__()

        indices = torch.arange(0, seq_len, dtype=torch.float)
        scale = 1 / (base ** (torch.arange(0, dim_emb, 2, dtype=torch.float) / dim_emb) + eps)

        position = torch.zeros(1, 1, seq_len, dim_emb)
        position[:, :, :, 0::2] = torch.sin(indices[None, None, :, None] * scale)
        position[:, :, :, 1::2] = torch.cos(indices[None, None, :, None] * scale)

        self.register_buffer("position", position)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the positional encoding module.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_heads, seq_len, dim_emb).

        Returns:
            Tensor: Output tensor after adding positional encodings.
        """
        x = x + self.position
        return x


class RotaryPositionalEncoding(Module):
    """
    Rotary positional encoding module.

    Args:
        seq_len (int): Length of the input sequence.
        dim_emb (int): Dimension of the input embeddings.
        base (int, optional): Base value for positional encoding. Defaults to 10000.
        eps (float, optional): Small value to avoid division by zero. Defaults to EPS.
    """

    def __init__(self, seq_len: int, dim_emb: int, base: int = 10000, eps: float = EPS) -> None:
        super().__init__()

        self.dim_emb = dim_emb
        indices = torch.arange(0, seq_len, dtype=torch.float)
        scale = 1 / (base ** (torch.arange(0, dim_emb, 2, dtype=torch.float) / dim_emb) + eps)

        position = torch.outer(indices, scale)
        position = torch.cat((position, position), dim=-1)

        position_cos = torch.cos(position[None, None, :, :])
        position_sin = torch.sin(position[None, None, :, :])

        self.register_buffer("position_cos", position_cos)
        self.register_buffer("position_sin", position_sin)

    def _rotate_half(self, x: Tensor) -> Tensor:
        x1, x2 = x[..., : self.dim_emb // 2], x[..., self.dim_emb // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the rotary positional encoding module.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_heads, seq_len, dim_emb).

        Returns:
            Tensor: Output tensor after adding positional encodings.
        """
        x = (x * self.position_cos) + (self._rotate_half(x) * self.position_sin)
        return x


class RMSNorm(Module):
    """
    RMS normalization module.

    Args:
        dim_last (int): Dimension along which normalization is applied.
        eps (float, optional): Small value to avoid division by zero. Defaults to EPS.
    """

    def __init__(self, dim_last: int, eps: float = EPS):
        super().__init__()
        self.scale = dim_last ** 0.5
        self.gain = Parameter(torch.ones(dim_last), requires_grad=True)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the RMS normalization module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Normalized tensor.
        """
        norm = torch.norm(x, 2, dim=-1, keepdim=True)
        x = self.scale * self.gain * x / (norm + self.eps)
        return x


class SwiGLU(Module):
    """
    Swish-Gated Linear Unit (SwiGLU) module.

    Args:
        dim_in (int): Input dimension.
        bias (bool, optional): Whether to include bias or not. Defaults to True.
    """

    def __init__(self, dim_in: int, bias: bool = True) -> None:
        super().__init__()

        self.dim_in = dim_in
        self.linear = Linear(dim_in, 2 * dim_in, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the SwiGLU module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after SwiGLU transformation.
        """
        x = self.linear(x)
        x = F.silu(x[..., : self.dim_in]) + x[..., self.dim_in :]
        return x


class SelfAttention(Module):
    """
    Self-attention module.

    Args:
        seq_len (int): Length of the input sequence.
        dim_emb (int): Dimension of the input embeddings.
        dim_k (int, optional): Dimension of the query and key vectors. Defaults to None.
        dim_v (int, optional): Dimension of the value vectors. Defaults to None.
        causal (bool, optional): Whether to use causal attention or not. Defaults to True.
    """

    def __init__(self, seq_len: int, dim_emb: int, dim_k: int = None, dim_v: int = None, causal: bool = True) -> None:
        super().__init__()

        self.dim_k = dim_k or dim_emb
        self.dim_v = dim_v or dim_emb
        self.causal = causal

        self.proj_q = Linear(dim_emb, self.dim_k, bias=False)
        self.proj_k = Linear(dim_emb, self.dim_k, bias=False)
        self.proj_v = Linear(dim_emb, self.dim_v, bias=False)
        self.proj_out = Linear(self.dim_v, self.dim_v, bias=False)

        self.register_buffer("causal_mask", torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool())

    def forward(self, x: Tensor, return_scores: bool = False) -> Tensor | Tuple[Tensor, Tensor]:
        """
        Forward pass of the self-attention module.

        Args:
            x (Tensor): Input tensor.
            return_scores (bool, optional): Whether to return attention scores or not. Defaults to False.

        Returns:
            Tensor or Tuple[Tensor, Tensor]: Output tensor or tuple of output tensor and attention scores.
        """
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)

        attn_scores = q @ torch.transpose(k, 2, 1)

        if self.causal:
            attn_scores.masked_fill_(self.causal_mask[None, ...], -torch.inf)

        attn_scores = torch.softmax(attn_scores * self.dim_k ** -0.5, dim=-1)
        out = self.proj_out(attn_scores @ v)

        if return_scores:
            return out, attn_scores
        else:
            return out


class MultiHeadAttention(Module):
    """
    Multi-head self-attention module.

    Args:
        seq_len (int): Length of the input sequence.
        num_heads (int): Number of attention heads.
        dim_emb (int): Dimension of the input embeddings.
        dim_k (int, optional): Dimension of the query and key vectors. Defaults to None.
        dim_v (int, optional): Dimension of the value vectors. Defaults to None.
        causal (bool, optional): Whether to use causal attention or not. Defaults to True.
    """

    def __init__(
        self,
        seq_len: int,
        num_heads: int,
        dim_emb: int,
        dim_k: int = None,
        dim_v: int = None,
        causal: bool = True,
    ) -> None:
        super().__init__()

        assert dim_emb % num_heads == 0, "num_heads must be a multiple of dim_emb"

        self.seq_len = seq_len
        self.num_heads = num_heads
        self.dim_head = dim_emb // num_heads
        self.dim_k = dim_k or dim_emb
        self.dim_v = dim_v or dim_emb
        self.causal = causal

        self.positional_encoding = RotaryPositionalEncoding(seq_len, dim_emb // num_heads)

        self.proj_qkv = Linear(dim_emb, 3 * dim_emb, bias=False)
        self.proj_out = Linear(self.dim_v, self.dim_v, bias=False)

        self.register_buffer("causal_mask", torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool())

    def forward(self, x: Tensor, return_scores: bool = False) -> Tensor | Tuple[Tensor, Tensor]:
        """
        Forward pass of the multi-head self-attention module.

        Args:
            x (Tensor): Input tensor.
            return_scores (bool, optional): Whether to return attention scores or not. Defaults to False.

        Returns:
            Tensor or Tuple[Tensor, Tensor]: Output tensor or tuple of output tensor and attention scores.
        """
        qkv = self.proj_qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(-1, self.seq_len, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        k = k.view(-1, self.seq_len, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
        v = v.view(-1, self.seq_len, self.num_heads, self.dim_head).permute(0, 2, 1, 3)

        q = self.positional_encoding(q)
        k = self.positional_encoding(k)

        attn_scores = (q @ k.permute(0, 1, 3, 2)) * self.dim_k ** -0.5

        if self.causal:
            attn_scores.masked_fill_(self.causal_mask[None, None, ...], -torch.inf)

        attn_scores = torch.softmax(attn_scores, dim=-1)
        out = attn_scores @ v

        out = out.permute(0, 2, 1, 3).contiguous().view(-1, self.seq_len, self.dim_v)
        out = self.proj_out(out)

        if return_scores:
            return out, attn_scores
        else:
            return out


class FeedForward(Sequential):
    """
    Feedforward network module with SwiGLU activation.

    Args:
        dim_in (int): Input dimension.
        dim_hidden (int): Hidden dimension of the feedforward network.
        bias (bool, optional): Whether to include bias or not. Defaults to False.
    """

    def __init__(self, dim_in: int, dim_hidden: int, bias: bool = False) -> None:
        super().__init__(
            Linear(dim_in, dim_hidden, bias=bias),
            SwiGLU(dim_hidden),
            Linear(dim_hidden, dim_in, bias=bias),
        )


class TransformerBlock(Module):
    """
    Transformer block module.

    Args:
        seq_len (int): Length of the input sequence.
        dim_emb (int): Dimension of the input embeddings.
        attn_num_heads (int): Number of attention heads.
        ffn_hidden_dim (int): Hidden dimension of the feedforward network.
        ffn_bias (bool, optional): Whether to include bias in the feedforward network. Defaults to False.
        attn_causal (bool, optional): Whether to use causal attention or not. Defaults to True.
    """

    def __init__(
        self,
        seq_len: int,
        dim_emb: int,
        attn_num_heads: int,
        ffn_hidden_dim: int,
        ffn_bias: bool = False,
        attn_causal: bool = True,
    ) -> None:
        super().__init__()

        self.norm_attn = RMSNorm(dim_emb)
        self.multihead_attn = MultiHeadAttention(seq_len, attn_num_heads, dim_emb, causal=attn_causal)
        self.norm_ffn = RMSNorm(dim_emb)
        self.feed_forward = FeedForward(dim_emb, ffn_hidden_dim, bias=ffn_bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the transformer block module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        x = x + self.multihead_attn(self.norm_attn(x))
        x = x + self.feed_forward(self.norm_ffn(x))
        return x
