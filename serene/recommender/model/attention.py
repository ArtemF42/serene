import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """Optimized implementation of self-attention"""

    def __init__(self, embedding_dim: int, num_heads: int, dropout_p: float, is_causal: bool = False) -> None:
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError("`embedding_dim` must be divisible by `num_heads`")

        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.is_causal = is_causal

        self.qkv_proj = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Note:
            Supports boolean and floating-point masks. For boolean masks, `True` indicates that the element
            **should take part in attention**. Floating-point masks are added directly to attention scores.

        Args:
            x (torch.Tensor): input tensor of shape (B, L, E), where B is batch size, L is sequence length,
                and E is embedding dimension.
            padding_mask (torch.Tensor | None, optional): mask indicating non-padding tokens.
                If provided, must be of shape (B, L). Defaults to None.
            attention_mask (torch.Tensor | None, optional): general mask over allowed attention positions.
                If provided, must be broadcastable to shape (B, H, L, L), where H is the number of attention heads.
                Defaults to None.

        Returns:
            torch.Tensor: attention output of shape (B, L, E).
        """
        B, L, _ = x.shape

        if padding_mask is None and attention_mask is None:
            is_causal = self.is_causal
        else:
            is_causal = False

            if attention_mask is None:
                attention_mask = torch.zeros(1, 1, L, L, dtype=x.dtype, device=x.device)
            else:
                try:
                    torch.broadcast_shapes(attention_mask.shape, (B, self.num_heads, L, L))
                except RuntimeError as error:
                    raise ValueError("`attention_mask` must be broadcastable to (B, H, L, L).") from error

                attention_mask = self._sanitize_mask(attention_mask, x.dtype)

            if padding_mask is not None:
                if padding_mask.shape != (B, L):
                    raise ValueError("`padding_mask` must have shape (B, L).")

                padding_mask = self._sanitize_mask(padding_mask, x.dtype)
                attention_mask = attention_mask + padding_mask[:, None, None, :]  # (B, 1, 1, L)

            if self.is_causal:
                causal_mask = torch.full((L, L), -torch.inf, dtype=x.dtype, device=x.device).triu(diagonal=1)
                attention_mask = attention_mask + causal_mask[None, None, :, :]  # (1, 1, L, L)

        q, k, v = map(self._split_heads, self.qkv_proj(x).chunk(3, dim=-1))

        attn_out = nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )  # fmt: skip

        return self.dropout(self.out_proj(self._merge_heads(attn_out)))

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor of shape (B, L, E).

        Returns:
            torch.Tensor: tensor of shape (B, H, L, E // H).
        """
        return x.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor of shape (B, H, L, E // H).

        Returns:
            torch.Tensor: tensor of shape (B, L, E).
        """
        return x.transpose(1, 2).flatten(-2)

    def _sanitize_mask(self, mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if torch.is_floating_point(mask):
            return mask.to(dtype)
        elif mask.dtype == torch.bool:
            return torch.zeros(mask.shape, dtype=dtype, device=mask.device).masked_fill_(~mask, -torch.inf)
        else:
            raise TypeError(f"only float and boolean masks are supported. Found {mask.dtype!r}.")
