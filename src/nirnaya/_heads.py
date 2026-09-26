"""Decision heads for Nirṇaya.

Both heads run in fp32 regardless of backbone dtype.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerHead(nn.Module):
    """Pointer-network head that scores each option against the [ANSWER] query."""

    def __init__(self, d_model: int, d_head: int = 256) -> None:
        super().__init__()
        self.d_head = d_head
        self.Wq = nn.Linear(d_model, d_head, bias=False)
        self.Wk = nn.Linear(d_model, d_head, bias=False)
        self.Wq.to(torch.float32)
        self.Wk.to(torch.float32)
        nn.init.xavier_uniform_(self.Wq.weight)
        nn.init.xavier_uniform_(self.Wk.weight)

    def forward(
        self,
        hidden: torch.Tensor,   # (B, T, d_model)
        opt_pos: torch.Tensor,  # (B, max_K) long, -1 = padding
        dec_pos: torch.Tensor,  # (B,) long
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (probs, logits) each of shape (B, max_K). Padded positions get prob 0."""
        h = hidden.float()
        B, T, _ = h.shape
        max_K = opt_pos.shape[1]

        safe_pos = opt_pos.clamp(min=0)

        bi = torch.arange(B, device=h.device).unsqueeze(1).expand(B, max_K)
        h_opts = h[bi, safe_pos]  # (B, max_K, d_model)

        bi_1d = torch.arange(B, device=h.device)
        h_dec = h[bi_1d, dec_pos]  # (B, d_model)

        q = self.Wq(h_dec)
        k = self.Wk(h_opts)

        scale = math.sqrt(self.d_head)
        logits = (q.unsqueeze(1) * k).sum(-1) / scale

        pad_mask = (opt_pos == -1)
        logits = logits.masked_fill(pad_mask, -1e9)

        probs = F.softmax(logits, dim=-1)
        return probs, logits


class NoulHead(nn.Module):
    """Binary yes/no head reading the [ANSWER] position."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, 1)
        nn.init.zeros_(self.proj.bias)
        self.ln.to(torch.float32)
        self.proj.to(torch.float32)

    def forward(
        self,
        hidden: torch.Tensor,   # (B, T, d_model)
        dec_pos: torch.Tensor,  # (B,) long
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (p_true, logit) each of shape (B,)."""
        h = hidden.float()
        bi = torch.arange(h.shape[0], device=h.device)
        h_dec = h[bi, dec_pos]
        h_dec = self.ln(h_dec)
        logit = self.proj(h_dec).squeeze(-1)
        p_true = torch.sigmoid(logit)
        return p_true, logit
