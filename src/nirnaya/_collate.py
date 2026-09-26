"""Pad a list of ``FormattedExample`` into a batched tensor bundle."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nirnaya._format import FormattedExample


@dataclass
class NirnayaBatch:
    input_ids: torch.Tensor       # (B, T) long
    attention_mask: torch.Tensor  # (B, T) long
    opt_pos: torch.Tensor         # (B, max_K) long, -1=pad
    dec_pos: torch.Tensor         # (B,) long
    type_ids: torch.Tensor        # (B,) long

    def to(self, device: str | torch.device) -> "NirnayaBatch":
        return NirnayaBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            opt_pos=self.opt_pos.to(device),
            dec_pos=self.dec_pos.to(device),
            type_ids=self.type_ids.to(device),
        )


def collate_fn(examples: list[FormattedExample], pad_token_id: int) -> NirnayaBatch:
    """Right-pad a list of examples into a NirnayaBatch.

    - input_ids: right-pad with pad_token_id to max sequence length.
    - attention_mask: 1 for real tokens, 0 for padding.
    - opt_pos: right-pad with -1 to max K in the batch (min width 1).
    - dec_pos, type_ids: stacked as-is.
    """
    max_len = max(len(ex.input_ids) for ex in examples)
    max_K = max((len(ex.opt_pos) for ex in examples), default=0)
    max_K = max(max_K, 1)

    input_ids_list: list[list[int]] = []
    attention_mask_list: list[list[int]] = []
    opt_pos_list: list[list[int]] = []
    dec_pos_list: list[int] = []
    type_ids_list: list[int] = []

    for ex in examples:
        seq_len = len(ex.input_ids)
        pad_len = max_len - seq_len

        input_ids_list.append(ex.input_ids + [pad_token_id] * pad_len)
        attention_mask_list.append([1] * seq_len + [0] * pad_len)

        k = len(ex.opt_pos)
        opt_pos_list.append(ex.opt_pos + [-1] * (max_K - k))

        dec_pos_list.append(ex.dec_pos)
        type_ids_list.append(ex.type_id)

    return NirnayaBatch(
        input_ids=torch.tensor(input_ids_list, dtype=torch.long),
        attention_mask=torch.tensor(attention_mask_list, dtype=torch.long),
        opt_pos=torch.tensor(opt_pos_list, dtype=torch.long),
        dec_pos=torch.tensor(dec_pos_list, dtype=torch.long),
        type_ids=torch.tensor(type_ids_list, dtype=torch.long),
    )
