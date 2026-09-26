"""Nirṇaya inference model — loads a trained adapter + heads, runs one forward pass.

Load-only. No training, no fine-tuning, no save. The public entry point is
``NirnayaModel.from_pretrained`` plus the ``.decide()`` method attached from
``_decide.py``.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from nirnaya._heads import NoulHead, PointerHead

DEFAULT_MODEL_ID = "cmul8-hf/nirnaya"


@dataclass
class NirnayaConfig:
    base_model_id: str = "Qwen/Qwen3-4B"
    format_version: int = 1
    d_model: int = 4096
    d_head: int = 256
    max_state_tokens: int = 384
    max_seq_tokens: int = 1024
    temperatures: dict = field(default_factory=dict)


class NirnayaModel(nn.Module):
    """Backbone (PeftModel over Qwen3-4B) + PointerHead + NoulHead."""

    def __init__(self, backbone, tokenizer, cfg: NirnayaConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.pointer_head = PointerHead(cfg.d_model, cfg.d_head)
        self.noul_head = NoulHead(cfg.d_model)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        subfolder: str = "best",
    ) -> "NirnayaModel":
        """Load a trained Nirṇaya model.

        Args:
            model_id_or_path: A Hugging Face repo id (e.g.
                ``"cmul8-hf/nirnaya"``) or a local path to a directory that
                contains ``nirnaya_config.json``, ``adapter/`` and
                ``heads.pt``.
            device: Target device. Defaults to ``"cuda"``; raises
                ``RuntimeError`` if CUDA is not available.
            subfolder: For HF repos, the subfolder inside the repo to load —
                ``"best"`` (default) or ``"latest"``. Ignored for local paths.
        """
        if device is None:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Nirṇaya v1.0 requires a CUDA GPU with bf16 support. "
                    "CPU and MPS are not supported in v1. See "
                    "https://huggingface.co/cmul8-hf/nirnaya for hardware "
                    "requirements."
                )
            device = "cuda"

        try:
            from huggingface_hub import snapshot_download
            from peft import PeftModel
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "Missing inference dependencies. Reinstall with "
                "`pip install nirnaya`."
            ) from e

        path = Path(model_id_or_path)
        if path.exists() and path.is_dir():
            save_dir = path
        else:
            local = snapshot_download(
                repo_id=model_id_or_path,
                allow_patterns=f"{subfolder}/**",
            )
            save_dir = Path(local) / subfolder

        cfg_path = save_dir / "nirnaya_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"nirnaya_config.json not found under {save_dir}. "
                f"Expected a Nirṇaya checkpoint layout: nirnaya_config.json + "
                f"adapter/ + heads.pt."
            )

        with open(cfg_path, encoding="utf-8") as f:
            raw_cfg = json.load(f)
        known_fields = {f.name for f in fields(NirnayaConfig)}
        unknown = set(raw_cfg) - known_fields
        if unknown:
            warnings.warn(
                f"Ignoring unknown nirnaya_config keys: {sorted(unknown)}. "
                f"Checkpoint may have been trained with a newer Nirṇaya "
                f"version than this inference package.",
                stacklevel=2,
            )
        cfg = NirnayaConfig(**{k: v for k, v in raw_cfg.items() if k in known_fields})

        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        backbone = AutoModel.from_pretrained(
            cfg.base_model_id,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        backbone = PeftModel.from_pretrained(backbone, str(save_dir / "adapter"))

        # Sanity check: adapter looks trained (lora_B is zero-init before training).
        lora_b_norms = [
            p.norm().item()
            for n, p in backbone.named_parameters()
            if "lora_B" in n
        ]
        if lora_b_norms and max(lora_b_norms) < 1e-6:
            raise RuntimeError(
                f"Loaded LoRA adapter looks untrained — all lora_B weights are "
                f"near zero. Verify {save_dir / 'adapter'} contains real "
                f"trained weights."
            )

        model = cls(backbone, tokenizer, cfg)

        heads_state = torch.load(
            save_dir / "heads.pt", map_location="cpu", weights_only=True
        )
        model.pointer_head.load_state_dict(heads_state["pointer_head"])
        model.noul_head.load_state_dict(heads_state["noul_head"])

        # Inference-only: eval mode + no gradients.
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        model.to(device)
        return model

    # ------------------------------------------------------------------
    # Forward — probs only, no loss (inference contract)
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,       # (B, T) long
        attention_mask: torch.Tensor,  # (B, T) long
        opt_pos: torch.Tensor,         # (B, max_K) long, -1=pad
        dec_pos: torch.Tensor,         # (B,) long
        type_ids: torch.Tensor,        # (B,) long
    ) -> dict[str, Any]:
        """One backbone pass, route each example to the right head, return per-example probs."""
        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )
        hidden = outputs.last_hidden_state  # (B, T, d_model)

        B = input_ids.shape[0]
        device = input_ids.device

        probs_out: list[torch.Tensor] = [
            torch.zeros(1, device=device) for _ in range(B)
        ]

        ptr_mask = type_ids != 2   # choice + score → PointerHead
        noul_mask = type_ids == 2  # noul → NoulHead

        if ptr_mask.any():
            ptr_idx = ptr_mask.nonzero(as_tuple=True)[0]
            ptr_hidden = hidden[ptr_idx]
            ptr_opt_pos = opt_pos[ptr_idx]
            ptr_dec_pos = dec_pos[ptr_idx]

            ptr_probs, _ = self.pointer_head(ptr_hidden, ptr_opt_pos, ptr_dec_pos)

            for j, orig_i in enumerate(ptr_idx.tolist()):
                probs_out[orig_i] = ptr_probs[j]

        if noul_mask.any():
            noul_idx = noul_mask.nonzero(as_tuple=True)[0]
            noul_hidden = hidden[noul_idx]
            noul_dec_pos = dec_pos[noul_idx]

            noul_p, _ = self.noul_head(noul_hidden, noul_dec_pos)

            for j, orig_i in enumerate(noul_idx.tolist()):
                probs_out[orig_i] = noul_p[j].unsqueeze(0)

        return {"probs": probs_out}
