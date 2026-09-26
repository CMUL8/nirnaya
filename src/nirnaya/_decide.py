"""Public decision API — Decision dataclass and the batched ``decide()`` method.

The method is attached to ``NirnayaModel`` at import time so users see it as
``model.decide(...)`` from a single import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from nirnaya._collate import collate_fn
from nirnaya._format import format_example
from nirnaya._model import NirnayaModel
from nirnaya._question import Question


@dataclass
class Decision:
    """Result of one typed question.

    Fields:
        type:            "choice" | "score" | "noul".
        probs:           probability per option; noul → ``[P(false), P(true)]``.
        answer:          argmax option index for choice/score; bool for noul.
        confidence:      calibrated confidence in [0, 1].
        expected_level:  1-indexed expected level for score; None otherwise.
        p_true:          P(true) for noul; None otherwise.
    """
    type: Literal["choice", "score", "noul"]
    probs: list[float]
    answer: int | bool
    confidence: float
    expected_level: float | None = None
    p_true: float | None = None


def _coerce_question(q_dict: dict[str, Any], idx: int) -> Question:
    """Public-shape dict → ``Question`` for the tokenizer.

    Rules:
        * ``qid`` auto-generated as ``f"q{idx}"`` if missing.
        * ``instructions`` defaults to ``""``.
        * For ``score``, ``k: int`` desugars to ``options=["1", ..., str(k)]``.
        * ``choice`` and ``score`` require options (or ``k`` for score).
        * ``noul`` must not carry options or ``k``.
    """
    q_type = q_dict.get("type")
    if q_type not in ("choice", "score", "noul"):
        raise ValueError(
            f"question {idx}: 'type' must be 'choice', 'score' or 'noul', "
            f"got {q_type!r}"
        )

    instructions = q_dict.get("instructions", "")
    qid = str(q_dict.get("id", f"q{idx}"))

    if q_type == "noul":
        if "options" in q_dict or "k" in q_dict:
            raise ValueError(
                f"noul question {qid}: must not carry 'options' or 'k'"
            )
        return Question(
            qid=qid, type="noul", instructions=instructions,
            options=None, label=False,
        )

    has_options = "options" in q_dict and q_dict["options"] is not None
    has_k = "k" in q_dict and q_dict["k"] is not None
    if has_options and has_k:
        raise ValueError(
            f"{q_type} question {qid}: pass either 'options' or 'k', not both"
        )
    if has_options:
        options = [str(o) for o in q_dict["options"]]
    elif has_k:
        if q_type != "score":
            raise ValueError(
                f"'k' shortcut is only valid for score questions; "
                f"choice question {qid} must pass 'options'"
            )
        k = int(q_dict["k"])
        if k < 2 or k > 64:
            raise ValueError(
                f"score question {qid}: k must be between 2 and 64, got {k}"
            )
        options = [str(i + 1) for i in range(k)]
    else:
        raise ValueError(
            f"{q_type} question {qid}: needs 'options' (choice/score) or "
            f"'k' (score)"
        )

    if len(options) < 2 or len(options) > 64:
        raise ValueError(
            f"{q_type} question {qid}: options must have length 2..64, "
            f"got {len(options)}"
        )

    return Question(
        qid=qid, type=q_type, instructions=instructions,
        options=options, label=0,
    )


def _probs_to_decision(
    probs_tensor: torch.Tensor,
    q_type: str,
    K: int,
) -> Decision:
    """Turn one raw prob tensor into a public ``Decision``."""
    if q_type == "noul":
        p_yes = float(probs_tensor[0])
        return Decision(
            type="noul",
            probs=[1.0 - p_yes, p_yes],
            answer=bool(p_yes > 0.5),
            confidence=abs(2.0 * p_yes - 1.0),
            p_true=p_yes,
        )

    # PointerHead pads probs to batch max_K; truncate to the real option count.
    probs_k = probs_tensor[:K]
    probs = [float(probs_k[i]) for i in range(K)]
    top_i = int(probs_k.argmax().item())
    p_max = float(probs_k.max())
    confidence = (p_max - 1.0 / K) / (1.0 - 1.0 / K) if K > 1 else 1.0

    if q_type == "score":
        expected = sum((i + 1) * probs[i] for i in range(K))
        return Decision(
            type="score",
            probs=probs,
            answer=top_i,
            confidence=confidence,
            expected_level=expected,
        )

    return Decision(
        type="choice",
        probs=probs,
        answer=top_i,
        confidence=confidence,
    )


@torch.no_grad()
def _decide_method(
    self: NirnayaModel,
    state: str,
    questions: list[dict[str, Any]],
    device: str | None = None,
) -> list[Decision]:
    """Answer several typed questions about ``state`` in a single forward pass.

    Args:
        state: The user/customer text to reason about.
        questions: List of question dicts. Each dict has ``type`` and, for
            choice/score, ``options`` or (for score) ``k``. ``instructions``
            is optional.
        device: Override the device the batch is placed on. Defaults to the
            device of the model's parameters.

    Returns:
        One :class:`Decision` per input question, in order.
    """
    if not questions:
        return []

    if device is None:
        device = next(self.parameters()).device

    tokenizer = self.tokenizer
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    coerced = [_coerce_question(q, i) for i, q in enumerate(questions)]
    formatted = [format_example(tokenizer, state, q) for q in coerced]
    batch = collate_fn(formatted, pad_token_id=pad_id).to(device)

    out = self(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        opt_pos=batch.opt_pos,
        dec_pos=batch.dec_pos,
        type_ids=batch.type_ids,
    )

    decisions: list[Decision] = []
    for q, probs in zip(coerced, out["probs"]):
        probs_cpu = probs.detach().float().cpu()
        K = 0 if q.options is None else len(q.options)
        decisions.append(_probs_to_decision(probs_cpu, q.type, K))
    return decisions


# Attach as a method on NirnayaModel so `model.decide(...)` works.
NirnayaModel.decide = _decide_method  # type: ignore[attr-defined]
