"""Smoke tests for the Nirṇaya inference API.

Runs on CPU for pure-logic paths; end-to-end model tests skip when no CUDA.
"""

from __future__ import annotations

import pytest
import torch

from nirnaya._decide import Decision, _coerce_question, _probs_to_decision


# ---------------------------------------------------------------------------
# _coerce_question — happy paths
# ---------------------------------------------------------------------------

def test_coerce_choice_with_options():
    q = _coerce_question(
        {"type": "choice", "options": ["a", "b", "c"]}, idx=0
    )
    assert q.type == "choice"
    assert q.options == ["a", "b", "c"]
    assert q.qid == "q0"
    assert q.label == 0


def test_coerce_score_with_k_desugars():
    q = _coerce_question({"type": "score", "k": 5}, idx=2)
    assert q.type == "score"
    assert q.options == ["1", "2", "3", "4", "5"]
    assert q.qid == "q2"


def test_coerce_score_with_options():
    q = _coerce_question(
        {"type": "score", "options": ["low", "med", "high"]}, idx=0
    )
    assert q.options == ["low", "med", "high"]


def test_coerce_noul():
    q = _coerce_question(
        {"type": "noul", "instructions": "Escalate?"}, idx=1
    )
    assert q.type == "noul"
    assert q.options is None
    assert q.label is False
    assert q.instructions == "Escalate?"


def test_coerce_uses_supplied_id():
    q = _coerce_question(
        {"type": "noul", "id": "custom"}, idx=0
    )
    assert q.qid == "custom"


def test_coerce_defaults_instructions_to_empty():
    q = _coerce_question(
        {"type": "choice", "options": ["a", "b"]}, idx=0
    )
    assert q.instructions == ""


# ---------------------------------------------------------------------------
# _coerce_question — error paths
# ---------------------------------------------------------------------------

def test_coerce_rejects_bad_type():
    with pytest.raises(ValueError, match="'type' must be"):
        _coerce_question({"type": "freeform"}, idx=0)


def test_coerce_rejects_noul_with_options():
    with pytest.raises(ValueError, match="must not carry"):
        _coerce_question(
            {"type": "noul", "options": ["yes", "no"]}, idx=0
        )


def test_coerce_rejects_both_options_and_k():
    with pytest.raises(ValueError, match="not both"):
        _coerce_question(
            {"type": "score", "options": ["a", "b"], "k": 3}, idx=0
        )


def test_coerce_rejects_k_on_choice():
    with pytest.raises(ValueError, match="only valid for score"):
        _coerce_question({"type": "choice", "k": 3}, idx=0)


def test_coerce_rejects_missing_options():
    with pytest.raises(ValueError, match="needs 'options'"):
        _coerce_question({"type": "choice"}, idx=0)


def test_coerce_rejects_k_out_of_range():
    with pytest.raises(ValueError, match="k must be between 2 and 64"):
        _coerce_question({"type": "score", "k": 1}, idx=0)
    with pytest.raises(ValueError, match="k must be between 2 and 64"):
        _coerce_question({"type": "score", "k": 65}, idx=0)


def test_coerce_rejects_option_count_out_of_range():
    with pytest.raises(ValueError, match="length 2..64"):
        _coerce_question(
            {"type": "choice", "options": ["only"]}, idx=0
        )


# ---------------------------------------------------------------------------
# Decision field population via _probs_to_decision
# ---------------------------------------------------------------------------

def test_decision_choice_populates_probs_and_answer():
    probs = torch.tensor([0.7, 0.2, 0.1, 0.0])  # 4th slot is a max_K pad
    d = _probs_to_decision(probs, q_type="choice", K=3)
    assert d.type == "choice"
    assert d.probs == pytest.approx([0.7, 0.2, 0.1], rel=1e-5)
    assert d.answer == 0
    assert d.expected_level is None
    assert d.p_true is None
    # confidence in [0, 1]
    assert 0.0 <= d.confidence <= 1.0


def test_decision_score_has_expected_level():
    probs = torch.tensor([0.05, 0.10, 0.15, 0.30, 0.40])
    d = _probs_to_decision(probs, q_type="score", K=5)
    assert d.type == "score"
    assert d.answer == 4
    assert d.expected_level == pytest.approx(
        1 * 0.05 + 2 * 0.10 + 3 * 0.15 + 4 * 0.30 + 5 * 0.40, rel=1e-4
    )
    assert d.p_true is None


def test_decision_noul_has_p_true_and_bool_answer():
    # NoulHead output is p_true in a length-1 tensor.
    probs = torch.tensor([0.71])
    d = _probs_to_decision(probs, q_type="noul", K=0)
    assert d.type == "noul"
    assert d.probs == pytest.approx([0.29, 0.71], rel=1e-5)
    assert d.answer is True
    assert d.p_true == pytest.approx(0.71, rel=1e-5)
    assert d.expected_level is None
    # noul confidence = |2p - 1|
    assert d.confidence == pytest.approx(abs(2 * 0.71 - 1.0), rel=1e-5)


def test_decision_is_dataclass_readable():
    d = Decision(
        type="noul", probs=[0.3, 0.7], answer=True,
        confidence=0.4, p_true=0.7,
    )
    assert d.type == "noul"
    assert d.probs == [0.3, 0.7]


# ---------------------------------------------------------------------------
# End-to-end (GPU only)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_end_to_end_decide_shape():
    """Load the model and run a 3-question decide. Only checks shape, not values."""
    from nirnaya import NirnayaModel

    model = NirnayaModel.from_pretrained("cmul8-hf/nirnaya")
    result = model.decide(
        state="Bill payment pending, please check urgently",
        questions=[
            {"type": "choice", "options": ["billing", "technical", "sales"]},
            {"type": "score", "k": 5, "instructions": "How urgent?"},
            {"type": "noul", "instructions": "Escalate?"},
        ],
    )
    assert len(result) == 3
    assert result[0].type == "choice"
    assert len(result[0].probs) == 3
    assert result[1].type == "score"
    assert result[1].expected_level is not None
    assert result[2].type == "noul"
    assert result[2].p_true is not None
    # Prob sums
    assert sum(result[0].probs) == pytest.approx(1.0, abs=1e-4)
    assert sum(result[1].probs) == pytest.approx(1.0, abs=1e-4)
