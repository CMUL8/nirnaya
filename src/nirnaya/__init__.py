"""Nirṇaya — India-first System-One decision model (inference).

Public API:
    from nirnaya import NirnayaModel, Decision

    model = NirnayaModel.from_pretrained("cmul8-hf/nirnaya")
    result: list[Decision] = model.decide(state=..., questions=[...])
"""

# Importing _decide has a side effect: it attaches the .decide() method to
# NirnayaModel. Order matters — _model must be imported first (which _decide
# does), then _decide runs the attachment.
from nirnaya._model import NirnayaModel  # noqa: F401
from nirnaya._decide import Decision  # noqa: F401

__version__ = "0.1.0"
__all__ = ["NirnayaModel", "Decision", "__version__"]
