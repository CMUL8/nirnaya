"""Minimal Nirṇaya inference example — the model-card quickstart.

Run:
    python examples/hello.py
"""

from nirnaya import NirnayaModel

model = NirnayaModel.from_pretrained("cmul8-hf/nirnaya")
result = model.decide(
    state="बिजली बिल का पेमेंट अभी तक नहीं हुआ, urgent hai please",
    questions=[
        {"type": "choice", "options": ["billing", "technical", "sales"]},
        {"type": "score", "k": 5, "instructions": "How urgent is this?"},
        {"type": "noul", "instructions": "Should this be escalated?"},
    ],
)

for i, d in enumerate(result):
    print(f"Q{i} [{d.type}]  probs={[f'{p:.3f}' for p in d.probs]}  "
          f"answer={d.answer}  confidence={d.confidence:.3f}", end="")
    if d.expected_level is not None:
        print(f"  expected_level={d.expected_level:.2f}", end="")
    if d.p_true is not None:
        print(f"  p_true={d.p_true:.3f}", end="")
    print()
