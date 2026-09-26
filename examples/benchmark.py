"""Nirṇaya latency benchmark — p50/p95/p99 and batched-vs-loop speedup.

Run:
    python examples/benchmark.py [--n-warmup 5] [--n-timed 100]
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from nirnaya import NirnayaModel


STATE = (
    "बिजली बिल का पेमेंट अभी तक नहीं हुआ, urgent hai please. "
    "Kal se light nahi hai, entire building affected."
)

QUESTIONS = [
    {"type": "choice", "options": ["billing", "technical", "sales"]},
    {"type": "score", "k": 5, "instructions": "How urgent is this?"},
    {"type": "noul", "instructions": "Should this be escalated?"},
]


def _percentile(values: list[float], p: float) -> float:
    """Return the p-th percentile (0..100) of ``values`` via linear interpolation."""
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _time_one(fn, n_warmup: int, n_timed: int) -> list[float]:
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    samples_ms: list[float] = []
    for _ in range(n_timed):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    return samples_ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-warmup", type=int, default=5)
    ap.add_argument("--n-timed", type=int, default=100)
    ap.add_argument("--model", default="cmul8-hf/nirnaya")
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    model = NirnayaModel.from_pretrained(args.model)
    print(f"Loaded. GPU: {torch.cuda.get_device_name()}")

    # ---- Batched: all 3 questions in one forward ----
    batched_samples = _time_one(
        lambda: model.decide(state=STATE, questions=QUESTIONS),
        n_warmup=args.n_warmup,
        n_timed=args.n_timed,
    )

    # ---- Loop: one forward per question ----
    def loop_call() -> None:
        for q in QUESTIONS:
            model.decide(state=STATE, questions=[q])

    loop_samples = _time_one(loop_call, n_warmup=args.n_warmup, n_timed=args.n_timed)

    def fmt(name: str, samples: list[float]) -> None:
        print(
            f"  {name:12s}  "
            f"p50={statistics.median(samples):6.1f} ms  "
            f"p95={_percentile(samples, 95):6.1f} ms  "
            f"p99={_percentile(samples, 99):6.1f} ms  "
            f"mean={statistics.mean(samples):6.1f} ms"
        )

    print(f"\n3-question decide (n_timed={args.n_timed}):")
    fmt("batched", batched_samples)
    fmt("loop", loop_samples)

    speedup = statistics.median(loop_samples) / statistics.median(batched_samples)
    print(f"\nSpeedup (loop / batched p50): {speedup:.2f}x")


if __name__ == "__main__":
    main()
