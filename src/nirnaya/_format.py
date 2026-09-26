"""Segment-by-segment tokenizer → token ids + anchor positions.

Template:
    [STATE]\\n{state}\\n[QUESTION]\\n{instructions}\\n[OPTIONS]\\n(1) {opt1}\\n(2) {opt2}\\n...\\n[ANSWER]

For noul: skip [OPTIONS] block; opt_pos = [].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nirnaya._question import Question

MAX_STATE_TOKENS = 384
MAX_SEQUENCE_TOKENS = 1024
TYPE_TO_ID = {"choice": 0, "score": 1, "noul": 2}

_ESCAPE_MAP = str.maketrans({"[": "⟦", "]": "⟧"})


def _escape(text: str) -> str:
    return text.translate(_ESCAPE_MAP)


@dataclass
class _MarkerCache:
    tokenizer_id: int
    state_marker: list[int]
    question_marker: list[int]
    options_marker: list[int]
    answer_marker: list[int]


_MARKER_CACHE: dict[int, _MarkerCache] = {}


def _get_marker_cache(tokenizer) -> _MarkerCache:
    tid = id(tokenizer)
    if tid not in _MARKER_CACHE:
        def enc(s: str) -> list[int]:
            return tokenizer.encode(s, add_special_tokens=False)

        _MARKER_CACHE[tid] = _MarkerCache(
            tokenizer_id=tid,
            state_marker=enc("[STATE]\n"),
            question_marker=enc("[QUESTION]\n"),
            options_marker=enc("[OPTIONS]\n"),
            answer_marker=enc("[ANSWER]"),
        )
    return _MARKER_CACHE[tid]


@dataclass
class FormattedExample:
    input_ids: list[int]
    opt_pos: list[int]
    dec_pos: int
    type_id: int


def _tokenize_state(tokenizer, state: str) -> list[int]:
    escaped = _escape(state)
    ids = tokenizer.encode(escaped, add_special_tokens=False)
    if len(ids) > MAX_STATE_TOKENS:
        sep = tokenizer.encode(" [...] ", add_special_tokens=False)
        half = (MAX_STATE_TOKENS - len(sep)) // 2
        if half <= 0:
            return ids[:MAX_STATE_TOKENS]
        ids = ids[:half] + sep + ids[-half:]
    return ids


def _tokenize_option_line(tokenizer, idx: int, option_text: str) -> list[int]:
    line = f"({idx + 1}) {_escape(option_text)}\n"
    return tokenizer.encode(line, add_special_tokens=False)


def format_example(tokenizer, state: str, question: "Question") -> FormattedExample:
    mc = _get_marker_cache(tokenizer)

    ids: list[int] = []
    bos = tokenizer.bos_token_id
    if bos is not None:
        ids.append(bos)

    state_ids = _tokenize_state(tokenizer, state)
    ids += mc.state_marker
    ids += state_ids
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    ids += newline_ids

    ids += mc.question_marker
    instr_ids = tokenizer.encode(_escape(question.instructions), add_special_tokens=False)
    ids += instr_ids
    ids += newline_ids

    opt_pos: list[int] = []
    if question.type != "noul":
        ids += mc.options_marker
        options = question.options or []
        for i, opt in enumerate(options):
            opt_line_ids = _tokenize_option_line(tokenizer, i, opt)
            ids += opt_line_ids
            opt_pos.append(len(ids) - 1)

    ids += mc.answer_marker
    dec_pos = len(ids) - 1

    if len(ids) > MAX_SEQUENCE_TOKENS:
        ids, opt_pos, dec_pos = _trim_to_budget(
            tokenizer, state, question, mc, bos, MAX_SEQUENCE_TOKENS
        )

    type_id = TYPE_TO_ID[question.type]

    # If trimming removed all options, the pointer head has nothing to score.
    if question.type != "noul" and len(opt_pos) == 0:
        raise ValueError(
            f"All options were trimmed to fit the {MAX_SEQUENCE_TOKENS}-token "
            f"budget. Shorten the state or the instructions."
        )

    return FormattedExample(
        input_ids=ids,
        opt_pos=opt_pos,
        dec_pos=dec_pos,
        type_id=type_id,
    )


def _trim_to_budget(tokenizer, state: str, question: "Question", mc: _MarkerCache,
                    bos: int | None, budget: int) -> tuple[list[int], list[int], int]:
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)

    skeleton: list[int] = []
    if bos is not None:
        skeleton.append(bos)

    state_ids = _tokenize_state(tokenizer, state)
    skeleton += mc.state_marker + state_ids + newline_ids
    skeleton += mc.question_marker
    instr_ids = tokenizer.encode(_escape(question.instructions), add_special_tokens=False)
    skeleton += instr_ids + newline_ids

    opt_pos: list[int] = []

    if len(skeleton) + len(mc.answer_marker) > budget:
        raise ValueError(
            f"Fixed skeleton ({len(skeleton)} tokens) + answer marker "
            f"({len(mc.answer_marker)}) exceeds token budget ({budget}). "
            f"Instructions are too long."
        )

    if question.type != "noul":
        skeleton += mc.options_marker
        options = question.options or []
        working = list(skeleton)
        for i, opt in enumerate(options):
            opt_line_ids = _tokenize_option_line(tokenizer, i, opt)
            if len(working) + len(opt_line_ids) + len(mc.answer_marker) > budget:
                break
            working += opt_line_ids
            opt_pos.append(len(working) - 1)
        working += mc.answer_marker
        dec_pos = len(working) - 1
        return working, opt_pos, dec_pos

    skeleton += mc.answer_marker
    dec_pos = len(skeleton) - 1
    return skeleton, [], dec_pos
