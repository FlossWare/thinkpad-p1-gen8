"""Token-accounting helpers for benchmarks/benchmark.py (issue #1).

Split out from benchmark.py so the accounting rules can be regression-tested
in pure Python, with no OpenVINO installed (see test_benchmark_accounting.py;
benchmark.py itself cannot be imported by tests because of its forkserver
re-import guard).

Core rule (issue #1): every token COUNT comes from the generated token IDs in
the GenAI result object (``EncodedResults.tokens`` when generate() is called
with TokenizedInputs). Streamer callback timestamps are used only for TIMING
boundaries (TTFT, start of decode). Streamer callbacks deliver text chunks,
not tokens: this build merges/skips chunks (measured: 31 callbacks for 32
generated tokens and 122 for 123 in the re-* rows; 30 for 32 requested and
101 for 128 requested -- actually 123 generated, capacity-truncated, issue
#13 -- in the superseded rows), so a callback count is NOT a token count.

Measurement boundaries:
  - TTFT: prompt submitted -> first streamer callback.
  - Prefill throughput: prompt_tokens / TTFT. Prompt tokens are the tokenizer
    encode() length of the prompt; no generated tokens are included.
  - Decode interval: first streamer callback -> generate() return. Every
    generated token after the first completes inside that interval, so the
    numerator (generated_tokens - 1) and the interval describe the same token
    range even when callbacks merge or skip tokens. (Ending the interval at
    the last callback instead would pair an all-tokens numerator with an
    interval that may have skipped trailing tokens.) generate() return
    includes a little final detokenization overhead, so the denominator is
    mildly conservative (throughput slightly underestimated).
  - Known limitation of the start boundary: the first streamer callback is
    assumed to mark the first generated token. A text streamer may withhold
    text for an incomplete multi-byte/multi-token piece, in which case the
    first callback arrives after token k > 1; the numerator (generated - 1)
    then slightly overstates the tokens inside the interval and TTFT is
    slightly overstated. For short English prompts with Qwen3 this is
    expected to be rare; the streamer_chunks vs generated_tokens fields in
    the raw results make large divergences visible.
  - generated_tokens counts every token ID GenAI reports, including EOS and
    other special tokens: each one cost a decode step, so each belongs in the
    numerator. EOS early-stop means generated_tokens may be < requested.
  - If no streamer callback fired at all, there is no first-token boundary;
    the only honest interval is the whole generate() call, with all
    generated_tokens inside it.
  - Warm-up: none inside a measured run. benchmark.py performs one separate,
    uncounted warm-up generation before the measured runs.
"""
from __future__ import annotations


def measure_run(
    token_times: list[float],
    t_gen0: float,
    t_gen_end: float,
    generated_tokens: int | None,
    prompt_tokens: int | None = None,
) -> dict:
    """Account one measured generation run.

    token_times: perf_counter timestamps of streamer callbacks (timing only).
    t_gen0: perf_counter just before generate() was called.
    t_gen_end: perf_counter just after generate() returned.
    generated_tokens: len(result.tokens[0]) from EncodedResults (batch=1),
        or None if the result object carried no token IDs.
    prompt_tokens: tokenizer-encoded prompt length (prefill denominator).

    Returns a run dict fragment with generated_tokens, streamer_chunks,
    ttft_s, decode_s, decode_tok_s and prefill_tok_s.
    """
    n_callbacks = len(token_times)
    run: dict = {
        # Reference only -- NOT a token count (issue #1).
        'streamer_chunks': n_callbacks,
        'generated_tokens': generated_tokens,
    }

    if token_times:
        ttft: float | None = token_times[0] - t_gen0
        decode_start = token_times[0]
        decode_tokens = (
            None if generated_tokens is None else generated_tokens - 1
        )
    else:
        ttft = None
        decode_start = t_gen0
        decode_tokens = generated_tokens

    decode_s = max(t_gen_end - decode_start, 0.0)
    run['ttft_s'] = round(ttft, 4) if ttft is not None else None
    run['decode_s'] = round(decode_s, 4)
    run['decode_tok_s'] = (
        round(decode_tokens / decode_s, 2)
        if decode_tokens is not None and decode_tokens > 0 and decode_s > 0
        else None
    )
    run['prefill_tok_s'] = (
        round(prompt_tokens / ttft, 1)
        if ttft is not None and ttft > 0 and prompt_tokens
        else None
    )
    return run


def sanity_check(run: dict, requested_tokens: int | None = None) -> list[str]:
    """Cross-checks for one accounted run; returned as printable notes."""
    notes: list[str] = []
    generated = run.get('generated_tokens')
    callbacks = run.get('streamer_chunks')
    if generated is None:
        notes.append('generated token count unavailable from result object')
    elif callbacks is not None and callbacks != generated:
        notes.append(
            f'streamer callbacks ({callbacks}) != generated tokens '
            f'({generated}); throughput uses the generated-token count'
        )
    if (
        requested_tokens is not None
        and generated is not None
        and generated > requested_tokens
    ):
        notes.append(f'generated {generated} > requested {requested_tokens}')
    return notes
