#!/usr/bin/env python3
"""Regression tests for benchmark token accounting (issue #1).

Pure Python: neither this file nor token_accounting.py imports OpenVINO
(benchmark.py itself cannot be imported by tests because of its forkserver
re-import guard, so the accounting lives in token_accounting.py and the
pipeline is faked here).

Run either way:
    python3 benchmarks/test_benchmark_accounting.py
    python3 -m pytest benchmarks/test_benchmark_accounting.py -q

Core regression being protected: the streamer callback count must NOT be
used as the generated-token count. The checked-in raw results show this
build's text streamer is not 1:1 with tokens (32 requested tokens -> 30
callbacks; 128 -> 101), so decode throughput must be derived from the token
IDs in the GenAI EncodedResults object instead.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_token_accounting():
    spec = importlib.util.spec_from_file_location(
        'token_accounting', HERE / 'token_accounting.py')
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


ta = _load_token_accounting()


class FakeClock:
    """Deterministic stand-in for time.perf_counter."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> float:
        self.now += dt
        return self.now


class TimingStreamer:
    """Clock-injected copy of benchmark.TimingStreamer (timing only)."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.token_times: list[float] = []

    def __call__(self, chunk: str) -> None:
        self.token_times.append(self.clock())


class FakeEncodedResults:
    """Stands in for genai EncodedResults: .tokens is [[id, id, ...]]."""

    def __init__(self, tokens: list[list[int]]) -> None:
        self.tokens = tokens
        self.texts = ['<fake generated text>']


class FakeGenerationConfig:
    """Like genai.GenerationConfig: NO streamer field (streamer is a
    generate() argument), and unknown attributes are rejected."""

    __slots__ = ('max_new_tokens',)

    def __init__(self, max_new_tokens: int) -> None:
        self.max_new_tokens = max_new_tokens


class FakePipeline:
    """Mimics this build's LLMPipeline.generate(TokenizedInputs, GenerationConfig).

    Emits ``n_generated`` token IDs in the result object but fires only
    ``n_callbacks`` streamer callbacks, reproducing the observed chunk
    merging/skipping (raw results: 32 requested -> 30 callbacks, 128 -> 101).
    The callbacks are spread evenly across ``decode_span_s`` (first callback
    at first token), and generate() returns ``tail_s`` after the last
    callback (final detokenization overhead).
    """

    def __init__(self, n_generated: int, n_callbacks: int, clock: FakeClock,
                 prefill_s: float = 0.25, decode_span_s: float = 0.72,
                 tail_s: float = 0.05) -> None:
        assert 1 <= n_callbacks <= n_generated
        self.n_generated = n_generated
        self.n_callbacks = n_callbacks
        self.clock = clock
        self.prefill_s = prefill_s
        self.decode_span_s = decode_span_s
        self.tail_s = tail_s

    def generate(self, tokenized, generation_config: FakeGenerationConfig,
                 streamer=None) -> FakeEncodedResults:
        self.clock.advance(self.prefill_s)  # prefill before first token
        for k in range(self.n_callbacks):
            if streamer is not None:
                streamer('<chunk>')
            if k < self.n_callbacks - 1:
                self.clock.advance(self.decode_span_s / (self.n_callbacks - 1))
        self.clock.advance(self.tail_s)
        return FakeEncodedResults([[1000 + i for i in range(self.n_generated)]])


def run_once(pipe: FakePipeline, clock: FakeClock, max_new: int,
             prompt_tokens: int = 5) -> dict:
    """Drive the fake pipeline exactly the way benchmark.py drives the real one."""
    streamer = TimingStreamer(clock)
    cfg = FakeGenerationConfig(max_new)
    t_gen0 = clock()
    out = pipe.generate(None, generation_config=cfg, streamer=streamer)
    t_gen_end = clock()
    run = {'requested_tokens': max_new, 'wall_s': round(t_gen_end - t_gen0, 3)}
    run.update(ta.measure_run(streamer.token_times, t_gen0, t_gen_end,
                              len(out.tokens[0]), prompt_tokens))
    return run


def old_callback_formula(token_times: list[float]) -> float:
    """The pre-fix (buggy) decode throughput: callbacks as token count."""
    return (len(token_times) - 1) / (token_times[-1] - token_times[0])


def test_decode_numerator_is_generated_token_ids() -> None:
    # Raw-results case: 32 requested -> 30 callbacks.
    clock = FakeClock()
    run = run_once(FakePipeline(n_generated=32, n_callbacks=30, clock=clock), clock, 32)
    assert run['requested_tokens'] == 32
    assert run['generated_tokens'] == 32      # from result.tokens, not callbacks
    assert run['streamer_chunks'] == 30       # recorded for reference only
    # Decode interval: first callback -> generate() return (0.72 + 0.05 s).
    assert run['decode_s'] == 0.77
    # Numerator is 31 generated tokens after the first, NOT 29 callbacks.
    assert run['decode_tok_s'] == round(31 / 0.77, 2)
    assert run['decode_tok_s'] != round(29 / 0.72, 2)  # old buggy value
    # TTFT and prefill unchanged in meaning.
    assert run['ttft_s'] == 0.25
    assert run['prefill_tok_s'] == round(5 / 0.25, 1)


def test_callback_count_divergence_is_not_silently_absorbed() -> None:
    # Raw-results case: 128 requested -> 101 callbacks. Here the callback
    # formula and the token formula disagree materially (~25%): if anyone
    # reverts the numerator to the callback count, this test fails.
    clock = FakeClock()
    pipe = FakePipeline(n_generated=128, n_callbacks=101, clock=clock,
                        decode_span_s=2.43, tail_s=0.01)
    streamer_times: list[float] = []

    class CapturingStreamer(TimingStreamer):
        def __call__(self, chunk: str) -> None:
            super().__call__(chunk)
            streamer_times.append(self.token_times[-1])

    streamer = CapturingStreamer(clock)
    cfg = FakeGenerationConfig(128)
    t_gen0 = clock()
    out = pipe.generate(None, generation_config=cfg, streamer=streamer)
    t_gen_end = clock()
    run = {'requested_tokens': 128, 'wall_s': round(t_gen_end - t_gen0, 3)}
    run.update(ta.measure_run(streamer.token_times, t_gen0, t_gen_end,
                              len(out.tokens[0]), 5))
    assert run['generated_tokens'] == 128 and run['streamer_chunks'] == 101
    correct = 127 / run['decode_s']
    buggy = old_callback_formula(streamer_times)
    assert abs(run['decode_tok_s'] - correct) < 0.01
    assert abs(correct - buggy) / buggy > 0.2  # materially different numbers
    # And the divergence is surfaced as a note, not hidden.
    notes = ta.sanity_check(run, 128)
    assert any('!= generated tokens' in n and '101' in n and '128' in n for n in notes)


def test_no_streamer_callbacks_still_accounts_tokens() -> None:
    # If the streamer never fires, the honest interval is the whole
    # generate() call with every generated token inside it.
    clock = FakeClock()
    pipe = FakePipeline(n_generated=10, n_callbacks=1, clock=clock,
                        prefill_s=0.2, decode_span_s=0.0, tail_s=0.3)
    # n_callbacks=1: single callback; then force the no-callback shape directly.
    run = ta.measure_run([], t_gen0=0.0, t_gen_end=0.5, generated_tokens=10,
                         prompt_tokens=5)
    assert run['streamer_chunks'] == 0
    assert run['ttft_s'] is None
    assert run['prefill_tok_s'] is None
    assert run['decode_s'] == 0.5
    assert run['decode_tok_s'] == round(10 / 0.5, 2)
    _ = pipe  # pipeline kept for symmetry; not needed for this shape


def test_single_token_generation_has_no_decode_rate() -> None:
    run = ta.measure_run([1.0], t_gen0=0.9, t_gen_end=1.2, generated_tokens=1,
                         prompt_tokens=5)
    assert run['decode_tok_s'] is None
    assert run['generated_tokens'] == 1


def test_missing_token_ids_disable_throughput_not_timing() -> None:
    # Fallback path (string prompt): no token IDs -> no throughput claims.
    run = ta.measure_run([1.0, 1.1, 1.2], t_gen0=0.9, t_gen_end=1.3,
                         generated_tokens=None, prompt_tokens=5)
    assert run['generated_tokens'] is None
    assert run['ttft_s'] == 0.1
    assert run['decode_tok_s'] is None
    assert any('unavailable' in n for n in ta.sanity_check(run, 32))


def test_generated_over_requested_is_flagged() -> None:
    run = ta.measure_run([1.0], t_gen0=0.9, t_gen_end=1.2, generated_tokens=5,
                         prompt_tokens=5)
    assert any('generated 5 > requested 4' in n
               for n in ta.sanity_check(run, requested_tokens=4))


def test_benchmark_source_uses_token_ids_not_callbacks() -> None:
    # benchmark.py cannot be imported (forkserver guard + OpenVINO), so check
    # its source: the measured-run path must take the token count from the
    # result object, pass the streamer to generate() (GenerationConfig has no
    # streamer field), and never use the callback count as a numerator.
    import ast
    src = (HERE / 'benchmark.py').read_text()
    tree = ast.parse(src)
    assert 'len(streamer.token_times)' not in src
    assert 'len(out.tokens[0])' in src
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                assert not (isinstance(t, ast.Attribute) and t.attr == 'streamer'), \
                    'streamer must be passed to generate(), not set on the config'
    gen_calls = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == 'generate' and n.args
                 and isinstance(n.args[0], ast.Name) and n.args[0].id == 'tokenized']
    assert gen_calls, 'tokenized-input generate() call not found'
    assert all(any(k.arg == 'streamer' for k in c.keywords) for c in gen_calls)


def _main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith('test_') and callable(fn)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f'PASS {name}')
        except AssertionError as e:
            failures += 1
            print(f'FAIL {name}: {e}')
    print(f'{len(tests) - failures}/{len(tests)} passed')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(_main())
