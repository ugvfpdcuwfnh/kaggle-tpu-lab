"""Continuous batching (`glm53.scheduler.Scheduler`) on the tiny resident engine: requests admitted at different
times run batched, a follow-up turn extends a finished context kept on the chips, the cache-set budget parks the
LRU context into the host snapshot store and a later turn resumes it, cancellation stops a stream — every request's
tokens equal its own single-stream greedy generation."""
import threading
import time
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from glm53.scheduler import Request, Scheduler, SnapStore  # noqa: E402
from glm53.tests.test_batched_rows import build_engine  # noqa: E402

STOP = 7                                                  # a token id that ends a generation


def reference(eng, prompt, max_new):
    """Single-stream greedy generation (argmax) -> emitted tokens (stop token included when hit)."""
    logits, caches, pos = eng.prefill(np.asarray(prompt, np.int32)[None])
    out = [int(jnp.argmax(logits[0]))]
    while out[-1] != STOP and len(out) < max_new:
        logits, caches, pos = eng.decode(np.array([out[-1]]), caches, pos)
        out.append(int(jnp.argmax(logits[0])))
    return out


def make_sched(eng, max_streams, max_sets, log):
    state = {}
    snaps = SnapStore(eng, int(50e9), rows_bucket=8, log=log, state=state)
    sched = Scheduler(eng, {STOP}, max_streams=max_streams, max_sets=max_sets, snaps=snaps, log=log, state=state,
                      base_min=10 ** 9, snap_min=1, piece=32)
    sched.start()
    return sched, state


def wait(req, timeout=600):
    assert req.done.wait(timeout), "request did not finish"
    assert req.error is None, req.error
    return req


def test_live_cache_is_reclaimed_if_extension_prefill_raises():
    """A cache popped from live is owned by the admission transaction before prefill."""
    sched = Scheduler.__new__(Scheduler)
    caches = object()
    sched.live = OrderedDict([(1, {"ids": [1], "caches": caches, "pos": 1, "logits": None})])
    sched.state = {}
    sched._admission_caches = None
    sched.match_len = lambda ids, pos, prompt, quiet=True: (1, None)
    sched._prefill = lambda *args: (_ for _ in ()).throw(RuntimeError("prefill failed"))
    freed = []
    sched._free_set = freed.append

    with pytest.raises(RuntimeError, match="prefill failed"):
        sched._admit(SimpleNamespace(prompt=[1, 2]))

    assert freed == [caches]
    assert sched._admission_caches is None


def test_batched_requests_and_context_reuse():
    rng = np.random.default_rng(5)
    eng, V = build_engine(True, rng)
    logs = []
    sched, state = make_sched(eng, max_streams=2, max_sets=2, log=lambda *a: logs.append(" ".join(map(str, a))))
    try:
        p1, p2, p3 = (rng.integers(0, V, size=(n,)).tolist() for n in (40, 25, 33))
        r1 = sched.submit(Request(p1, 6, 0.0, 1.0, rid="r1"))
        r2 = sched.submit(Request(p2, 6, 0.0, 1.0, rid="r2"))
        r3 = sched.submit(Request(p3, 5, 0.0, 1.0, rid="r3"))          # queued: max_streams 2
        for r, p in ((r1, p1), (r2, p2), (r3, p3)):
            wait(r)
            assert r.out == reference(eng, p, r.max_new), r.rid
            assert r.stop_reason in ("stop", "length")
        assert state["step_tokens"] > state["steps"]                   # some steps really carried 2 rows
        # follow-up turn of r1: prompt = its prompt + everything it emitted -> the live context (on the chips) is
        # extended by exactly the last emitted token (never fed) -> 1-token prefill
        p4 = p1 + r1.out
        r4 = wait(sched.submit(Request(p4, 5, 0.0, 1.0, rid="r4")))
        assert r4.reused == len(p4) - 1 and r4.out == reference(eng, p4, 5)
        # only 2 cache sets fit: admitting r4 parked the LRU live context (r2's or r3's) into the host store
        assert state.get("snap_parks", 0) >= 1
        # a turn extending a PARKED context resumes it from the store
        parked = list(sched.snaps.entries.values())[0]
        for r, p in ((r2, p2), (r3, p3)):
            if list(parked["ids"]) == p + r.out[:-1]:
                p5 = p + r.out
                r5 = wait(sched.submit(Request(p5, 4, 0.0, 1.0, rid="r5")))
                assert state.get("snap_hits", 0) >= 1 and r5.reused == len(p5) - 1
                assert r5.out == reference(eng, p5, 4)
                break
        else:
            raise AssertionError("no parked entry matches a finished request")
    finally:
        sched.stop()


def test_cancel_and_capacity_error():
    rng = np.random.default_rng(6)
    eng, V = build_engine(True, rng)
    sched, state = make_sched(eng, max_streams=2, max_sets=3, log=lambda *a: None)
    try:
        got = []
        stop_after = threading.Event()
        p = rng.integers(0, V, size=(30,)).tolist()
        r = Request(p, 50, 0.0, 1.0, rid="c", on_token=lambda t: (got.append(t), stop_after.set() if len(got) == 3 else None))
        sched.submit(r)
        assert stop_after.wait(300)
        r.cancel()
        wait(r)
        assert r.stop_reason == "cancelled" and 3 <= len(r.out) <= 6
        ref = reference(eng, p, 50)
        assert r.out == ref[:len(r.out)]
        big = sched.submit(Request(list(range(eng.max_len)), 4, rid="big"))
        assert big.done.is_set() and big.stop_reason == "error"
    finally:
        sched.stop()


def test_budget_forces_tokens_in_a_batched_step():
    """A request with a budget (n, end, forced): when `end` was not emitted within its first n tokens, the next tokens
    are the forced ones (inside a 2-row step next to an unbudgeted stream), then generation continues from them;
    a budget that is not reached changes nothing."""
    rng = np.random.default_rng(9)
    eng, V = build_engine(True, rng)
    sched, state = make_sched(eng, max_streams=2, max_sets=2, log=lambda *a: None)
    try:
        p1, p2 = (rng.integers(0, V, size=(n,)).tolist() for n in (30, 22))
        ref1, ref2 = reference(eng, p1, 12), reference(eng, p2, 12)
        assert STOP not in ref1[:3], ref1
        end = next(t for t in range(8, V) if t not in ref1[:3])
        a, b = [t for t in range(8, V) if t != end][:2]
        r1 = sched.submit(Request(p1, 12, 0.0, 1.0, rid="b1", budget=(3, end, [a, b, end])))
        r2 = sched.submit(Request(p2, 12, 0.0, 1.0, rid="b2", budget=(50, end, [a, b, end])))
        wait(r1); wait(r2)
        assert r2.out == ref2, (r2.out, ref2)                                  # budget not reached: untouched
        assert r1.out[:3] == ref1[:3] and r1.out[3:6] == [a, b, end], r1.out    # the model's 3, then the forced 3
        cont = reference(eng, p1 + r1.out[:6], 6)                               # then the model continues from them
        assert r1.out[6:] == cont[:len(r1.out) - 6] and (len(r1.out) == 12 or r1.out[-1] == STOP), (r1.out, cont)
        assert state["step_tokens"] > state["steps"]                            # the two rows really shared steps
    finally:
        sched.stop()
