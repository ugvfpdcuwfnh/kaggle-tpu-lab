"""Continuous batching for `ResidentLayerEngine.decode_rows`: independent streams decoded together, admitted between
steps, prefilled piece by piece with decode steps of the running streams in between, each keeping its own cache set.

One engine thread (`Scheduler.run`) owns every JAX call. Client threads `submit` a `Request` and wait on its
`done` event (tokens arrive through `on_token`, called on the engine thread). Cache sets on the chips = the active
streams' + finished contexts kept LIVE for the next turn (LRU), at most `max_sets` in total; beyond that, finished
contexts are parked in host memory (`SnapStore`) and resumed when a later prompt extends them. Between steps only the
sampled ids cross to the host: token ids, positions and the sampler state stay on the device.
"""
import collections
import threading
import time

import numpy as np
import jax
import jax.numpy as jnp

from glm53.engine import DeviceSampler


def match_prefix(live_ids, pos, prompt, quiet=False):
    """Default matcher: (k, fed) when the context (live_ids[:pos]) is a prefix of `prompt`: prompt[k:] must still be
    prefilled and `fed` = the ids the engine will have seen after that; (0, None) otherwise."""
    if pos == 0 or len(prompt) < pos:
        return 0, None
    a = np.asarray(live_ids[:pos]); b = np.asarray(prompt[:pos])
    if a.shape != b.shape or not np.array_equal(a, b):
        return 0, None
    return pos, list(live_ids[:pos]) + list(prompt[pos:])


class Request:
    """A generation request. `prompt`: token ids (the server's signature ids: image tokens negative, `imgs` maps
    them; the scheduler's `feed` turns slices into real ids + embedding overrides). Results: `out` (emitted tokens,
    the stop token included when hit), `stop_reason` ("stop" | "length" | "cancelled" | "error"), timings."""

    def __init__(self, prompt, max_new, temperature=1.0, top_p=1.0, imgs=None, on_token=None, rid=None, on_done=None,
                 budget=None):
        self.prompt = [int(t) for t in prompt]
        self.max_new, self.temperature, self.top_p = int(max_new), float(temperature), float(top_p)
        self.imgs, self.on_token, self.rid, self.on_done = imgs, on_token, rid, on_done
        self.budget = budget            # (n, end_id, forced_ids): unless `end_id` was emitted within the first n output
                                        # tokens, the next tokens are forced to `forced_ids` (a thinking budget)
        self.out = []
        self.done = threading.Event()
        self.error = None
        self.cancelled = False
        self.stop_reason = None
        self.reused = 0
        self.prefill_s = 0.0
        self.t_submit, self.t_first, self.t_end = time.time(), None, None

    def cancel(self):
        self.cancelled = True

    @property
    def decode_s(self):
        return (self.t_end or time.time()) - (self.t_first or self.t_submit)


class _Stream:
    """An admitted request: its own cache set, its position, the ids fed so far and the last (unfed) token."""

    def __init__(self, req, caches, pos, fed, tok, logits):
        self.req, self.caches, self.pos, self.fed, self.tok = req, caches, pos, list(fed), tok
        self.seen = []                                   # tokens fed beyond `fed`
        self.logits = logits                             # logits [1,V] after the last fed token (predict `tok`)
        self.max_new = req.max_new
        self.open = req.budget is not None               # the budgeted block (thinking) is still open
        self.force = []                                  # tokens to feed instead of the sampled ones (budget reached)


class SnapStore:
    """Host-memory LRU of compact context snapshots keyed by their token ids (see `ResidentLayerEngine.snapshot_prefix`).
    `park` moves a context off the chips, `lookup` finds the entry a prompt extends (the matcher handles dropped
    thinking blocks and boundary drift when the server passes its own), `restore` brings it back into fresh
    full-capacity caches. A parked conversation keeps its two latest turn boundaries; older strict prefixes are
    dropped unless pinned (pinned = a conversation's system section, the branch point new sessions start from)."""

    def __init__(self, eng, max_bytes, rows_bucket=1024, match_len=match_prefix, log=print, state=None):
        self.eng, self.max_bytes, self.rows_bucket = eng, max_bytes, rows_bucket
        self.match_len, self.log = match_len, log
        self.state = state if state is not None else {}
        self.entries = collections.OrderedDict()                     # tuple(ids) -> entry, oldest first
        self.bytes = 0

    def park(self, ids, caches, pos, logits, pinned=False):
        """Snapshot `caches` (a context of `pos` tokens = `ids`) into host memory; the caches stay valid."""
        key = tuple(int(t) for t in ids)
        if key in self.entries:
            e = self.entries[key]
            e["pinned"] = e["pinned"] or pinned
            if logits is not None and e["logits"] is None:
                e["logits"] = np.asarray(logits)
            self.entries.move_to_end(key)
            return e
        t = time.time()
        snap = self.eng.snapshot_prefix(caches, pos, rows_bucket=self.rows_bucket)
        host = self.eng.snapshot_to_host(snap)
        del snap
        e = {"ids": list(key), "n": pos, "snap": host, "logits": None if logits is None else np.asarray(logits),
             "bytes": host["bytes"], "pinned": pinned}
        if not pinned:                                                # supersede our own earlier turns but the latest
            older = sorted((k for k, o in self.entries.items() if not o["pinned"] and len(k) < len(key) and key[:len(k)] == k),
                           key=len)
            for k in older[:-1]:
                self._drop(k)
        self.entries[key] = e
        self.bytes += e["bytes"]
        while self.bytes > self.max_bytes and len(self.entries) > 1:
            victims = [k for k, o in self.entries.items() if not o["pinned"]] or list(self.entries)
            self._drop(victims[0])
        st = self.state
        st["snap_parks" if not pinned else "snap_pins"] = st.get("snap_parks" if not pinned else "snap_pins", 0) + 1
        st["snap_entries"], st["snap_bytes"] = len(self.entries), self.bytes
        st["snap_s"] = st.get("snap_s", 0.0) + time.time() - t
        self.log(f"parked {pos} tokens ({e['bytes'] / 1e6:.0f} MB, {'pinned' if pinned else 'lru'}) in {time.time() - t:.2f}s; "
                 f"store {len(self.entries)} entries, {self.bytes / 1e9:.2f} GB")
        return e

    def _drop(self, key):
        e = self.entries.pop(key)
        self.bytes -= e["bytes"]

    def lookup(self, prompt):
        """-> (k, fed, entry) for the entry that covers most of `prompt`, or (0, None, None)."""
        best = (0, None, None)
        for key, e in list(self.entries.items()):
            if e["n"] <= best[0]:
                continue
            k, fed = self.match_len(e["ids"], e["n"], prompt, quiet=True)
            if k > best[0] and not (k == len(prompt) and e["logits"] is None):
                best = (k, fed, e)
        if best[2] is not None:
            self.entries.move_to_end(tuple(best[2]["ids"]))
        return best

    def restore(self, e):
        t = time.time()
        caches = self.eng.restore_prefix(self.eng.snapshot_from_host(e["snap"]))
        self.state["snap_hits"] = self.state.get("snap_hits", 0) + 1
        self.state["snap_s"] = self.state.get("snap_s", 0.0) + time.time() - t
        return caches


def _argmax_first(logits_row, temperature, top_p):
    return int(np.argmax(logits_row))


class Scheduler:
    """See the module docstring. `feed(req, a, b) -> (ids [n] int32, embeds or None)` renders req.prompt[a:b] for
    the engine (the server attaches its image data to the request); `match_len(live_ids, pos, prompt, quiet=False) -> (k, fed)`; `system_end(prompt)` = length of
    the system section worth pinning (0 = none); `first_sample(logits_row, temperature, top_p) -> int` samples the
    first token of a request on the host (the rest come from the device sampler)."""

    def __init__(self, eng, stop_ids, max_streams=4, max_sets=None, feed=None, match_len=match_prefix,
                 system_end=None, first_sample=_argmax_first, snaps=None, log=print, state=None,
                 base_min=512, snap_min=256, piece=None, seed=None, min_free_gb=0.65, max_wait_s=90.0, engine_lock=None):
        self.eng, self.stop_ids = eng, set(int(t) for t in stop_ids)
        self.max_streams = max(1, int(max_streams))
        self.max_sets = max(self.max_streams, int(max_sets if max_sets is not None else max_streams + 1))
        self.feed = feed or (lambda req, a, b: (np.asarray(req.prompt[a:b], np.int32), None))
        self.match_len, self.system_end = match_len, (system_end or (lambda prompt: 0))
        self.first_sample = first_sample
        self.snaps = snaps if snaps is not None else SnapStore(eng, 0, 1, match_len, log, state)
        self.log, self.state = log, (state if state is not None else {})
        self.base_min, self.snap_min = base_min, snap_min
        self.piece = piece or eng.prefill_piece
        self.min_free_gb = min_free_gb                  # HBM headroom an admission needs (prefill temporaries); 0 = off
        self.max_wait_s = max_wait_s                    # a request queued longer than this fails ("queue_timeout")
        self.engine_lock = engine_lock or threading.RLock()  # serializes every JAX/TPU operation with vision work
        self.live = collections.OrderedDict()            # finished contexts on the chips (LRU): key -> ctx dict
        self._live_n = 0
        self.active = []
        self.pending = collections.deque()
        self.cond = threading.Condition()
        self.sampler = DeviceSampler(1.0, 1.0, seed=seed)
        self._members = None                             # ids of the streams the device state below belongs to
        self._toks = self._pos = None
        self._stop = False
        self._stopping = False
        self._paused, self._idle = False, threading.Event()
        self._waits = 0
        self._t_step = None
        self.thread = None
        self.steps = 0

    # ---- client side
    def submit(self, req: Request):
        if len(req.prompt) + 2 > self.eng.max_len:
            self._fail(req, ValueError(f"prompt of {len(req.prompt)} tokens exceeds the context capacity {self.eng.max_len}"))
            return req
        with self.cond:
            if self._stopping:
                self._fail(req, RuntimeError("scheduler is shutting down"), "shutdown")
                return req
            self.pending.append(req)
            self.cond.notify()
        return req

    def start(self):
        self.thread = threading.Thread(target=self.run, daemon=True, name="glm-scheduler")
        self.thread.start()
        return self.thread

    def stop(self, timeout=60.0):
        """Atomically close admission and wake every queued client before joining.

        A bounded join prevents a stuck JAX call from making HTTP handlers wait
        forever. Cache release is deferred until the engine worker has really exited.
        """
        with self.cond:
            self._stopping = self._stop = True
            pending = list(self.pending)
            self.pending.clear()
            self.cond.notify_all()
        for req in pending:
            self._fail(req, RuntimeError("scheduler is shutting down"), "shutdown")
        if self.thread is not None:
            self.thread.join(timeout=timeout)
            if self.thread.is_alive():
                self.log(f"scheduler did not stop within {timeout:.0f}s; retaining caches until worker exits")
                return False
        for ctx in self.live.values():
            self._free_set(ctx["caches"])
        for stream in self.active:
            self._free_set(stream.caches)
        self.live.clear(); self.active.clear(); self._toks = self._pos = None
        return True

    @staticmethod
    def _free_set(caches):
        """Release a dropped cache set's HBM now (`Array.delete`), whatever else still references the arrays: a
        dropped set kept alive by a stray reference (seen once after a concurrent burst, 2026-09-13) cost a stream
        slot for the rest of the session — refcounts alone are not a guarantee."""
        for x in jax.tree.leaves(caches):
            try:
                x.delete()
            except Exception:  # noqa: BLE001
                pass

    @property
    def n_sets(self):
        return len(self.active) + len(self.live)

    def free_gb(self):
        """Free HBM on chip 0 in GB, or None when the backend reports no memory statistics (CPU tests)."""
        st = self.eng.mesh.devices.flat[0].memory_stats() or {}
        return (st["bytes_limit"] - st["bytes_in_use"]) / 1e9 if "bytes_limit" in st else None

    def _headroom(self):
        """True when an admission can allocate a cache set and run its prefill: parks live contexts (LRU) until the
        set budget and the HBM margin allow it; False when only active streams hold the chips (wait for one)."""
        self._make_room()
        f = self.free_gb()
        return f is None or not self.min_free_gb or f >= self.min_free_gb

    # ---- engine thread
    def pause(self, timeout=60.0):
        """Stop the engine thread between steps (running streams stall, nothing is admitted) until `resume`; returns
        once the thread is idle. For probe jobs that need the chips."""
        with self.cond:
            self._paused = True
            self.cond.notify_all()
        return self._idle.wait(timeout)

    def resume(self):
        with self.cond:
            self._paused = False
            self.cond.notify_all()

    def run(self):
        while not self._stop:
            with self.cond:
                while not self._stop and (self._paused or (not self.pending and not self.active)):
                    if self._paused:
                        self._idle.set()
                    self._t_step = None
                    self.cond.wait(timeout=1.0)
                self._idle.clear()
                if self._stop:
                    break
                req = None
                while self.pending and self.max_wait_s and time.time() - self.pending[0].t_submit > self.max_wait_s:
                    old = self.pending.popleft()                        # queued too long: fail it (the client can retry)
                    self._fail(old, TimeoutError(f"queued for {time.time() - old.t_submit:.0f}s without a free cache set"), "queue_timeout")
                if self.pending and len(self.active) < self.max_streams and self._headroom():
                    req = self.pending.popleft()
                elif self.pending and not self.active:                  # nothing running and still no room
                    if self._waits % 10 == 0:
                        self.log(f"admission waits: {len(self.live)} live contexts, free HBM {self.free_gb()} GB")
                    self._waits += 1
                    self.cond.wait(timeout=1.0)
            if req is not None:
                if req.cancelled:
                    self._fail(req, None, "cancelled"); continue
                try:
                    with self.engine_lock:
                        self._admit(req)
                except Exception as e:  # noqa: BLE001
                    import gc, traceback
                    self.log("admission failed:", traceback.format_exc()[-1500:])
                    self._fail(req, RuntimeError(f"{type(e).__name__}: {str(e)[:800]}"))   # no traceback: its frames
                    del e                                                                    # would pin the half-built caches
                    gc.collect()
                continue
            if self.active:
                try:
                    with self.engine_lock:
                        self._step()
                except Exception as e:  # noqa: BLE001
                    import gc, traceback
                    self.log("decode step failed:", traceback.format_exc()[-1500:])
                    err = RuntimeError(f"{type(e).__name__}: {str(e)[:800]}")
                    del e
                    for s in list(self.active):
                        s.req.error = err
                        self._finish(s, "error", keep=False)
                    self.active = []; self._members = None; self._toks = self._pos = None
                    gc.collect()

    def _fail(self, req, error, reason="error"):
        """Finish a request that never became a stream (capacity error, admission failure, cancelled while queued)."""
        req.error, req.stop_reason, req.t_end = error, reason, time.time()
        if req.on_done is not None:
            try:
                req.on_done()
            except Exception:  # noqa: BLE001
                pass
        req.done.set()

    def _make_room(self):
        """Park LRU live contexts until a cache set can be allocated within the set budget AND the HBM margin
        (`min_free_gb`, prefill temporaries); a parked set is freed (donated caches)."""
        while self.live:                                                 # (no gc.collect() here: a full collection
            f = self.free_gb()                                           #  costs ~14 s in a process with hundreds of
            if self.n_sets < self.max_sets and (f is None or not self.min_free_gb or f >= self.min_free_gb):
                break                                                    #  compiled programs; refcounts free the set)
            key, ctx = self.live.popitem(last=False)
            if ctx["pos"] >= self.snap_min:
                self.snaps.park(ctx["ids"], ctx["caches"], ctx["pos"], ctx["logits"])
            self._free_set(ctx["caches"])
            del ctx

    def _prefill(self, req, a, b, caches, pos):
        """prompt[a:b] into `caches` at `pos`, one piece at a time with a decode step of the running streams between
        pieces (chunked admission). Returns (logits, caches, pos)."""
        logits = None
        for x in range(a, b, self.piece):
            y = min(b, x + self.piece)
            ids, emb = self.feed(req, x, y)
            ids = np.asarray(ids, np.int32).reshape(1, -1)
            logits, caches, pos = self.eng.prefill(ids, caches, pos, embeds=emb)
            if y < b and self.active:
                self._step()
        return logits, caches, pos

    def _admit(self, req):
        prompt = req.prompt
        t0 = time.time()
        st = self.state
        # 1. a finished context still on the chips that the prompt extends
        best = (0, None, None)
        for key, c in self.live.items():
            k, fed = self.match_len(c["ids"], c["pos"], prompt, quiet=True)
            if k > best[0] and not (k == len(prompt) and c["logits"] is None):
                best = (k, fed, key)
        c = None                                                         # (keep no reference to a context _make_room may drop)
        if best[2] is None and self.live:
            self.log(f"no live context matched ({len(self.live)} live, {len(self.snaps.entries)} parked)")
        if best[2] is not None:
            k, fed, key = best
            ctx = self.live.pop(key)
            st["prefix_hits"] = st.get("prefix_hits", 0) + 1
            st["prefix_tokens_reused"] = st.get("prefix_tokens_reused", 0) + k
            if k == len(prompt):
                logits, caches, pos = ctx["logits"], ctx["caches"], ctx["pos"]
            else:
                logits, caches, pos = self._prefill(req, k, len(prompt), ctx["caches"], ctx["pos"])
            req.reused = k
        else:
            self._make_room()
            k, fed, entry = self.snaps.lookup(prompt)
            if k > 0:                                                    # 2. a parked context the prompt extends
                st["prefix_tokens_reused"] = st.get("prefix_tokens_reused", 0) + k
                caches = self.snaps.restore(entry)
                if k == len(prompt):
                    logits, pos = entry["logits"], entry["n"]
                else:
                    logits, caches, pos = self._prefill(req, k, len(prompt), caches, entry["n"])
                req.reused = k
            else:                                                        # 3. from scratch (pin the system section)
                fed = list(prompt)
                n_sys = self.system_end(prompt)
                if n_sys >= self.base_min and n_sys < len(prompt):
                    _, caches, pos = self._prefill(req, 0, n_sys, None, 0)
                    self.snaps.park(prompt[:n_sys], caches, pos, None, pinned=True)
                    logits, caches, pos = self._prefill(req, n_sys, len(prompt), caches, pos)
                else:
                    logits, caches, pos = self._prefill(req, 0, len(prompt), None, 0)
        jax.block_until_ready(logits)
        self._t_step = None
        req.prefill_s = time.time() - t0
        st["prefill_s"] = st.get("prefill_s", 0.0) + req.prefill_s
        tok = self.first_sample(np.asarray(logits)[0], req.temperature, req.top_p)
        s = _Stream(req, caches, pos, fed, tok, logits)
        s.max_new = max(1, min(req.max_new, self.eng.max_len - pos - 1))
        req.t_first = time.time()
        self.active.append(s); self._members = None
        self._emit(s, tok)

    def _emit(self, s, tok):
        req = s.req
        req.out.append(tok)
        if s.open and tok == req.budget[1]:
            s.open = False
        if req.on_token is not None and not req.cancelled:
            try:
                req.on_token(tok)
            except Exception as e:  # noqa: BLE001
                self.log(f"on_token failed ({e!r}): cancelling {req.rid}")
                req.cancelled = True
        if tok in self.stop_ids:
            self._finish(s, "stop")
        elif len(req.out) >= s.max_new or s.pos + 1 >= self.eng.max_len:
            self._finish(s, "length")
        elif req.cancelled:
            self._finish(s, "cancelled")

    def _finish(self, s, reason, keep=True):
        """The stream's context becomes a live (on-chip) context: ids = everything the engine has seen (the last
        emitted token is never fed), logits = the prediction after them."""
        req = s.req
        if s in self.active:
            self.active.remove(s); self._members = None
        if not keep:
            self._free_set(s.caches)
        if keep:
            self._live_n += 1
            lg = s.logits[0][s.logits[1]:s.logits[1] + 1] if isinstance(s.logits, tuple) else s.logits
            self.live[self._live_n] = {"ids": s.fed + s.seen, "caches": s.caches, "pos": s.pos, "logits": lg}
        req.stop_reason = req.stop_reason or reason
        req.t_end = time.time()
        self.state["tokens"] = self.state.get("tokens", 0) + len(req.out)
        if req.on_done is not None:
            try:
                req.on_done()
            except Exception as e:  # noqa: BLE001
                self.log(f"on_done failed ({e!r}) for {req.rid}")
        req.done.set()

    def _step(self):
        """One batched decode step of the active streams."""
        for s in list(self.active):
            if s.req.cancelled:
                self._finish(s, "cancelled")
        act = list(self.active)                                         # (streams finishing below leave self.active)
        if not act:
            return
        B = len(act)
        members = tuple(id(s) for s in act)
        if members != self._members:                                    # membership changed: re-place the small state
            self._pos = self.eng.device_positions([s.pos for s in act])
            self._toks = jnp.asarray([s.tok for s in act], jnp.int32)
            self.sampler.set([s.req.temperature for s in act], [s.req.top_p for s in act])
            self._members = members
        t0 = time.perf_counter()
        ids, logits, sets, pos_next = self.eng.decode_rows(self._toks, [s.caches for s in act], self._pos, self.sampler)
        ids_h = np.asarray(ids)                                         # the one device -> host sync per step
        forced = False
        for r, s in enumerate(act):                                     # a budget reached: feed its forced tokens
            if s.open and not s.force and len(s.req.out) >= s.req.budget[0]:
                s.force = list(s.req.budget[2])
            if s.force:
                if not forced:
                    ids_h, forced = np.array(ids_h), True
                ids_h[r] = s.force.pop(0)
        if forced:
            ids = jnp.asarray(ids_h, jnp.int32)
        t1 = time.perf_counter()
        self._toks, self._pos = ids, pos_next
        self.steps += 1
        st = self.state
        st["steps"] = self.steps
        st["step_tokens"] = st.get("step_tokens", 0) + B
        st["decode_s"] = st.get("decode_s", 0.0) + (t1 - t0)          # inside the engine (dispatch + device + sync)
        if self._t_step is not None:                                    # back-to-back steps: the loop's own overhead
            st["step_s"] = st.get("step_s", 0.0) + (t1 - self._t_step)
            st["steps_busy"] = st.get("steps_busy", 0) + 1
        self._t_step = t1
        for r, s in enumerate(act):                                     # every row takes its new caches first
            s.caches, s.pos = sets[r], s.pos + 1
            s.seen.append(s.tok)
            s.tok = int(ids_h[r])
            s.logits = (logits, r)                                      # sliced only when the stream finishes
        for s in act:
            self._emit(s, s.tok)

    # ---- warm-up
    def warmup(self, max_B=None, token=0):
        """Compile the batched decode programs for every batch size up to max_streams (zero caches, position 0)."""
        max_B = max_B or self.max_streams
        for B in range(1, max_B + 1):
            t = time.time()
            sets = [self.eng.alloc_caches(1) for _ in range(B)]
            samp = DeviceSampler([1.0] * B, [0.95] * B, seed=0)
            ids, lg, sets, pos = self.eng.decode_rows(np.full((B,), token, np.int32), sets, [0] * B, samp)
            jax.block_until_ready(ids)
            del sets
            self.log(f"warm-up batched decode B={B}: {time.time() - t:.0f}s")
