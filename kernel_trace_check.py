#!/usr/bin/env python3
"""kernel_trace_check: read a torch.profiler trace and name the one lever its GPU time points at.

    python3 kernel_trace_check.py trace.json.gz

Input: the Chrome-trace JSON that torch.profiler (Kineto) exports, plain or gzipped, a folder
of them (the largest one that holds kernels is read), or '-' for stdin. vLLM's --profiler-config
traces are the same format. Kernel events carry grid, block and blocks per SM, and link to the
CPU call that launched them through args "correlation".

Windows are the profiled steps (ProfilerStep#N, their CPU spans). Inside them, a stretch where
no kernel, copy or fill runs on any stream is a gap. A gap is HOST WAIT when the call that
launched the next GPU work started after the GPU went idle (the test Holistic Trace Analysis
applies per stream); otherwise that work was already queued and the gap is OTHER idle.

The rule stops at the first line that applies:
  refuse   no kernels; every window a warm-up (cudaMalloc, stream creation); too few kernels;
           NCCL over 50% of kernel time
  (a)      host wait before eager kernel launches >= 30% of step time: CUDA graphs
  refuse   idle no launch change reaches (work already queued, the CPU blocked in a CUDA call or
           waiting on the DataLoader, the CPU busy between graph replays) >= 30% of step time
  (b)      kernels on grids under 1 block per SM >= 25% of compute-kernel time: a bigger batch
           or a rounded shape
  (c)      eager elementwise kernels >= 40% of compute-kernel time, or >= 25% and more than matmul
           and attention together: fusion
  (d)      otherwise: leave the kernels alone, and say which group holds the time

Standard library only. Read-only. Exit 0 on a lever, 2 on INCONCLUSIVE, 1 on unreadable input.
"""
import argparse
import bisect
import gzip
import json
import os
import re
import statistics
import sys
import textwrap

__version__ = "1.1.0"

# The bars. README.md ("Why the bars sit where they do") gives the reason for each.
HOST_WAIT_MIN = 0.30      # (a) host wait before eager launches / step time
UNREACHED_MAX = 0.30      # refuse: (queued + blocked + data + between graph replays) idle / step time
NCCL_MAX = 0.50           # refuse: collective kernels / kernel time
SMALL_GRID_MIN = 0.25     # (b) compute kernels under 1 block per SM / compute-kernel time
ELEMENTWISE_MIN = 0.40    # (c) eager elementwise kernels / compute-kernel time
ELEMENTWISE_OVER_MATH = 0.25  # (c) also fires from here when elementwise outweighs matmul + attention
MIN_KERNELS = 50          # refuse: fewer kernels than this in the windows judged
SPLIT_MS = 50.0           # no step markers: split the GPU timeline at idle stretches this long

GPU_CATS = {"kernel": "kernel", "gpu_memcpy": "copy", "memcpy": "copy",
            "gpu_memset": "fill", "memset": "fill"}
API_CATS = {"cuda_runtime", "runtime", "cuda_driver"}

# CUDA and HIP calls on the CPU. A launch hands work to the GPU; a warm-up call means the process
# is still setting up; a blocking call can hold the CPU until the GPU drains.
KERNEL_LAUNCH = re.compile(r"Launch")
ANY_LAUNCH = re.compile(r"Launch|Memcpy|Memset")
GRAPH_LAUNCH = re.compile(r"GraphLaunch")
WARMUP = re.compile(r"^(cudaMalloc|cudaFree|cuMemAlloc|cuMemAlloc_v2|cuMemFree|cuMemFree_v2|hipMalloc|hipFree)$"
                    r"|StreamCreate|GetDeviceProperties|^cuModuleLoad|^cuLibraryLoad|^hipModuleLoad")
BLOCKING = re.compile(r"Synchronize|Memcpy|^(cudaMalloc|cudaFree|cuMemAlloc|cuMemAlloc_v2|cuMemFree|"
                      r"cuMemFree_v2|hipMalloc|hipFree)$")

# Kernel classes, matched on the lowercased name without its argument list.
COMMS = re.compile(r"nccl|rccl|cross_device_reduce|all_?reduce|all_?gather|reduce_?scatter|all_?to_?all")
NOT_MATH = re.compile(r"reshape_and_cache|concat_and_cache|append_?paged_?kv|sampling|rotary|rope|norm")
ATTENTION = re.compile(r"flash|fmha|attention|attn|sdpa|(?<![a-z])mla(?![a-z])")
MATMUL = re.compile(r"(?<!fb)gemm|gemv|xmma|cutlass|cublas|scudnn|wgmma|matmul|conv(?!ert)|implicit|dgrad|wgrad|"
                    r"nvjet|marlin|machete|fused_moe|bmm|winograd|fft|splitk|^cijk_")
# Eager PyTorch kernels that torch.compile regenerates and fuses: elementwise ops, reductions,
# norms, softmax, copies, concatenation, fills. Library kernels that are fused already (embedding
# lookups, sorts, multi-tensor optimizers, vLLM custom ops) are left out.
ELEMENTWISE = re.compile(r"at::native::|elementwise_kernel|reduce_kernel|softmax_warp|cunn_|cudnn::bn_|"
                         r"cuapply(layer|rms)norm|cucompute(part)?grad")
LIBRARY = re.compile(r"fbgemm|embedding|multi_tensor_apply|cub::|radix|sort|nonzero|unique|topk|segment_|vllm::")
FP32_MATMUL = re.compile(r"sgemm|scudnn")   # FP32 kernels without tensor cores (TF32 ones read s1688 or tf32)

CLASSES = ("matmul", "attention", "elementwise", "other memory-bound", "compiled", "NCCL")
GAP_KINDS = ("launch", "replay", "blocked", "data", "queued")
DATALOADER = "enumerate(DataLoader)"   # PyTorch's own annotation around each DataLoader __next__


class TraceError(Exception):
    """Input that is not a readable torch.profiler trace; main() turns it into exit code 1."""


# ---------------------------------------------------------------- reading

def _compact(ev):
    """One event dict as a small tuple holding only the fields this tool reads, or None when the
    tool never reads the event. Called while the JSON parses, so a large trace stays small in memory:
      ("g", kind, name, ts, dur, device, stream, correlation, grid, block, blocks per SM)  GPU work
      ("c", name, ts, dur, correlation)                                                  API call
      ("s", name, ts, dur)                                                          ProfilerStep#N
      ("d", ts, dur)                                                         DataLoader __next__"""
    if ev.get("ph") != "X":
        return None
    ts, dur = _num(ev.get("ts")), _num(ev.get("dur"), 0.0)
    if ts is None:
        return None
    dur = max(dur, 0.0)
    cat = str(ev.get("cat", "")).lower()
    name = ev.get("name")
    name = sys.intern(name) if isinstance(name, str) else str(name)
    a = ev.get("args") or {}
    if cat in GPU_CATS:
        dev = a.get("device", ev.get("pid"))
        try:
            dev = int(dev)
        except (TypeError, ValueError):
            pass
        return ("g", GPU_CATS[cat], name, ts, dur, dev, a.get("stream"), a.get("correlation"),
                _dims(a.get("grid")), _dims(a.get("block")), _num(a.get("blocks per SM")))
    if cat in API_CATS:
        return ("c", name, ts, dur, a.get("correlation"))
    if cat == "gpu_user_annotation":
        return None
    if name.startswith("ProfilerStep#"):
        return ("s", name, ts, dur)
    if name.startswith(DATALOADER):
        return ("d", ts, dur)
    return None


def _hook(d):
    return _compact(d) if "ph" in d else d


def read_trace(path):
    """(top-level metadata, kept events) from a trace file or '-' for stdin, plain or gzipped."""
    try:
        if path == "-":
            raw = sys.stdin.buffer.read()
        else:
            with open(path, "rb") as f:
                raw = f.read()
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        text = raw.decode("utf-8", errors="replace")
        del raw
        doc = json.loads(text, object_hook=_hook, strict=False)
    except (OSError, EOFError, ValueError) as e:
        raise TraceError(f"cannot read {path}: {e}")
    if isinstance(doc, list):
        return {}, [e for e in doc if e]
    if not isinstance(doc, dict) or not isinstance(doc.get("traceEvents"), list):
        raise TraceError(f"{path} has no traceEvents list; is it a torch.profiler trace?")
    return doc, [e for e in doc["traceEvents"] if e]


def _has_kernels(events):
    return any(e[0] == "g" and e[1] == "kernel" for e in events)


def load(path):
    """(label, metadata, events). A folder (vLLM writes one trace per worker, and can add a CPU-only
    frontend trace) reads its largest trace that holds kernel events."""
    if path == "-" or not os.path.isdir(path):
        meta, events = read_trace(path)
        return ("stdin" if path == "-" else os.path.basename(path)), meta, events
    found = [os.path.join(path, n) for n in os.listdir(path)
             if n.endswith((".json", ".json.gz")) and os.path.isfile(os.path.join(path, n))]
    if not found:
        raise TraceError(f"no .json or .json.gz trace in {path}")
    found.sort(key=lambda p: -os.path.getsize(p))
    first = None
    for p in found:
        meta, events = read_trace(p)
        if first is None:
            first = (p, meta, events)
        if _has_kernels(events):
            first = (p, meta, events)
            break
    p, meta, events = first
    label = os.path.basename(p)
    if len(found) > 1:
        label += f" (the largest trace with kernels of {len(found)} in {path})"
    return label, meta, events


class Act:
    """One piece of GPU work: a kernel, a copy or a fill."""
    __slots__ = ("kind", "name", "ts", "end", "dev", "stream", "corr", "grid", "block", "bps",
                 "launch", "via", "cls")

    def __init__(self, kind, name, ts, dur, dev, stream, corr, grid, block, bps):
        self.kind, self.name, self.ts, self.end = kind, name, ts, ts + dur
        self.dev, self.stream, self.corr = dev, stream, corr
        self.grid, self.block, self.bps = grid, block, bps
        self.launch, self.via, self.cls = None, None, None

    @property
    def dur(self):
        return self.end - self.ts


class Call:
    """One CUDA or HIP API call on the CPU."""
    __slots__ = ("name", "ts", "end", "corr")

    def __init__(self, name, ts, end, corr):
        self.name, self.ts, self.end, self.corr = name, ts, end, corr


def _num(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _dims(x):
    if isinstance(x, (list, tuple)) and x:
        try:
            return tuple(int(v) for v in x)
        except (TypeError, ValueError):
            return None
    return None


def blocks(grid):
    n = 1
    for v in grid:
        n *= v
    return n


def parse(events):
    """GPU work, API calls, step markers and DataLoader spans, each in its own list. Takes the
    tuples read_trace returns, or raw event dicts."""
    acts, calls, steps, seen, loads = [], [], [], set(), []
    for e in events:
        if isinstance(e, dict):
            e = _compact(e)
            if e is None:
                continue
        tag = e[0]
        if tag == "g":
            _, kind, name, ts, dur, dev, stream, corr, grid, block, bps = e
            acts.append(Act(kind, name, ts, dur, dev, stream, corr, grid, block, bps))
        elif tag == "c":
            _, name, ts, dur, corr = e
            calls.append(Call(name, ts, ts + dur, corr))
        elif tag == "s":
            _, name, ts, dur = e
            if name not in seen:
                seen.add(name)
                steps.append((name, ts, ts + dur))
        elif tag == "d":
            loads.append((e[1], e[1] + e[2]))
    steps.sort(key=lambda s: s[1])
    calls.sort(key=lambda c: c.ts)
    loads.sort()
    return acts, calls, steps, loads


def head(name):
    """A kernel name without 'void' and without its argument list (template arguments stay)."""
    n = re.sub(r"^void\s+", "", name.strip()).replace("(anonymous namespace)::", "")
    depth = 0
    for i, ch in enumerate(n):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif ch == "(" and depth == 0 and i > 0:
            return n[:i]
    return n


def classify(name):
    """matmul, attention, elementwise (eager, fusable), other memory-bound, compiled or NCCL."""
    h = head(name).lower()
    if COMMS.search(h):
        return "NCCL"
    if h.startswith("triton"):
        if ATTENTION.search(h):
            return "attention"
        return "matmul" if h.startswith("triton_tem") else "compiled"
    if not NOT_MATH.search(h):
        if ATTENTION.search(h):
            return "attention"
        if MATMUL.search(h):
            return "matmul"
    if ELEMENTWISE.search(h) and not LIBRARY.search(h):
        return "elementwise"
    return "other memory-bound"


def link_launches(acts, calls):
    """Each piece of GPU work gets the start and the name of the CPU call that launched it."""
    by_corr = {}
    for c in calls:
        if c.corr is None:
            continue
        old = by_corr.get(c.corr)
        if old is None or (not ANY_LAUNCH.search(old.name) and ANY_LAUNCH.search(c.name)):
            by_corr[c.corr] = c
    for a in acts:
        c = by_corr.get(a.corr)
        if c is not None:
            a.launch, a.via = c.ts, c.name


def sm_count(meta, dev, kernels):
    """SMs on the device: deviceProperties when the trace has it, else grid blocks / blocks per SM."""
    for p in meta.get("deviceProperties") or []:
        if isinstance(p, dict) and p.get("id") == dev and p.get("numSms"):
            return int(p["numSms"]), str(p.get("name") or "") or None, "deviceProperties"
    ratios = [blocks(k.grid) / k.bps for k in kernels if k.grid and k.bps and k.bps > 0]
    if ratios:
        return int(round(statistics.median(ratios))), None, "grid blocks / blocks per SM"
    return None, None, None


# ---------------------------------------------------------------- the timeline

def merge(acts):
    """Busy intervals: the union of all GPU work on the device over every stream. Each interval
    keeps the work that opened it (of the pieces starting first, the one launched first)."""
    out = []
    for a in sorted(acts, key=lambda a: (a.ts, a.launch if a.launch is not None else float("inf"))):
        if out and a.ts <= out[-1][1]:
            if a.end > out[-1][1]:
                out[-1][1] = a.end
        else:
            out.append([a.ts, a.end, a])
    return out


def find_gaps(merged, calls, loads=()):
    """Every idle stretch on the device, as (start, end, kind, next work, blocking call). Kinds:
      launch   host wait: the next kernel was launched after the GPU went idle, one call at a time
      replay   host wait: the next work came from a CUDA graph replayed after the GPU went idle
      blocked  host wait: the CPU was inside a blocking CUDA call (a sync, a copy to host, an
               allocation) when the GPU drained, so the launch that ended the gap came after it
      data     host wait: the CPU spent most of the gap inside a DataLoader __next__
      queued   other idle: the next work was already queued when the GPU went idle"""
    blocking = sorted((c for c in calls if BLOCKING.search(c.name)), key=lambda c: c.end)
    ends = [c.end for c in blocking]
    gaps = []
    if not merged:
        return gaps

    starts = [d[0] for d in loads]

    def host(nxt, g0=None, g1=None):
        if g0 is not None and loads:
            k = max(0, bisect.bisect_right(starts, g0) - 1)
            fed = 0.0
            while k < len(loads) and loads[k][0] < g1:
                fed += overlap(loads[k][0], loads[k][1], g0, g1)
                k += 1
            if fed > 0.5 * (g1 - g0):
                return "data"
        return "replay" if (nxt.via and GRAPH_LAUNCH.search(nxt.via)) else "launch"

    gaps.append((float("-inf"), merged[0][0], host(merged[0][2]), merged[0][2], None))
    for i in range(1, len(merged)):
        g0, g1, nxt = merged[i - 1][1], merged[i][0], merged[i][2]
        if nxt.launch is None or nxt.launch <= g0:
            gaps.append((g0, g1, "queued", nxt, None))
            continue
        blocker = None
        j = bisect.bisect_left(ends, g0)
        while j < len(blocking) and blocking[j].end <= g1:
            if blocking[j].ts < g0:
                blocker = blocking[j]
                break
            j += 1
        gaps.append((g0, g1, "blocked" if blocker else host(nxt, g0, g1), nxt, blocker))
    gaps.append((merged[-1][1], float("inf"), "launch", None, None))
    return gaps


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


class Window:
    __slots__ = ("name", "t0", "t1", "warm", "busy", "idle", "count", "kernels")

    def __init__(self, name, t0, t1):
        self.name, self.t0, self.t1 = name, t0, t1
        self.warm, self.kernels = {}, []
        self.busy = 0.0
        self.idle = {k: 0.0 for k in GAP_KINDS}
        self.count = {k: 0 for k in GAP_KINDS}

    @property
    def span(self):
        return self.t1 - self.t0


def build_windows(steps, merged, split_us):
    """One window per ProfilerStep#N (its CPU span); with no step markers, the GPU timeline split
    at idle stretches of split_us or more."""
    if steps:
        return [Window(n, t0, t1) for n, t0, t1 in steps if t1 > t0], "steps"
    wins, cur = [], None
    for s, e, _ in merged:
        if cur is not None and s - cur[1] < split_us:
            cur[1] = max(cur[1], e)
            continue
        if cur is not None:
            wins.append(cur)
        cur = [s, e]
    if cur is not None:
        wins.append(cur)
    return [Window(f"burst {i + 1}", s, e) for i, (s, e) in enumerate(wins) if e > s], "split"


def fill_windows(wins, source, acts, merged, gaps, calls):
    """Busy time, idle time by kind, warm-up calls and kernels, per window."""
    mstarts = [m[0] for m in merged]
    gstarts = [g[0] for g in gaps]
    cstarts = [c.ts for c in calls]
    for w in wins:
        i = max(0, bisect.bisect_right(mstarts, w.t0) - 1)
        while i < len(merged) and merged[i][0] < w.t1:
            w.busy += overlap(merged[i][0], merged[i][1], w.t0, w.t1)
            i += 1
        j = max(0, bisect.bisect_right(gstarts, w.t0) - 1)
        while j < len(gaps) and gaps[j][0] < w.t1:
            c = overlap(gaps[j][0], gaps[j][1], w.t0, w.t1)
            if c > 0:
                w.idle[gaps[j][2]] += c
                w.count[gaps[j][2]] += 1
            j += 1
        k = bisect.bisect_left(cstarts, w.t0)
        while k < len(calls) and calls[k].ts < w.t1:
            if WARMUP.search(calls[k].name):
                w.warm[calls[k].name] = w.warm.get(calls[k].name, 0) + 1
            k += 1
    # A kernel belongs to the step its launch call fell in. With no launch linked, or in a split
    # window (whose bounds are GPU times), its start decides.
    t0s = [w.t0 for w in wins]
    for a in acts:
        if a.kind != "kernel":
            continue
        t = a.launch if (source == "steps" and a.launch is not None) else a.ts
        x = bisect.bisect_right(t0s, t) - 1
        if x >= 0 and t < wins[x].t1:
            wins[x].kernels.append(a)


# ---------------------------------------------------------------- the verdict

def group(kernels):
    """Kernels grouped by (name, grid, block): time, launches, median duration, blocks per SM."""
    g = {}
    for k in kernels:
        row = g.setdefault((k.name, k.grid, k.block), {"time": 0.0, "durs": [], "bps": k.bps, "n": 0})
        row["time"] += k.dur
        row["durs"].append(k.dur)
        row["n"] += 1
    out = [{"name": name, "grid": grid, "block": block, "time_us": row["time"], "launches": row["n"],
            "median_us": statistics.median(row["durs"]), "blocks_per_sm": row["bps"]}
           for (name, grid, block), row in g.items()]
    out.sort(key=lambda r: -r["time_us"])
    return out


def _plural(n, word):
    return f"{n:,} {word}{'' if n == 1 else 's'}"


def fusion_applies(sh):
    """(c): eager elementwise kernels hold ELEMENTWISE_MIN of compute-kernel time, or at least
    ELEMENTWISE_OVER_MATH and more of it than matmul and attention together, at which point
    'the math is the work' is false."""
    ew = sh["compute"]["elementwise"]
    return ew >= ELEMENTWISE_MIN or (ew >= ELEMENTWISE_OVER_MATH and ew > sh["matmul_attention"])


def analyze(meta, events, device=None, split_ms=SPLIT_MS, label="trace"):
    """Every number the report prints, and the verdict, as one dict."""
    acts, calls, steps, loads = parse(events)
    r = {"tool": "kernel_trace_check", "version": __version__, "trace": label, "device": None}
    kern_all = [a for a in acts if a.kind == "kernel"]
    if not kern_all:
        r.update(verdict="INCONCLUSIVE", lever=None,
                 reason="no kernel events in this trace: it is CPU-only, or CUDA activity was off "
                        "(profile with activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])")
        return r
    ktime = {}
    for k in kern_all:
        ktime[k.dev] = ktime.get(k.dev, 0.0) + k.dur
    if device is None:
        dev = max(ktime, key=lambda d: ktime[d])
    elif device in ktime:
        dev = device
    else:
        r.update(verdict="INCONCLUSIVE", lever=None,
                 reason=f"no kernels on device {device}; devices with kernels: "
                        + ", ".join(str(d) for d in sorted(ktime, key=str)))
        return r

    acts = [a for a in acts if a.dev == dev]
    link_launches(acts, calls)
    kern_dev = [a for a in acts if a.kind == "kernel"]
    for k in kern_dev:
        k.cls = classify(k.name)
    sms, gpu, sms_src = sm_count(meta, dev, kern_dev)
    for k in kern_dev:
        if k.bps is None and k.grid and sms:
            k.bps = blocks(k.grid) / sms
    merged = merge(acts)
    gaps = find_gaps(merged, calls, loads)
    wins, source = build_windows(steps, merged, split_ms * 1000.0)
    fill_windows(wins, source, acts, merged, gaps, calls)
    unit = "step time" if source == "steps" else "window time"
    r.update(device=dev, devices_with_kernels=len(ktime), gpu=gpu, sms=sms, sms_source=sms_src,
             streams=len({a.stream for a in acts if a.stream is not None}), window_source=source,
             split_ms=split_ms if source == "split" else None, unit=unit)
    r["windows"] = [{"name": w.name, "start_us": w.t0, "end_us": w.t1, "span_us": w.span, "busy_us": w.busy,
                     "idle_us": dict(w.idle), "gaps": dict(w.count), "kernels": len(w.kernels),
                     "warm_up_calls": w.warm, "judged": not w.warm} for w in wins]

    judged = [w for w in wins if not w.warm]
    warm = [w for w in wins if w.warm]
    kernels = [k for w in judged for k in w.kernels]
    if len(kernels) < MIN_KERNELS:
        if warm:
            seen = {}
            for w in warm:
                for n, c in w.warm.items():
                    seen[n] = seen.get(n, 0) + c
            what = ", ".join(f"{n} x{c}" for n, c in sorted(seen.items(), key=lambda kv: -kv[1])[:3])
            held = sum(len(w.kernels) for w in warm)
            total = sum(len(w.kernels) for w in wins)
            r.update(verdict="INCONCLUSIVE", lever=None,
                     reason=f"a warm-up trace: {len(warm)} of {len(wins)} windows allocate device memory or "
                            f"set up the process ({what}) and hold {held:,} of the {_plural(total, 'kernel')}; "
                            "profile steady state with schedule(wait=1, warmup=1, active=3)")
        else:
            r.update(verdict="INCONCLUSIVE", lever=None,
                     reason=f"{_plural(len(kernels), 'kernel')} in the windows, too few to judge "
                            f"(the floor is {MIN_KERNELS})")
        return r

    span = sum(w.span for w in judged)
    busy = sum(w.busy for w in judged)
    idle = {k: sum(w.idle[k] for w in judged) for k in GAP_KINDS}
    count = {k: sum(w.count[k] for w in judged) for k in GAP_KINDS}
    kt = {c: 0.0 for c in CLASSES}
    for k in kernels:
        kt[k.cls] += k.dur
    ktotal = sum(kt.values())
    compute = [k for k in kernels if k.cls != "NCCL"]
    ctotal = ktotal - kt["NCCL"]
    has_grid = any(k.bps is not None for k in compute)
    small = [k for k in compute if k.bps is not None and k.bps < 1.0]
    small_t = sum(k.dur for k in small)
    mm = [k for k in compute if k.cls in ("matmul", "attention")]
    fp32 = sum(k.dur for k in mm if FP32_MATMUL.search(head(k.name).lower()))

    def share(x, d):
        return x / d if d else 0.0

    sh = {"busy": share(busy, span), "idle": {k: share(idle[k], span) for k in GAP_KINDS},
          "host_wait": share(idle["launch"] + idle["replay"] + idle["blocked"] + idle["data"], span),
          "unreached": share(idle["replay"] + idle["blocked"] + idle["data"] + idle["queued"], span),
          "nccl": share(kt["NCCL"], ktotal),
          "compute": {c: share(kt[c], ctotal) for c in CLASSES if c != "NCCL"},
          "small_grid": share(small_t, ctotal) if has_grid else None,
          "matmul_attention": share(kt["matmul"] + kt["attention"], ctotal),
          "fp32_of_matmul": share(fp32, kt["matmul"] + kt["attention"])}
    # (d) names whichever of these holds the most compute-kernel time
    groups = {"matmul and attention": sh["matmul_attention"],
              "other memory-bound": sh["compute"]["other memory-bound"], "compiled": sh["compute"]["compiled"]}
    sh["largest_group"] = max(groups, key=lambda g: groups[g])
    r.update(judged=[w.name for w in judged], warm_up=[w.name for w in warm], span_us=span, busy_us=busy,
             idle_us=idle, gaps=count,
             host_wait_gaps=count["launch"] + count["replay"] + count["blocked"] + count["data"],
             kernels=len(kernels), kernel_us=ktotal, compute_kernel_us=ctotal,
             median_kernel_us=statistics.median(k.dur for k in kernels), kernel_us_by_class=kt,
             small_grid_us=small_t, shares=sh)

    # each gap's length inside the judged windows (windows are sorted and do not overlap)
    j0s = [w.t0 for w in judged]
    in_judged = []
    for g in gaps:
        x, c = max(0, bisect.bisect_right(j0s, g[0]) - 1), 0.0
        while x < len(judged) and judged[x].t0 < g[1]:
            c += overlap(g[0], g[1], judged[x].t0, judged[x].t1)
            x += 1
        if c > 0:
            in_judged.append((c, g))
    in_judged.sort(key=lambda x: -x[0])
    r["largest_gaps"] = [{"kind": g[2], "idle_us": c, "start_us": g[0], "end_us": g[1],
                          "next": g[3].name if g[3] else None,
                          "next_launched_us": g[3].launch if g[3] else None,
                          "next_launched_by": g[3].via if g[3] else None,
                          "blocking_call": g[4].name if g[4] else None}
                         for c, g in in_judged[:20] if g[0] != float("-inf")]

    def rows(ks, denom):
        return [dict(x, share=share(x["time_us"], denom)) for x in group(ks)[:3]]

    lever, reason, top, then = None, None, [], []
    if sh["nccl"] > NCCL_MAX:
        verdict = "INCONCLUSIVE"
        reason = (f"NCCL kernels take {sh['nccl'] * 100:.1f}% of kernel time; a collective waits on the slowest "
                  "rank, and this tool reads one GPU's compute: profile one GPU alone, or read every rank")
        top = rows([k for k in kernels if k.cls == "NCCL"], ktotal)
    elif sh["idle"]["launch"] >= HOST_WAIT_MIN:
        verdict, lever = "LEVER", "cuda-graphs"
        waited = {}
        for c, g in in_judged:
            if g[2] != "launch" or g[3] is None or g[3].kind != "kernel":
                continue
            key = (g[3].name, g[3].grid, g[3].block)
            row = waited.setdefault(key, {"idle": 0.0, "gaps": 0, "acts": []})
            row["idle"] += c
            row["gaps"] += 1
            row["acts"].append(g[3])
        for key, row in sorted(waited.items(), key=lambda kv: -kv[1]["idle"])[:3]:
            x = group(row["acts"])[0]
            top.append(dict(x, idle_us=row["idle"], gaps=row["gaps"], share=share(row["idle"], span)))
    elif sh["unreached"] >= UNREACHED_MAX:
        verdict = "INCONCLUSIVE"
        part = max(("queued", "blocked", "data", "replay"), key=lambda k: idle[k])
        where = "step" if source == "steps" else "window"
        if part == "queued":
            reason = (f"the GPU sat idle {sh['idle']['queued'] * 100:.1f}% of {where} time with its next work "
                      "already queued: it waited on a dependency (another stream, another rank, a copy), not on "
                      "the CPU launching and not on the kernels")
        elif part == "blocked":
            reason = (f"the GPU sat idle {sh['idle']['blocked'] * 100:.1f}% of {where} time after the CPU blocked "
                      "in a CUDA call (a sync, a copy to host, an allocation): move those calls out of the step, "
                      "then read the kernels again")
        elif part == "data":
            reason = (f"the GPU sat idle {sh['idle']['data'] * 100:.1f}% of {where} time while the CPU waited on "
                      "the DataLoader: the input pipeline is the bottleneck, not the kernels (raise num_workers, "
                      "set pin_memory=True, prefetch the next batch)")
        else:
            reason = (f"the GPU sat idle {sh['idle']['replay'] * 100:.1f}% of {where} time waiting for the CPU "
                      "between CUDA graph replays: graphs are on already, so the CPU's own work per step "
                      "(scheduling, sampling, Python) is what the GPU waits on")
        for c, g in [x for x in in_judged if x[1][2] == part][:3]:
            cpu, best = None, 0.0
            for call in calls:
                if call.ts >= g[1]:
                    break
                o = overlap(call.ts, call.end, g[0], g[1])
                if o > best and not KERNEL_LAUNCH.search(call.name):
                    cpu, best = call, o
            nxt = g[3]
            top.append({"idle_us": c, "start_us": g[0], "share": share(c, span),
                        "next": nxt.name if nxt else None, "via": nxt.via if nxt else None,
                        "queued_us": (g[1] - nxt.launch) if (nxt and nxt.launch is not None) else None,
                        "cpu_call": cpu.name if cpu else None, "cpu_call_us": (cpu.end - cpu.ts) if cpu else None})
    elif sh["small_grid"] is not None and sh["small_grid"] >= SMALL_GRID_MIN:
        verdict, lever = "LEVER", "batch-or-shape"
        top = rows(small, ctotal)
    elif fusion_applies(sh):
        verdict, lever = "LEVER", "fusion"
        top = rows([k for k in compute if k.cls == "elementwise"], ctotal)
    else:
        verdict, lever = "LEVER", "leave-alone"
        pick = {"matmul and attention": ("matmul", "attention"), "other memory-bound": ("other memory-bound",),
                "compiled": ("compiled",)}[sh["largest_group"]]
        top = rows([k for k in compute if k.cls in pick], ctotal)

    if lever == "cuda-graphs":
        if sh["small_grid"] is not None and sh["small_grid"] >= SMALL_GRID_MIN:
            then.append("batch-or-shape")
        if fusion_applies(sh):
            then.append("fusion")
    elif lever == "batch-or-shape" and fusion_applies(sh):
        then.append("fusion")
    ceiling = None
    if lever == "cuda-graphs":      # closing every launch gap is the most a graph removes
        ceiling = idle["launch"]
    elif lever == "fusion":         # fusion cannot save more than those kernels' own run time
        ceiling = min(kt.get("elementwise", 0.0), span)
    r.update(verdict=verdict, lever=lever, reason=reason, top=top, then=then,
             ceiling_us=ceiling, ceiling_share=share(ceiling, span) if ceiling is not None else None)
    return r


# ---------------------------------------------------------------- the report

W = 13   # label column


def lab(s):
    return s.ljust(W)


def p1(x):
    return f"{x * 100:.1f}%"


def ms(us):
    return f"{us / 1000:,.1f} ms"


def dims(d):
    return "x".join(str(v) for v in d) if d else "?"


def short(name, width=64):
    """A kernel name cut to fit a terminal; the prefix stays searchable in a trace viewer."""
    n = head(name)
    for ns in ("at::native::", "at::detail::", "c10::", "std::"):
        n = n.replace(ns, "")
    n = re.sub(r"\s+", " ", n).strip()
    return n if len(n) <= width else n[:width - 3] + "..."


SHAPE_HEAD = f"{'median':>9}  {'grid':<11} {'block':<9} {'blk/SM':>7}"


def _bps(bps):
    if bps is None:
        return "?"
    return f"{bps:.2f}" if bps < 100 else f"{bps:,.0f}"


def _shape(t):
    return (f"{t['median_us']:>6,.0f} us  {dims(t['grid']):<11} {dims(t['block']):<9} "
            f"{_bps(t.get('blocks_per_sm')):>7}")


def _table(out, label, rows):
    out.append(lab(label) + f"{'share':>6}  {'launches':>8}  " + SHAPE_HEAD + "  kernel")
    for t in rows:
        out.append(lab("") + f"{p1(t['share']):>6}  {t['launches']:>8,}  " + _shape(t) + "  " + short(t["name"], 56))


def _wrap(label, text, width=100):
    lines = textwrap.wrap(text, width=width)
    return [lab(label) + lines[0]] + [lab("") + ln for ln in lines[1:]]


def _ceiling(r, condition):
    """The most the named lever can save: its share of step time, and per step when steps exist."""
    txt = f"{p1(r['ceiling_share'])} of {r['unit']} saved"
    if r["window_source"] == "steps" and r["judged"]:
        now = r["span_us"] / len(r["judged"]) / 1000
        low = (r["span_us"] - r["ceiling_us"]) / len(r["judged"]) / 1000
        txt += f": {now:,.1f} ms a step down to no less than {low:,.1f} ms"
    return f"{txt}, {condition}"


LEVER_NAME = {"cuda-graphs": "CUDA graphs", "batch-or-shape": "a bigger batch or a rounded shape",
              "fusion": "fusion", "leave-alone": "leave the kernels alone"}


def compare(a, b):
    """Two reports side by side: what a change bought, per step, and where the time moved."""
    out = ["kernel_trace_check · compare"]
    for name, r in (("before", a), ("after", b)):
        if r["device"] is None or not r["judged"]:
            return "\n".join(out + [""] + _wrap("INCONCLUSIVE", f"{r['trace']}: {r['reason']}"))
    per = lambda r: r["span_us"] / len(r["judged"]) / 1000
    each = "a step" if a["window_source"] == b["window_source"] == "steps" else "a window"
    for name, r in (("before", a), ("after", b)):
        verdict = f"LEVER {LEVER_NAME[r['lever']]}" if r["verdict"] == "LEVER" else "INCONCLUSIVE"
        out.append(lab(name) + f"{r['trace']}: {len(r['judged'])} windows on {r.get('gpu') or 'GPU ' + str(r['device'])}, "
                   f"{per(r):,.1f} ms {each}, {verdict}")
    if (a.get("gpu") or a["device"]) != (b.get("gpu") or b["device"]):
        out.append(lab("caveat") + "different GPUs: the difference is the hardware as much as any change")
    s0, s1 = per(a), per(b)
    out.append(lab("step time") + f"{s0:,.1f} ms to {s1:,.1f} ms {each} ({(s1 - s0) / s0 * 100:+.1f}%)")
    sa, sb = a["shares"], b["shares"]
    out.append(lab("busy") + f"{p1(sa['busy'])} to {p1(sb['busy'])} of {a['unit']}")
    out.append(lab("host wait") + f"{p1(sa['idle']['launch'])} to {p1(sb['idle']['launch'])} before eager launches")
    out.append(lab("kernels") + f"{a['kernels']:,} to {b['kernels']:,}, median {a['median_kernel_us']:,.0f} to "
               f"{b['median_kernel_us']:,.0f} us")
    moved = [(c, sa["compute"].get(c, 0.0), sb["compute"].get(c, 0.0)) for c in CLASSES if c != "NCCL"]
    moved = [m for m in moved if max(m[1], m[2]) >= 0.005]
    moved.sort(key=lambda m: -abs(m[2] - m[1]))
    out.append(lab("compute") + ", ".join(f"{c} {p1(x)} to {p1(y)}" for c, x, y in moved))
    out.append(lab("NCCL") + f"{p1(sa['nccl'])} to {p1(sb['nccl'])} of kernel time")
    return "\n".join(out)


def render(r):
    out = [f"kernel_trace_check · {r['trace']}"]
    if r["device"] is None:
        return "\n".join(out + [""] + _wrap("INCONCLUSIVE", r["reason"]))
    d = f"{r['device']}"
    if r.get("gpu"):
        d += f", {r['gpu']}"
    d += f", {r['sms']} SMs (from {r['sms_source']})" if r.get("sms") else ", SM count not recorded"
    if r["devices_with_kernels"] > 1:
        d += f"; the busiest of {r['devices_with_kernels']} devices with kernels"
    out.append(lab("device") + d)
    wins = r["windows"]
    if not wins:    # every step marker had zero length, so no window survived to judge
        return "\n".join(out + [""] + _wrap("INCONCLUSIVE", r["reason"]))
    judged = [w for w in wins if w["judged"]]
    shown = judged or wins
    steps = r["window_source"] == "steps"
    unit = r["unit"]
    if steps:
        first, last = shown[0]["name"], shown[-1]["name"]
        rng = first if len(shown) == 1 else f"{first} to {last.replace('ProfilerStep', '')}"
        count = (f"{len(judged)} of {len(wins)} steps" if len(judged) != len(wins)
                 else _plural(len(wins), "step"))
        w_line = f"{count} ({rng})"
    else:
        count = (f"{len(judged)} of {len(wins)} bursts" if len(judged) != len(wins)
                 else _plural(len(wins), "burst"))
        w_line = f"{count} of GPU work, split at idle stretches of {r['split_ms']:g} ms or more"
    if r.get("span_us") is not None:
        w_line += f", {ms(r['span_us'])} of {unit}" + (" judged" if len(judged) != len(wins) else "")
    out.append(lab("windows") + w_line)
    warm = [w for w in wins if not w["judged"]]
    if warm:
        seen = {}
        for w in warm:
            for n, c in w["warm_up_calls"].items():
                seen[n] = seen.get(n, 0) + c
        what = ", ".join(f"{n} x{c}" for n, c in sorted(seen.items(), key=lambda kv: -kv[1])[:3])
        names = ", ".join(w["name"] for w in warm[:3]) + (", ..." if len(warm) > 3 else "")
        out.append(lab("caveat") + f"{_plural(len(warm), 'window')} ({names}) set up the process ({what}) "
                   f"and {'is' if len(warm) == 1 else 'are'} left out")
    if r.get("span_us") is None:
        return "\n".join(out + [""] + _wrap("INCONCLUSIVE", r["reason"]))
    if not steps:
        out.append(lab("caveat") + "no ProfilerStep markers, so the windows are bursts, not steps; call "
                   "prof.step() once per iteration")
    elif len(judged) == 1:
        out.append(lab("caveat") + "one step only, and one step can hold a one-off stall; profile three with "
                   "schedule(wait=1, warmup=1, active=3)")
    sh = r["shares"]
    si = sh["idle"]
    where = "its one stream" if r["streams"] == 1 else f"at least one of {r['streams']} streams"
    out.append(lab("busy") + f"{p1(sh['busy'])} of {unit}: a kernel, copy or fill running on {where}")
    out.append(lab("idle") + f"{p1(1 - sh['busy'])}: host wait {p1(sh['host_wait'])} over "
               f"{r['host_wait_gaps']:,} gaps, other {p1(si['queued'])} with the next work already queued")
    parts = [f"{p1(si[k])} {t}" for k, t in (("launch", "before eager launches"), ("replay", "before graph replays"),
                                             ("blocked", "after the CPU blocked in a CUDA call"),
                                             ("data", "while the CPU waited on the DataLoader"))
             if r["idle_us"][k] > 0]
    if parts:
        out.append(lab("host wait") + ", ".join(parts))
    where_k = "steps" if steps else "bursts"
    out.append(lab("kernels") + f"{r['kernels']:,} in the {where_k}, median {r['median_kernel_us']:,.0f} us, "
               f"{ms(r['kernel_us'])} in all: NCCL {p1(sh['nccl'])}, compute {p1(1 - sh['nccl'])}")
    comp = sorted(((c, v) for c, v in sh["compute"].items() if v > 0), key=lambda cv: -cv[1])
    out.append(lab("compute") + f"{ms(r['compute_kernel_us'])}: " + ", ".join(f"{c} {p1(v)}" for c, v in comp))
    if sh["small_grid"] is None:
        out.append(lab("grids") + "not recorded in this trace (no grid or blocks per SM args)")
    else:
        out.append(lab("grids") + f"{p1(sh['small_grid'])} of compute-kernel time on grids under 1 block per SM")
    hosts = [g for g in r.get("largest_gaps", []) if g["kind"] != "queued"]
    if hosts:
        g = hosts[0]
        out.append(lab("largest gap") + f"{g['idle_us'] / 1000:,.2f} ms of host wait from ts {g['start_us']:.0f}, "
                   f"before {short(g['next'], 44) if g['next'] else 'the window ended'}")
    out.append("")
    v, lever = r["verdict"], r["lever"]
    if v == "INCONCLUSIVE":
        out += _wrap("INCONCLUSIVE", r["reason"])
        if r["top"] and "cpu_call" in r["top"][0]:
            out.append(lab("longest") + f"{'idle':>8}  {'from ts':<17} {'next work':<18} {'queued for':>11}  CPU was in")
            for t in r["top"]:
                q = f"{t['queued_us'] / 1000:,.1f} ms" if t["queued_us"] is not None else "not in trace"
                cpu = f"{t['cpu_call']} ({t['cpu_call_us'] / 1000:,.1f} ms)" if t["cpu_call"] else "no CUDA call"
                out.append(lab("") + f"{t['idle_us'] / 1000:>5,.1f} ms  {t['start_us']:<17.0f} "
                           f"{short(t['next'] or 'nothing', 18):<18} {q:>11}  {cpu}")
        elif r["top"]:
            _table(out, "top 3", r["top"])
        return "\n".join(out)
    if lever == "cuda-graphs":
        out.append(lab("LEVER") + f"CUDA graphs: the GPU sat idle waiting for the CPU to launch the next kernel "
                   f"for {p1(si['launch'])} of {unit}")
        out.append(lab("") + f"{r['kernels']:,} kernels of median {r['median_kernel_us']:,.0f} us went out one launch "
                   "at a time, slower than the GPU ran them")
        out.append(lab("") + "torch.compile(mode=\"reduce-overhead\") replays them as one CUDA graph; vLLM does "
                   "unless it runs with --enforce-eager")
        if r["idle_us"]["blocked"] > 0:
            out.append(lab("") + f"the {p1(si['blocked'])} after a blocking CUDA call stays: a graph cannot span a sync")
        out.append(lab("at most") + _ceiling(r, "if every launch gap closes"))
        out.append(lab("waited on") + f"{'idle':>6} {'gaps':>8}  " + SHAPE_HEAD + "  kernel")
        for t in r["top"]:
            out.append(lab("") + f"{p1(t['share']):>6} {t['gaps']:>8,}  " + _shape(t) + "  " + short(t["name"], 56))
    else:
        if lever == "batch-or-shape":
            out.append(lab("LEVER") + f"a bigger batch or a rounded shape: {p1(sh['small_grid'])} of compute-kernel "
                       "time ran on grids under 1 block per SM")
            out.append(lab("") + f"a grid of fewer blocks than the {r['sms']} SMs leaves the rest idle for the whole "
                       "kernel")
            out.append(lab("") + "raise the batch (--max-num-seqs in vLLM), or pad the dimension that sets the grid "
                       "to a multiple of 64 or 128")
        elif lever == "fusion":
            ew = sh["compute"]["elementwise"]
            why = ("" if ew >= ELEMENTWISE_MIN else
                   f", more than matmul and attention together ({p1(sh['matmul_attention'])})")
            out.append(lab("LEVER") + f"fusion: eager elementwise kernels take {p1(ew)} of compute-kernel time{why}")
            out.append(lab("") + "each reads its input from memory and writes its result back; fused, a chain keeps "
                       "values on chip")
            out.append(lab("") + "torch.compile fuses chains of elementwise, reduction and norm kernels into single "
                       "Triton kernels")
            out.append(lab("at most") + _ceiling(r, "if fusion removed those kernels' time entirely"))
        else:
            g = sh["largest_group"]
            if g == "matmul and attention":
                out.append(lab("LEVER") + f"leave the kernels alone: matmul and attention take "
                           f"{p1(sh['matmul_attention'])} of compute-kernel time")
                if sh["fp32_of_matmul"] >= 0.5:
                    out.append(lab("") + f"{p1(sh['fp32_of_matmul'])} of that math runs FP32 kernels (sgemm, scudnn) "
                               "with no tensor cores:")
                    out.append(lab("") + "bf16 or fp16 autocast is the gain, not a kernel change")
                else:
                    out.append(lab("") + "the math is the work: the next gain is fewer FLOPs or fewer bytes (a "
                               "smaller model, lower precision)")
            elif g == "compiled":
                out.append(lab("LEVER") + f"leave the kernels alone: compiled kernels (triton_*) take "
                           f"{p1(sh['compute']['compiled'])} of compute-kernel time")
                out.append(lab("") + "torch.compile fused these already; its mode (max-autotune) or fewer graph "
                           "breaks is the next gain")
            else:
                out.append(lab("LEVER") + f"leave the kernels alone: other memory-bound kernels take "
                           f"{p1(sh['compute']['other memory-bound'])} of compute-kernel time")
                out.append(lab("") + "library kernels (embedding lookups, sorts, fused optimizers, custom ops) that "
                           "torch.compile does not regenerate")
        _table(out, "top 3", r["top"])
    for nxt in r["then"]:
        if nxt == "fusion":
            out.append(lab("then") + f"fusion: eager elementwise kernels take {p1(sh['compute']['elementwise'])} of "
                       "compute-kernel time")
        else:
            out.append(lab("then") + f"a bigger batch or a rounded shape: {p1(sh['small_grid'])} of compute-kernel "
                       "time ran on grids under 1 block per SM")
    return "\n".join(out)


def to_json(r):
    def clean(x):
        if isinstance(x, float):
            return None if x in (float("inf"), float("-inf")) else round(x, 6)
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        return x
    return json.dumps(clean(r), indent=2)


class _Parser(argparse.ArgumentParser):
    """Usage errors exit 1 like unreadable input, so exit code 2 always means INCONCLUSIVE."""

    def error(self, message):
        self.print_usage(sys.stderr)
        print(f"kernel_trace_check: {message}", file=sys.stderr)
        sys.exit(1)


def main(argv=None):
    ap = _Parser(description="Read a torch.profiler trace and name the one lever its GPU time points at.")
    ap.add_argument("trace", nargs="?", default="-",
                    help="a trace .json or .json.gz, a folder of them, or - for stdin (the default)")
    ap.add_argument("--device", type=int, help="GPU index to read (default: the one with the most kernel time)")
    ap.add_argument("--split-ms", type=float, default=SPLIT_MS,
                    help=f"with no ProfilerStep markers, split at idle stretches this long (default {SPLIT_MS:g})")
    ap.add_argument("--json", action="store_true", help="print every number as JSON instead of the report")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                    help="read two traces of the same job and print what changed between them")
    ap.add_argument("--version", action="version", version=f"kernel_trace_check {__version__}")
    a = ap.parse_args(argv)
    if a.compare:
        reports = []
        for path in a.compare:
            try:
                label, meta, events = load(path)
            except TraceError as e:
                print(f"kernel_trace_check: {e}", file=sys.stderr)
                return 1
            reports.append(analyze(meta, events, device=a.device, split_ms=a.split_ms, label=label))
        print(compare(*reports))
        return 0 if all(r["verdict"] == "LEVER" for r in reports) else 2
    if a.trace == "-" and sys.stdin.isatty():
        ap.error("pass a trace file or folder, or pipe one in")
    try:
        label, meta, events = load(a.trace)
    except TraceError as e:
        print(f"kernel_trace_check: {e}", file=sys.stderr)
        return 1
    r = analyze(meta, events, device=a.device, split_ms=a.split_ms, label=label)
    print(to_json(r) if a.json else render(r))
    return 0 if r["verdict"] == "LEVER" else 2


if __name__ == "__main__":
    sys.exit(main())
