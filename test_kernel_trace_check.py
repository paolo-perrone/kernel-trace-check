#!/usr/bin/env python3
"""Replays eight recorded torch.profiler traces and checks kernel_trace_check's arithmetic by hand.

The recorded traces are real ones from Meta's Holistic Trace Analysis test data, trimmed to the
events the tool reads (fixtures/README.md gives the source of each file and what was trimmed).
The unit checks build tiny traces by hand, where every gap and every share is known in advance.
Run: python3 test_kernel_trace_check.py
"""
import contextlib
import gzip
import io
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FX = os.path.join(HERE, "fixtures")
sys.path.insert(0, HERE)    # so the tests run from any folder, and under python -I

import kernel_trace_check as ktc  # noqa: E402


def report(fixture):
    label, meta, events = ktc.load(os.path.join(FX, fixture))
    r = ktc.analyze(meta, events, label=label)
    return r, ktc.render(r)


# (fixture, verdict, lever, strings the report must contain)
RECORDED = [
    ("h100-train.json.gz", "LEVER", "cuda-graphs", [
        "windows      6 steps (ProfilerStep#59 to #64), 888.2 ms of step time",
        "busy         51.5% of step time",
        "host wait    43.2% before eager launches, 0.4% after the CPU blocked in a CUDA call",
        "NCCL 23.9%, compute 76.1%",
        "LEVER        CUDA graphs: the GPU sat idle waiting for the CPU to launch the next kernel for 43.2% "
        "of step time",
        "then         a bigger batch or a rounded shape: 30.0% of compute-kernel time",
    ]),
    ("v100-inference.json.gz", "LEVER", "cuda-graphs", [
        "80 SMs (from grid blocks / blocks per SM)",
        "caveat       one step only",
        "for 77.1% of step time",
        "the 5.1% after a blocking CUDA call stays",
    ]),
    ("v100-cnn-train.json.gz", "LEVER", "leave-alone", [
        "busy         97.4% of step time",
        "LEVER        leave the kernels alone: matmul and attention take 63.6% of compute-kernel time",
        "77.4% of that math runs FP32 kernels (sgemm, scudnn)",
    ]),
    ("a100-embedding-train.json.gz", "LEVER", "fusion", [
        "compute      283.3 ms: other memory-bound 35.6%, elementwise 35.3%, matmul 29.2%",
        "LEVER        fusion: eager elementwise kernels take 35.3% of compute-kernel time, more than matmul and "
        "attention together (29.2%)",
    ]),
    ("a100-graphs-train.json.gz", "INCONCLUSIVE", None, [
        "INCONCLUSIVE the GPU sat idle 63.1% of step time with its next work already queued",
        "cudaMemcpyAsync (145.6 ms)",
    ]),
    ("a100-nccl-train.json.gz", "INCONCLUSIVE", None, [
        "INCONCLUSIVE NCCL kernels take 65.3% of kernel time",
        "ncclKernel_SendRecv_RING_SIMPLE_Sum_int8_t",
    ]),
    ("a100-alexnet-first-pass.json.gz", "INCONCLUSIVE", None, [
        "INCONCLUSIVE a warm-up trace: 4 of 6 windows allocate device memory",
        "cudaMalloc x14",
    ]),
    ("cpu-only.json.gz", "INCONCLUSIVE", None, [
        "INCONCLUSIVE no kernel events in this trace",
    ]),
]


# ---------------------------------------------------------------- hand-written traces

SMS = 100


def kernel(name, ts, dur, corr, stream=7, grid=(1000, 1, 1), bps=None, device=0):
    args = {"device": device, "stream": stream, "correlation": corr, "grid": list(grid), "block": [128, 1, 1]}
    if bps is not None:
        args["blocks per SM"] = bps
    return {"ph": "X", "cat": "kernel", "name": name, "pid": device, "tid": stream, "ts": ts, "dur": dur,
            "args": args}


def call(name, ts, dur, corr=None):
    return {"ph": "X", "cat": "cuda_runtime", "name": name, "pid": 1, "tid": 1, "ts": ts, "dur": dur,
            "args": {} if corr is None else {"correlation": corr}}


def step(n, ts, dur):
    return {"ph": "X", "cat": "user_annotation", "name": f"ProfilerStep#{n}", "pid": 1, "tid": 1, "ts": ts,
            "dur": dur}


def run(events, props=True):
    meta = {"deviceProperties": [{"id": 0, "name": "Test GPU", "numSms": SMS}]} if props else {}
    return ktc.analyze(meta, events, label="hand-written")


GEMM = "void cutlass::Kernel<cutlass_80_tensorop_bf16_s16816gemm_bf16_128x128_32x3_tn_align8>(Params)"
SGEMM = "ampere_sgemm_128x64_tn"
ADD = "void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>>(int, Array)"
SORT = "void at_cuda_detail::cub::DeviceRadixSortOnesweepKernel<Policy800, float>(int*)"
NCCL = "ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<4096ul>)"


def back_to_back(specs, start=0.0):
    """One step holding kernels that run back to back, every launch queued at the step start, so the
    only idle time is the 10 us before the first kernel. specs: (name, dur, blocks per SM)."""
    events, t = [], start + 10.0
    for i, (name, dur, bps) in enumerate(specs):
        events.append(call("cudaLaunchKernel", start + 1.0 + i * 0.01, 0.005, corr=i + 1))
        events.append(kernel(name, t, dur, corr=i + 1, grid=(int(bps * SMS) or 1, 1, 1), bps=bps))
        t += dur
    events.append(step(1, start, t - start))
    return events


def spaced(n, gap_kind):
    """n kernels of 2 us, the first at 10 us, with 10 us of idle before each later one. gap_kind
    decides why the GPU idles: the launch came late, it was queued early, it came from a graph
    replay, or the CPU was inside cudaStreamSynchronize when the GPU drained."""
    events, t = [], 10.0
    for i in range(n):
        late = t - 5.0 if i else 1.0
        if gap_kind == "queued":
            events.append(call("cudaLaunchKernel", 0.1 + i * 0.001, 0.0005, corr=i + 1))
        elif gap_kind == "replay":
            events.append(call("cudaGraphLaunch", late, 1.0, corr=i + 1))
        else:
            events.append(call("cudaLaunchKernel", late, 1.0, corr=i + 1))
        if gap_kind == "blocked" and i:
            events.append(call("cudaStreamSynchronize", t - 11.0, 4.0))
        events.append(kernel(GEMM, t, 2.0, corr=i + 1, bps=10.0))
        t += 12.0
    events.append(step(1, 0.0, t - 10.0))
    return events


def unit_checks():
    """Returns (checks run, failures)."""
    fails, ran = [], [0]

    def check(cond, what):
        ran[0] += 1
        if not cond:
            fails.append(what)

    # 1. host wait: the next kernel's launch starts after the GPU went idle (HTA's test)
    ev = [step(1, 0, 1000), call("cudaLaunchKernel", 90, 5, 1), kernel(GEMM, 100, 200, 1),
          call("cudaLaunchKernel", 350, 5, 2), kernel(GEMM, 400, 200, 2)]
    w = run(ev)["windows"][0]
    check(w["busy_us"] == 400 and w["idle_us"]["launch"] == 600 and w["idle_us"]["queued"] == 0,
          f"host wait arithmetic: busy 400, launch gaps 100 + 100 + 400 = 600, got {w}")

    # 2. other idle: the next kernel was launched before the GPU went idle, so it was queued
    ev = [step(1, 0, 600), call("cudaLaunchKernel", 90, 5, 1), kernel(GEMM, 100, 200, 1),
          call("cudaLaunchKernel", 250, 5, 2), kernel(GEMM, 400, 200, 2)]
    w = run(ev)["windows"][0]
    check(w["idle_us"]["queued"] == 100 and w["idle_us"]["launch"] == 100,
          f"queued gap 300 to 400, leading gap 0 to 100: got {w['idle_us']}")

    # 3. a kernel on a second stream covers the gap, so the GPU never idled
    ev = [step(1, 100, 500), call("cudaLaunchKernel", 90, 5, 1), kernel(GEMM, 100, 200, 1),
          call("cudaLaunchKernel", 240, 5, 3), kernel(ADD, 250, 200, 3, stream=9),
          call("cudaLaunchKernel", 350, 5, 2), kernel(GEMM, 400, 200, 2)]
    w = run(ev)["windows"][0]
    check(w["busy_us"] == 500 and sum(w["idle_us"].values()) == 0, f"union over streams: got {w}")

    # 4. the CPU sat in cudaStreamSynchronize when the GPU drained: blocked, not launch overhead
    ev = [step(1, 0, 600), call("cudaLaunchKernel", 90, 5, 1), kernel(GEMM, 100, 200, 1),
          call("cudaStreamSynchronize", 200, 110), call("cudaLaunchKernel", 350, 5, 2),
          kernel(GEMM, 400, 200, 2)]
    w = run(ev)["windows"][0]
    check(w["idle_us"]["blocked"] == 100, f"blocked gap 300 to 400: got {w['idle_us']}")

    # 5. graph replay: two kernels share the cudaGraphLaunch correlation; the gap before them is a replay
    ev = [step(1, 0, 700), call("cudaLaunchKernel", 90, 5, 1), kernel(GEMM, 100, 200, 1),
          call("cudaGraphLaunch", 350, 5, 7), kernel(GEMM, 400, 100, 7), kernel(ADD, 500, 100, 7),
          call("cudaLaunchKernel", 360, 5, 8), kernel(ADD, 650, 50, 8)]
    w = run(ev)["windows"][0]
    check(w["idle_us"]["replay"] == 100 and w["idle_us"]["queued"] == 50,
          f"replay gap 300 to 400, queued gap 600 to 650: got {w['idle_us']}")

    # 6. a gap that straddles two steps is split between them at the step boundary
    ev = [step(1, 0, 500), step(2, 500, 500), call("cudaLaunchKernel", 1, 1, 1), kernel(GEMM, 10, 440, 1),
          call("cudaLaunchKernel", 520, 1, 2), kernel(GEMM, 550, 450, 2)]
    ws = run(ev)["windows"]
    check(ws[0]["idle_us"]["launch"] == 60 and ws[1]["idle_us"]["launch"] == 50,
          f"gap 450 to 550 splits 50 and 50 (plus the 10 us lead): got {[w['idle_us'] for w in ws]}")

    # 7. a kernel belongs to the step its launch fell in, even when it runs in the next one
    ev = [step(1, 0, 100), step(2, 100, 100), call("cudaLaunchKernel", 50, 1, 1), kernel(GEMM, 150, 20, 1)]
    ws = run(ev)["windows"]
    check(ws[0]["kernels"] == 1 and ws[1]["kernels"] == 0, f"launch decides the step: got {ws}")

    # 8. a step with cudaMalloc inside is warm-up and left out; the verdict comes from the others
    ev = back_to_back([(GEMM, 100, 10.0)] * 60)
    ev += [call("cudaMalloc", 5, 3000)]
    later = back_to_back([(GEMM, 100, 10.0)] * 60, start=100000.0)
    for e in later:
        if e.get("name", "").startswith("ProfilerStep#"):
            e["name"] = "ProfilerStep#2"
        if "correlation" in e.get("args", {}):
            e["args"]["correlation"] += 1000
    r = run(ev + later)
    check(r["warm_up"] == ["ProfilerStep#1"] and r["judged"] == ["ProfilerStep#2"] and r["verdict"] == "LEVER",
          f"warm-up step left out: got {r.get('warm_up')}, {r.get('judged')}, {r['verdict']}")
    r = run(ev)
    check(r["verdict"] == "INCONCLUSIVE" and "warm-up" in r["reason"], f"all warm-up: got {r['reason']}")

    # 9. no ProfilerStep markers: the timeline splits at idle stretches of 50 ms or more
    ev = [call("cudaLaunchKernel", 0, 1, 1), kernel(GEMM, 5, 10, 1), call("cudaLaunchKernel", 20, 1, 2),
          kernel(GEMM, 25, 10, 2), call("cudaLaunchKernel", 100000, 1, 3), kernel(GEMM, 100005, 10, 3)]
    r = run(ev)
    check(r["window_source"] == "split" and len(r["windows"]) == 2, f"two bursts: got {r['windows']}")

    # 10. no deviceProperties: SMs = grid blocks / blocks per SM (64 blocks at 0.8 per SM on 80 SMs)
    ev = [step(1, 0, 100), call("cudaLaunchKernel", 1, 1, 1), kernel(GEMM, 5, 10, 1, grid=(64, 1, 1), bps=0.8)]
    r = run(ev, props=False)
    check(r["sms"] == 80 and r["sms_source"] == "grid blocks / blocks per SM", f"derived SMs: got {r['sms']}")

    # 11. kernel classes, on names from real traces
    cases = [
        ("void fbgemm_gpu::permute_2D_data_kernel<false, int, int>(int, int, int const*)", "other memory-bound"),
        ("void split_embedding_codegen_forward_unweighted_kernel<float, float, float, long, 2ul>"
         "(at::GenericPackedTensorAccessor<float, 1ul>, fbgemm_gpu::FixedDivisor)", "other memory-bound"),
        ("void splitKreduce_kernel<__half, __half, float, __half, true, false>(cublasSplitKParams<float>)",
         "matmul"),
        (ADD, "elementwise"),
        ("void at::native::(anonymous namespace)::layer_norm_grad_input_kernel<float, float>(float const*)",
         "elementwise"),
        ("void cudnn::bn_bw_1C11_kernel_new<float, float, float2, 128, true, 1>(float)", "elementwise"),
        ("void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int)", "matmul"),
        ("triton_poi_fused_add_cos_sin_0", "compiled"),
        ("triton_tem_fused_mm_0", "matmul"),
        (NCCL, "NCCL"),
        ("void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4>>(Flash_fwd_params)", "attention"),
        ("void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16>(__nv_bfloat16 const*)", "other memory-bound"),
        ("nvjet_tst_192x192_64x3_2x1_v_bz_coopA_TNT", "matmul"),
        ("sm90_xmma_gemm_f32f32_tf32f32_f32_nn_n_tilesize128x128x32_cgasize1x1x1", "matmul"),
        (SORT, "other memory-bound"),
        ("void multi_tensor_apply_kernel<TensorListMetadata<4>, AdamFunctor<float>>(int)", "other memory-bound"),
    ]
    for name, want in cases:
        got = ktc.classify(name)
        check(got == want, f"classify {name[:50]}: {got}, want {want}")

    # 12. each lever, on a step built to trigger it
    r = run(spaced(60, "launch"))
    check(r["lever"] == "cuda-graphs", f"launch gaps -> CUDA graphs, got {r['verdict']} {r['lever']}")
    r = run(spaced(60, "queued"))
    check(r["verdict"] == "INCONCLUSIVE" and "already queued" in r["reason"], f"queued: {r.get('reason')}")
    r = run(spaced(60, "blocked"))
    check(r["verdict"] == "INCONCLUSIVE" and "blocked" in r["reason"], f"blocked: {r.get('reason')}")
    r = run(spaced(60, "replay"))
    check(r["verdict"] == "INCONCLUSIVE" and "graph replays" in r["reason"], f"replay: {r.get('reason')}")
    ev = spaced(60, "launch") + [{"ph": "X", "cat": "user_annotation", "pid": 1, "tid": 1, "ts": 0.0, "dur": 800.0,
                                   "name": "enumerate(DataLoader)#_MultiProcessingDataLoaderIter.__next__"}]
    r = run(ev)
    check(r["verdict"] == "INCONCLUSIVE" and "DataLoader" in r["reason"] and r["gaps"]["data"] > 0,
          f"gaps spent inside DataLoader __next__ are not launch overhead: {r.get('reason')}")
    r = run(back_to_back([(GEMM, 100, 0.5)] * 60))
    check(r["lever"] == "batch-or-shape" and abs(r["shares"]["small_grid"] - 1.0) < 1e-9,
          f"half-filled grids -> batch-or-shape, got {r['lever']}")
    r = run(back_to_back([(ADD, 100, 10.0)] * 40 + [(GEMM, 100, 10.0)] * 20))
    check(r["lever"] == "fusion", f"two thirds elementwise -> fusion, got {r['lever']}")
    r = run(back_to_back([(ADD, 30, 10.0)] * 20 + [(GEMM, 25, 10.0)] * 20 + [(SORT, 45, 10.0)] * 20))
    check(r["lever"] == "fusion", f"30% elementwise over 25% matmul -> fusion, got {r['lever']}")
    r = run(back_to_back([(ADD, 20, 10.0)] * 20 + [(GEMM, 15, 10.0)] * 20 + [(SORT, 65, 10.0)] * 20))
    check(r["lever"] == "leave-alone" and r["shares"]["largest_group"] == "other memory-bound",
          f"library kernels hold the time -> leave alone, got {r['lever']} {r['shares']['largest_group']}")
    r = run(back_to_back([(GEMM, 80, 10.0)] * 30 + [(ADD, 20, 10.0)] * 30))
    check(r["lever"] == "leave-alone" and r["shares"]["fp32_of_matmul"] == 0,
          f"math holds the time -> leave alone, got {r['lever']}")
    r = run(back_to_back([(SGEMM, 80, 10.0)] * 30 + [(ADD, 20, 10.0)] * 30))
    check("FP32 kernels" in ktc.render(r), "FP32 math gets the autocast line")
    r = run(back_to_back([(NCCL, 60, 10.0)] * 30 + [(GEMM, 40, 10.0)] * 30))
    check(r["verdict"] == "INCONCLUSIVE" and "NCCL" in r["reason"], f"NCCL 60% -> refuse, got {r['verdict']}")
    r = run(back_to_back([(GEMM, 100, 10.0)] * 10))
    check(r["verdict"] == "INCONCLUSIVE" and "too few" in r["reason"], f"10 kernels -> refuse: {r.get('reason')}")
    r = run([step(1, 0, 100), call("cudaMalloc", 1, 5)])
    check(r["verdict"] == "INCONCLUSIVE" and "no kernel events" in r["reason"], "no kernels -> refuse")

    # 13. exit codes, file forms and folders through main()
    tmp = tempfile.mkdtemp()
    lever_doc = {"deviceProperties": [{"id": 0, "numSms": SMS}], "traceEvents": back_to_back([(GEMM, 100, 10.0)] * 60)}
    plain = os.path.join(tmp, "a.json")
    with open(plain, "w") as f:
        json.dump(lever_doc, f)
    zipped = os.path.join(tmp, "b.json.gz")
    with gzip.open(zipped, "wt") as f:
        json.dump(lever_doc["traceEvents"], f)        # a bare event list is a valid Chrome trace too
    bad = os.path.join(tmp, "c.json")
    with open(bad, "w") as f:
        f.write("{not json")
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        codes = [ktc.main([plain]), ktc.main([zipped]), ktc.main([bad]), ktc.main([os.path.join(tmp, "none.json")]),
                 ktc.main([os.path.join(FX, "cpu-only.json.gz")]), ktc.main([plain, "--json"])]
    check(codes == [0, 0, 1, 1, 2, 0], f"exit codes 0 lever, 1 unreadable, 2 inconclusive: got {codes}")
    folder = os.path.join(tmp, "vllm_profile")
    os.mkdir(folder)
    with open(os.path.join(folder, "frontend.pt.trace.json"), "w") as f:     # the larger file, CPU only
        json.dump({"traceEvents": [call("cudaGetDevice", i, 1) for i in range(5000)]}, f)
    with open(os.path.join(folder, "worker.pt.trace.json"), "w") as f:
        json.dump(lever_doc, f)
    label, meta, events = ktc.load(folder)
    check(label.startswith("worker.pt.trace.json"), f"a folder reads the largest trace with kernels: {label}")

    # 14. the README's known-good output is the tool's real output, byte for byte
    with open(os.path.join(HERE, "README.md"), encoding="utf-8") as f:
        readme = f.read()
    marker = "```\nkernel_trace_check · h100-train.json.gz\n"
    shown = readme.split(marker, 1)[1].split("\n```", 1)[0] if marker in readme else None
    check(shown is not None and "kernel_trace_check · h100-train.json.gz\n" + shown == report("h100-train.json.gz")[1],
          "README.md shows the H100 fixture's output exactly as the tool prints it")

    # 15. by hand, on the H100 fixture: redo one step's busy time and the largest gap from the raw JSON
    with gzip.open(os.path.join(FX, "h100-train.json.gz"), "rt") as f:
        raw = json.load(f)
    evs = raw["traceEvents"]
    gpu = sorted((e["ts"], e["ts"] + e["dur"]) for e in evs
                 if e["cat"] in ("kernel", "gpu_memcpy", "gpu_memset") and e["args"].get("device") == 1)
    s59 = [e for e in evs if e["name"] == "ProfilerStep#59"][0]
    t0, t1 = s59["ts"], s59["ts"] + s59["dur"]
    covered, reach = 0.0, t0
    for a, b in gpu:
        a, b = max(a, reach), min(b, t1)
        if b > a:
            covered += b - a
            reach = b
    r, _ = report("h100-train.json.gz")
    check(abs(covered - r["windows"][0]["busy_us"]) < 1e-3, f"busy in #59 by hand {covered}, tool "
                                                             f"{r['windows'][0]['busy_us']}")
    g = next((x for x in r["largest_gaps"] if x["kind"] == "launch"), None)
    if g is None:
        check(False, "the H100 fixture must report a host-wait gap between launches")
    else:
        touching = [(a, b) for a, b in gpu if a < g["end_us"] and b > g["start_us"]]
        launch = {e["args"]["correlation"]: e["ts"] for e in evs if e["cat"] == "cuda_runtime"}
        nxt = [e for e in evs if e["cat"] == "kernel" and e["ts"] == g["end_us"] and e["args"].get("device") == 1]
        check(not touching and nxt and all(launch[e["args"]["correlation"]] > g["start_us"] for e in nxt),
              "largest gap: nothing runs inside it, and the next kernel was launched after it began")

    # a step marker of zero length leaves no window: the report says INCONCLUSIVE instead of crashing
    ev = [call("cudaLaunchKernel", 1, 0.5, 1), kernel(GEMM, 2, 5, 1, bps=10.0), step(1, 0, 0)]
    try:
        text = ktc.render(run(ev))
    except Exception as e:  # noqa: BLE001
        text = f"crashed: {type(e).__name__}"
    check("INCONCLUSIVE" in text, f"zero-length step must print INCONCLUSIVE, got {text[-80:]!r}")

    # the most the named lever can save: a graph can at best close every launch gap
    r, text = report("h100-train.json.gz")
    check(abs(r["ceiling_share"] - r["shares"]["idle"]["launch"]) < 1e-9,
          "CUDA graphs' ceiling is the host wait before eager launches")
    check("at most      43.2% of step time saved: 148.0 ms a step down to no less than 84.0 ms, if every launch "
          "gap closes" in text, "the H100 report prints its ceiling per step")
    check(text.count("\nthen ") == 2 and "then         fusion:" in text,
          "every other lever whose bar is met prints on its own then line")
    r, text = report("a100-embedding-train.json.gz")
    check(abs(r["ceiling_us"] - r["kernel_us_by_class"]["elementwise"]) < 1e-6,
          "fusion's ceiling is the eager elementwise kernels' own run time")
    r, text = report("v100-cnn-train.json.gz")
    check(r["ceiling_us"] is None and "at most" not in text, "leave the kernels alone names no saving")

    # --compare: what a change bought, per step, and a warning when the GPU changed too
    a, b = report("a100-embedding-train.json.gz")[0], report("v100-embedding-train.json.gz")[0]
    same = ktc.compare(a, a)
    check("(+0.0%)" in same and "caveat" not in same, "a trace compared with itself changes nothing")
    both = ktc.compare(a, b)
    check("caveat       different GPUs" in both and "77.5 ms to 124.1 ms a step (+60.1%)" in both,
          "a change of GPU is flagged, and the step time moves as measured")
    with contextlib.redirect_stdout(io.StringIO()):
        code = ktc.main(["--compare", os.path.join(FX, "a100-embedding-train.json.gz"),
                         os.path.join(FX, "v100-embedding-train.json.gz")])
    check(code == 0, f"--compare exits 0 on two traces with verdicts, got {code}")
    return ran[0], fails


def main():
    ok = 0
    for fixture, verdict, lever, want in RECORDED:
        r, text = report(fixture)
        missing = [w for w in want if w not in text]
        if r["verdict"] != verdict or r["lever"] != lever or missing:
            print(f"FAIL {fixture}: {r['verdict']} {r['lever']}, missing {missing}\n{text}\n")
        else:
            ok += 1
    print(f"{ok}/{len(RECORDED)} recorded traces pass")
    ran, fails = unit_checks()
    for f in fails:
        print(f"FAIL unit: {f}")
    print(f"{ran - len(fails)}/{ran} unit checks pass")
    return 0 if ok == len(RECORDED) and not fails else 1

if __name__ == "__main__":
    sys.exit(main())
