# kernel-trace-check

> Tested on eight recorded torch.profiler traces from Meta's Holistic Trace Analysis test data, V100 to H100. Not yet run on a live vLLM serving trace.

Companion to [What is a GPU Kernel?](https://theaiengineer.substack.com/p/what-is-a-gpu-kernel), The AI Engineer.

## Start here: watch the three speedups on your own GPU

`kernel_speedups.ipynb` reproduces the issue's three effects in about two minutes on any NVIDIA GPU:
a size padded to a multiple of 64, two steps merged into one kernel, and small kernels replayed as a
CUDA graph. [Open it in Colab](https://colab.research.google.com/github/paolo-perrone/kernel-trace-check/blob/main/kernel_speedups.ipynb),
pick Runtime > Change runtime type > T4 GPU, then Runtime > Run all. On a free Colab T4 with PyTorch 2.11
on 2 October 2026 it printed:

    vocabulary 50,257: 44.22 ms per multiply
    vocabulary 50,304: 26.80 ms per multiply
    the padded size runs 1.65x as fast
    two kernels: 4.54 ms | one merged kernel: 2.28 ms | 2.0x as fast
    20 layers, one kernel each: 0.538 ms | the same kernels replayed as a CUDA graph: 0.063 ms | 8.5x as fast

Your numbers will differ by GPU. Then run the checker below on a trace of your own training loop to
see which of the three your model needs.

## The checker

For engineers who train or serve a model with PyTorch or vLLM on NVIDIA GPUs,
and want to know what their GPU time is waiting on before they touch a kernel:
the CPU, SMs left empty, memory, or nothing at all.

It reads the trace you already export with torch.profiler (Chrome-trace JSON,
plain or .gz, vLLM's `--profiler-config` traces included) and prints how much of
each profiled step the GPU sat idle and why, how kernel time splits across
matmul, attention, NCCL and memory-bound kernels, and how much of it ran on
grids smaller than the SM count. Then it names ONE lever, with the number that
decided it and the three kernels behind that number: CUDA graphs, a bigger
batch or a rounded shape, fusion, or leave the kernels alone.

    python3 kernel_trace_check.py trace.json.gz

When a trace cannot carry a verdict (no kernels, a warm-up pass, NCCL over half
the kernel time, or a GPU idle for a reason no launch change reaches), it prints
INCONCLUSIVE and the number that stopped it instead of guessing. Exit code 0 on a
lever, 2 on INCONCLUSIVE, 1 on a file it cannot read.

Read-only, standard-library Python, no GPU and no API key: copy the trace off the
machine and run it on a laptop.

## What it prints, on a case with a known answer

`fixtures/h100-train.json.gz` is a real trace from the test data of Meta's Holistic Trace Analysis:
six steps of a training job on an H100 (rank 1 of 2) with embedding lookups, layer norms and NCCL.

```
python3 kernel_trace_check.py fixtures/h100-train.json.gz
```

```
kernel_trace_check · h100-train.json.gz
device       1, NVIDIA H100, 132 SMs (from deviceProperties)
windows      6 steps (ProfilerStep#59 to #64), 888.2 ms of step time
busy         51.5% of step time: a kernel, copy or fill running on at least one of 4 streams
idle         48.5%: host wait 43.7% over 18,193 gaps, other 4.9% with the next work already queued
host wait    43.2% before eager launches, 0.4% after the CPU blocked in a CUDA call
kernels      37,614 in the steps, median 3 us, 498.4 ms in all: NCCL 23.9%, compute 76.1%
compute      379.4 ms: elementwise 40.4%, matmul 36.2%, other memory-bound 23.4%
grids        30.0% of compute-kernel time on grids under 1 block per SM
largest gap  6.74 ms of host wait from ts 1689861289120202, before vectorized_elementwise_kernel<4, FillFunc...

LEVER        CUDA graphs: the GPU sat idle waiting for the CPU to launch the next kernel for 43.2% of step time
             37,614 kernels of median 3 us went out one launch at a time, slower than the GPU ran them
             torch.compile(mode="reduce-overhead") replays them as one CUDA graph; vLLM does unless it runs with --enforce-eager
             the 0.4% after a blocking CUDA call stays: a graph cannot span a sync
at most      43.2% of step time saved: 148.0 ms a step down to no less than 84.0 ms, if every launch gap closes
waited on      idle     gaps     median  grid        block      blk/SM  kernel
               4.6%    1,215       3 us  512x1x1     128x1x1      3.88  layer_norm_grad_input_kernel<float, float>
               3.1%    1,197       3 us  512x1x1     32x4x1       3.88  vectorized_layer_norm_kernel<float, float>
               2.8%    1,239       1 us  1x1x1       128x1x1      0.01  unrolled_elementwise_kernel<AUnaryFunctor<float, floa...
then         a bigger batch or a rounded shape: 30.0% of compute-kernel time ran on grids under 1 block per SM
then         fusion: eager elementwise kernels take 40.4% of compute-kernel time
```

The GPU ran something for half of each step. For 43.2% of the step it sat idle because the next
kernel had not been launched yet, over thousands of gaps between kernels that run for 3
microseconds: launch overhead, which a CUDA graph removes by replaying the whole sequence from one
launch. The `at most` line is the ceiling on that fix: if every launch gap closed, a 148.0 ms step
would drop to 84.0 ms and no lower. The `waited on` rows are the kernel shapes the GPU waited for
longest, and each `then` line is another lever whose bar this trace also meets, in the order the
rule checks them.

## Export a trace

From a PyTorch loop, profile five steps and keep the last three:

```python
from torch.profiler import profile, schedule, tensorboard_trace_handler, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             schedule=schedule(wait=1, warmup=1, active=3),
             on_trace_ready=tensorboard_trace_handler("./trace", use_gzip=True)) as prof:
    for _, batch in zip(range(5), loader):
        train_step(batch); prof.step()
```

then run `python3 kernel_trace_check.py ./trace`. The schedule matters twice: `prof.step()` writes
the `ProfilerStep#N` markers the tool uses as windows, and the skipped first step is the one that
allocates memory and loads kernels.

From vLLM (v0.13 or later), start the server with a profiler config and bracket some traffic:

```
vllm serve <model> --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile", "warmup_iterations": 2, "active_iterations": 5, "torch_profiler_with_stack": false}'
curl -X POST localhost:8000/start_profile
# send a few requests
curl -X POST localhost:8000/stop_profile
python3 kernel_trace_check.py ./vllm_profile
```

A `warmup_iterations` above zero turns on vLLM's profiler schedule, and that is what writes one
`ProfilerStep#N` per engine step; without it the tool splits the timeline at idle stretches of 50 ms
and says so. Given a folder, it reads the largest trace in it that holds kernels, because vLLM writes
one per worker and can add a CPU-only one for its frontend. Stack recording is off in that config
because the tool never reads it and it multiplies the file size.

## What to change, lever by lever

- **CUDA graphs.** In a PyTorch loop, `model = torch.compile(model, mode="reduce-overhead")` records
  the step's launches as a CUDA graph and replays them with one launch. It needs the same shapes every
  step (pad or bucket the batch) and no CPU sync inside the step: no `.item()`, no printing a tensor,
  no branching on a GPU value. vLLM captures graphs by default; remove `--enforce-eager` if it is set.
- **A bigger batch or a rounded shape.** Raise the batch (`--max-num-seqs` in vLLM) until step time
  per sample stops falling, or pad the dimension that sets the grid to a multiple of 64, the way
  nanoGPT's vocabulary went from 50,257 to 50,304.
- **Fusion.** `model = torch.compile(model)` in its default mode fuses chains of elementwise, norm and
  reduction kernels into single Triton kernels. vLLM already compiles its models, so the gain there is
  a custom op, not a flag.
- **Leave the kernels alone.** The kernels are doing the math. The next gain is less math: bf16 or fp16
  autocast when the FP32 line appears, a smaller model, or quantized weights.

Then profile the same steps again and check what the change bought.

## Check what the change bought

`--compare BEFORE AFTER` reads two traces of the same job and prints what moved: the step time, the
busy and host-wait shares, the kernel count and the compute split. `fixtures/v100-embedding-train`
holds the same six training steps as `fixtures/a100-embedding-train`, run on a V100 (Holistic Trace
Analysis ships the pair for its own trace-diff tests), so the example below compares hardware
rather than a code change, and the `caveat` line says so:

```
python3 kernel_trace_check.py --compare fixtures/a100-embedding-train.json.gz fixtures/v100-embedding-train.json.gz
```

```
kernel_trace_check · compare
before       a100-embedding-train.json.gz: 6 windows on NVIDIA A100-PG509-200, 77.5 ms a step, LEVER fusion
after        v100-embedding-train.json.gz: 6 windows on Tesla V100-SXM2-16GB, 124.1 ms a step, LEVER leave the kernels alone
caveat       different GPUs: the difference is the hardware as much as any change
step time    77.5 ms to 124.1 ms a step (+60.1%)
busy         83.6% to 96.3% of step time
host wait    13.0% to 0.1% before eager launches
kernels      8,568 to 9,876, median 9 to 11 us
compute      matmul 29.2% to 37.0%, elementwise 35.3% to 29.0%, other memory-bound 35.6% to 34.1%
NCCL         36.6% to 31.3% of kernel time
```

## Next to Holistic Trace Analysis

Meta's Holistic Trace Analysis reads every rank of a job and breaks its time down many ways, and
this tool borrows its test for host wait. What this adds is the decision: one lever, the number that
decided it, the most that lever can save, and a before-and-after compare, from a single file that
needs nothing beyond the Python standard library.

## How each number is computed

- **windows**: the CPU span of each `ProfilerStep#N` event, summed. With no step markers, the GPU
  timeline splits at idle stretches of 50 ms or more (`--split-ms`). A window holding a warm-up call
  (`cudaMalloc`, `cudaFree`, stream creation, a device query, a module load) is left out and named on
  a `caveat` line.
- **busy**: the union of every kernel, copy and fill interval on the chosen GPU, over all streams,
  clipped to the windows, divided by their total.
- **idle** and **host wait**: each stretch where nothing runs is a gap. It is host wait when the CPU
  call that launched the next GPU work started after the gap began, so the queue was empty and the
  GPU waited on the CPU. That is Holistic Trace Analysis's test (`ts_runtime > prev_end_ts`), applied
  to the union of streams, so a gap on one stream while NCCL runs on another is not idle. Host wait
  splits four ways: before an eager launch; before a CUDA graph replay (`cudaGraphLaunch`); after the
  CPU sat in a blocking CUDA call (a sync, a copy, an allocation) while the GPU drained; or while the
  CPU sat in a DataLoader `__next__`. Other idle means the next work was already queued: it waited on
  another stream, another rank or a copy. Each share is rounded on its own, so parts can miss their
  total by 0.1.
- **kernels** and **compute**: the kernels launched inside the windows (a kernel counts for the step
  its launch call fell in), summed by class. Compute is kernel time minus NCCL. The class comes from
  the kernel name without its argument list:
  - NCCL: `nccl`, `rccl`, vLLM's custom all-reduce;
  - attention: `flash`, `fmha`, `attention`, `attn`, `sdpa`, `mla`;
  - matmul: `gemm`, `gemv`, `cutlass`, `cublas`, `xmma`, `wgmma`, `nvjet`, `scudnn`, `conv`, `dgrad`,
    `wgrad`, `marlin`, `machete`, fused MoE and Triton matmul templates;
  - elementwise: eager PyTorch kernels (ATen elementwise ops, reductions, norms, softmax, copies,
    cuDNN batch norm), the chains torch.compile fuses;
  - compiled: Triton kernels torch.compile already generated (`triton_poi`, `triton_red`, `triton_per`);
  - other memory-bound: the rest, mostly library kernels fused by their authors (embedding lookups,
    sorts, multi-tensor optimizers, vLLM custom ops).
- **grids**: compute-kernel time in kernels whose `blocks per SM` (grid blocks divided by the SM
  count, as Kineto records it) is under 1.
- **at most**: the most the named lever can save. For CUDA graphs it is the host wait before eager
  launches, since a graph can at best close every one of those gaps; for fusion it is the eager
  elementwise kernels' own run time. With step markers it prints the step time now and the step time
  minus that ceiling. The real saving is lower: a graph cannot capture code with a sync or a
  data-dependent shape, and a fused kernel still reads and writes memory once. The other two levers
  print no ceiling, because a bigger batch changes the work itself.
- **largest gap**: the longest host-wait gap and the timestamp where it starts, in microseconds as
  in the trace. Open the trace in ui.perfetto.dev, go to that time, and the GPU rows stay empty until
  the named kernel, whose launch call starts after the gap began.

`--json` prints every number, per window, with the 20 longest gaps and the call that launched the
work after each.

## The rule

The first line that applies decides, so the tool never names two levers. Every other lever whose
bar the trace also meets prints below the verdict on its own `then` line, in rule order.

1. **INCONCLUSIVE** when the trace has no kernel events, when every window is a warm-up, when fewer
   than 50 kernels remain, or when NCCL kernels take over 50% of kernel time.
2. **CUDA graphs** when host wait before eager launches is 30% of step time or more.
3. **INCONCLUSIVE** when idle that no launch change reaches (queued work, a blocking call, the
   DataLoader, gaps between graph replays) is 30% of step time or more. The line names the largest
   part and the CPU call that ran through the longest gaps.
4. **A bigger batch or a rounded shape** when kernels on grids under 1 block per SM take 25% of
   compute-kernel time or more.
5. **Fusion** when eager elementwise kernels take 40% of compute-kernel time or more, or 25% or more
   and more than matmul and attention together.
6. **Leave the kernels alone** otherwise, naming the group that holds the time: matmul and attention
   (with a line when most of that math runs FP32 kernels without tensor cores), compiled kernels, or
   library kernels.

### Why the bars sit where they do

They were set by running the rule by hand on ten real traces, before the tests froze them. Each
line gives the margin: the range over which the bar could move without changing a verdict here.

- **Host wait, 30%.** The two traces it fires on sit at 43.2% and 77.1%; the highest judged trace
  below them sits at 13.0%. Any bar from 14% to 43% gives the same verdicts on this corpus.
- **Idle no launch reaches, 30%.** One real trace sits at 64%: a job replaying CUDA graphs whose GPU
  idled 63.1% of each step with its next work queued, while its CPU thread sat about 140 ms in one
  `cudaMemcpyAsync` per step. Every other judged trace sits under 6%, so any bar from 6% to 63%
  agrees. Naming a kernel lever there would point the owner at the wrong half of the step.
- **NCCL, 50%.** The two collective-bound traces sit at 63.7% and 65.3%; the highest below sits at
  47.4%. Any bar from 48% to 63% agrees.
- **Warm-up.** `cudaMalloc`, `cudaFree`, stream creation, device queries and module loads mark a
  window that is still setting up. Occupancy queries do not: the H100 trace makes 5,022 of them in
  steady state. The two cold traces in the corpus hold 78 and 79 of their 79 kernels in such windows,
  where single `cudaMalloc` calls run up to 765 ms, so their host wait describes a first pass
  nobody ships.
- **Fusion, 40% or outweighing the math.** A fixed 40% alone sent an embedding training job (35.3%
  eager elementwise against 29.2% matmul) to "leave the kernels alone", which is false when the
  unfused time outweighs the math. The 25% floor keeps a trace dominated by library kernels out.
  The nearest trace on the other side, a convolution job at 34.5% elementwise against 63.6% matmul,
  is left alone with its FP32 line.
- **Small grids, 25%.** The two traces with most small-grid time sit at 30.0% and 34.9%, both behind
  CUDA graphs, where the tool names it on the `then` line; the next sits at 19.2%.
- **50 kernels.** The real training and inference traces hold 1,154 to 37,614 kernels in their
  windows; every trace under 50 in the corpus is a single-op benchmark or a cold pass.

## Check it before you trust it

`python3 test_kernel_trace_check.py` replays eight real traces with known verdicts, one or more for
every lever and refusal the corpus reaches, then runs 55 unit checks, most on hand-written traces
where every gap is known in advance. It must print:

```
8/8 recorded traces pass
55/55 unit checks pass
```

One unit check redoes the H100 fixture by hand from its raw JSON: the busy time of ProfilerStep#59
from an independent sweep, and the largest gap (nothing runs inside it, and the next kernel's launch
call started after it began). Another holds the output printed above to the tool's real output, byte
for byte. The fixtures are the real traces trimmed to the events the tool reads, and the tool prints
the same report on each as on its full source; `fixtures/README.md` gives the source of each, what
was trimmed, and HTA's MIT license.

## What it does not do

- It reads one GPU: the busiest, or `--device N`. A collective waits on the slowest rank, so a trace
  dominated by NCCL is refused rather than judged; read every rank together with Holistic Trace
  Analysis.
- It names a lever and the most that lever can save; it cannot make the change. Make it, profile
  again, and run `--compare` on the two traces.
- It does not see inside a kernel. Memory bandwidth, achieved occupancy and tensor-core use are
  Nsight Compute's job; the one inference from a name is the FP32 line (`sgemm`, `scudnn`).
- A small grid here means under one block per SM. A grid of 1.05 waves wastes most of its second
  wave and does not show up.
- Classes come from kernel names. A custom kernel with an unfamiliar name lands in other
  memory-bound, which the fusion lever never counts: read the top-3 rows before acting.
- Host wait before an eager launch includes any Python the CPU ran between two launches. A graph
  removes it only when that stretch of code can be captured: no syncs, no data-dependent shapes, no
  CPU-side branching on GPU values.
- It has not read a trace from vLLM itself: no public one was available. vLLM writes the same Kineto
  format, and graph replays were tested on a real `cudaGraphLaunch` trace and on hand-written ones.

## Support

Tested on 2026-09-29 with Python 3.9 and 3.14; CI runs 3.9, 3.12 and 3.13. Standard library only,
so there is no lockfile to drift. It reads Kineto's Chrome-trace JSON (`schemaVersion` 1), tested on
traces from 2022 (the older `Kernel` and `Runtime` category names) to 2026 (a Blackwell trace from
PyTorch 2.10), with launches through `cudaLaunchKernel`, `cudaLaunchKernelExC`, `cuLaunchKernel`,
`cudaGraphLaunch` and their HIP equivalents. Fields read: each GPU event's `ts`, `dur`, `name` and
args `device`, `stream`, `correlation`, `grid`, `block`, `blocks per SM`; each API call's `ts`,
`dur`, `name` and `correlation`; `ProfilerStep#N` and `enumerate(DataLoader)` spans; and the
top-level `deviceProperties[].numSms`, derived from grid blocks and blocks per SM when it is absent.
Memory runs about six times the uncompressed trace: a 181 MB synthetic trace holding 400,000
kernels read in 9.4 seconds with 1.1 GB. Supported through 2027-09-30: the field names get
re-checked against each PyTorch and vLLM release in that window. Open an issue if a trace from your
version will not parse.
