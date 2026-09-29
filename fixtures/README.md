# Where the fixtures come from

Every fixture is a real torch.profiler trace from the test data of Meta's Holistic Trace Analysis
(HTA), github.com/facebookresearch/HolisticTraceAnalysis, read at commit `0502deb4` (main on
2026-09-24; `tests/data` last changed at `594e90ee` on 2026-01-17). HTA ships them to test its own
analyzers. Each is a real Kineto export with its own device properties and timings (393,683 events
in the H100 one), not a synthetic file. The file names below are ours, chosen for what the kernels
in each trace show.

| Fixture | HTA path under `tests/data/` (blob) | GPU | Verdict |
|---|---|---|---|
| `h100-train.json.gz` | `h100/h100_trace.json.gz` (`6f54a7e7ea24`) | H100, 132 SMs | LEVER CUDA graphs, 43.2% of step time |
| `v100-inference.json.gz` | `inference_single_rank/inference_rank_0.json.gz` (`0f0db8ca5ca3`) | 80 SMs, derived | LEVER CUDA graphs, 77.1% |
| `v100-cnn-train.json.gz` | `ns_resolution_trace/rank-0.Apr_03_18_51_38.1102.pt.trace.json.gz` (`e8fe4d1b5b47`) | V100, 80 SMs | LEVER leave the kernels alone, FP32 math |
| `a100-embedding-train.json.gz` | `trace_diff/control/control.json.gz` (`b1ceb4c08168`) | A100, 108 SMs | LEVER fusion, 35.3% |
| `v100-embedding-train.json.gz` | `trace_diff/test/test.json.gz` (`949c3c47e47c`) | V100, 80 SMs | LEVER leave the kernels alone, FP32 math |
| `a100-graphs-train.json.gz` | `negative_queue_length_values_check/rank0.json.gz` (`d71a78d5110a`) | A100 80GB, 108 SMs | INCONCLUSIVE, 63.1% idle with work queued |
| `a100-nccl-train.json.gz` | `timeline_analysis/sampled_rank-0.json.gz` (`c56feed5f481`) | 108 SMs, derived | INCONCLUSIVE, NCCL 65.3% |
| `a100-alexnet-first-pass.json.gz` | `critical_path/alexnet/benchmark_result_2869224_1695835535_trace.json.gz` (`4bcd84bf13cb`) | A100, 108 SMs | INCONCLUSIVE, a warm-up pass |
| `cpu-only.json.gz` | `cpu_only/rank-34.Jul_15_10_52_41.1074.pt.trace.json.gz` (`1aff90f85c87`) | none | INCONCLUSIVE, no kernels |

What each trace shows, read from its kernels and its own metadata: `h100-train` is one of two
ranks of a training job with embedding lookups, layer norms and NCCL all-to-all; `v100-inference`
is one step of fp16 inference on a single stream; `v100-cnn-train` is convolution training in
FP32 (cuDNN, batch norm) under DDP with one rank; `a100-embedding-train` is rank 3 of 8 of a
training job built on FBGEMM embedding tables; `v100-embedding-train` is HTA's trace-diff partner to `a100-embedding-train`, the same six step numbers run on a V100, shipped for the `--compare` example; `a100-graphs-train` is rank 0 of 16 of a job that
replays CUDA graphs (`cudaGraphLaunch`) holding Triton and CUTLASS kernels; `a100-nccl-train` is
rank 0 of 128, sampled; `a100-alexnet-first-pass` is one cold AlexNet pass with no backward
operators and no step markers; `cpu-only` has a step marker and no GPU activity.

## What was trimmed

Each fixture keeps only what `kernel_trace_check` reads:

- the top-level `schemaVersion` and `deviceProperties`, verbatim;
- every kernel, copy and fill event (`kernel`/`Kernel`, `gpu_memcpy`/`Memcpy`, `gpu_memset`/`Memset`),
  with the fields `ph`, `cat`, `name`, `pid`, `tid`, `ts`, `dur` and the args `device`, `stream`,
  `correlation`, `grid`, `block` and `blocks per SM`;
- the CUDA runtime and driver calls that launched those events (matched on `correlation`), plus
  every warm-up call (`cudaMalloc`, `cudaFree`, stream creation, device queries, module loads) and
  every blocking call (syncs, copies), with `ph`, `cat`, `name`, `pid`, `tid`, `ts`, `dur` and
  `correlation`;
- the `ProfilerStep#N` markers and PyTorch's `enumerate(DataLoader)` spans.

Everything else is dropped: CPU operators, Python function events, flow arrows, thread metadata,
the per-kernel fields the tool never reads (`registers per thread`, `shared memory`, `warps per SM`,
`est. achieved occupancy %`, `queued`, `context`, `External id`), and top-level job metadata such as
host, user and cluster names. Nothing was reconstructed: every kept value is the source's own; the
JSON was re-serialized without whitespace and re-gzipped.

The trimming is lossless for this tool. On 2026-09-28, `kernel_trace_check.py` printed the same
report, and the same `--json`, on each fixture as on its full HTA source file. `v100-embedding-train` was trimmed and checked the same way on 2026-09-29.

## Traces read but not shipped

To keep the folder under 3 MB, these ran by hand against their full HTA files and are not
redistributed: `vision_transformer/rank-0.json.gz` (V100, 64 ranks: INCONCLUSIVE, NCCL 63.7% of
kernel time), `critical_path/simple_add/...` (INCONCLUSIVE, a warm-up pass), and HTA's small
traces (`triton_example`, `amd_trace`, `trace_compare/base`, `trace_file_list`, `rank_non_gpu`,
`cupti_profiler`), which the tool refuses for having too few kernels or none.

Five micro-benchmark traces from github.com/bojieli/ai-infra-book (Apache-2.0; `experiments/ch04/04-02`
and `experiments/ch05/05-04`, an RTX PRO 6000 Blackwell with 188 SMs) were used to hand-check the
kernel classes on 2026 kernel names. They hold 13 to 30 kernels each, under the tool's floor of 50,
so they carry no verdict and are not included.

## HTA's license

The traces above are redistributed under HTA's MIT license, reproduced here as it requires:

```
MIT License

Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
