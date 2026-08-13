# DeepSeek-V4-Flash on 2× DGX Spark, on upstream vLLM

Running **DeepSeek-V4-Flash-0731** across two NVIDIA DGX Sparks (GB10, **sm_121**, TP=2
over a 200G RoCE link) on **upstream-lineage vLLM** — not the community 0.25 port that
every published Spark recipe depends on.

Everything here was measured on real GB10 hardware on 2026-08-12/13. Where a number is
uncertain or a comparison isn't like-for-like, it says so.

---

## TL;DR

| | |
|---|---|
| **It works** | vLLM `main` (419 commits past the v0.27.0 tag) + [vllm-project/vllm#51959](https://github.com/vllm-project/vllm/pull/51959), built for `sm_121` |
| **Weights** | `deepseek-ai/DeepSeek-V4-Flash-0731` @ `9e165c30…`, 79.5 GiB/node |
| **MoE backend** | `DEEPGEMM_MXFP4` (b12x is **not usable for this model at all** — kernel limitation, see finding 2) |
| **KV cache** | `fp8_ds_mla`, tuned to **1.30M tokens / 4.94× concurrency** |
| **Single-stream decode** | **56.9 tok/s** @ 2048-token prompt (their script, like-for-like) |

Stock `vllm/vllm-openai:v0.27.1-aarch64` **cannot** serve this model on GB10. It fails
in DeepGEMM with `Unknown SF transformation`, then `Unsupported architecture`.

**Status update (2026-08-13):** both DeepGEMM failures are now fixed upstream by
[vllm#52035](https://github.com/vllm-project/vllm/pull/52035) (merged 2026-08-12), which
repins DeepGEMM to `deepseek-ai/DeepGEMM 8b1392b9` (nv_dev tip). We validated the
equivalent repin via #51959's branch at `a6b593d`; **we have not tested the merged
`8b1392b9` pin on sm_121**. The b12x blocker (finding 2) remains open.

---

## Three findings worth reporting upstream

### 1. PR #51959 works on real sm_121 hardware

[vllm#51959](https://github.com/vllm-project/vllm/pull/51959) repins DeepGEMM from
`vllm-project/DeepGEMM e21c821` (which ships **no** `sm120_*` kernels — only `sm90_*`
and `sm100_*`) to `deepseek-ai/DeepGEMM a6b593d`, which carries
`sm120_tf32_hc_prenorm_gemm` and the `arch_major == 12` dispatch branch.

Built and validated here: model loads (79.54 GiB, 223 s), serves, generates correct
output. **The PR had zero human validation when we tested it** — only the welcome bot.

We also confirmed [deepseek-ai/DeepGEMM#403](https://github.com/deepseek-ai/DeepGEMM/pull/403)'s
minimal repro now passes on **sm_121** — its author could only test sm_120:

```
device: NVIDIA GB10 (12, 1)
transform_sf_into_required_layout(sf, 128, 256, (1,1,32), 4, False)
  → OK, (4, 128, 2) torch.int32
```

**Caveat:** our build stacks #51959 on top of eugr's sm_12x patches
(`preserve_sm12x_target`, `sm120_cooperative_topk`). We can say #51959 is *necessary and
works in that combination*; we cannot say it is *sufficient alone*.

### 2. `flashinfer_b12x` cannot serve DeepSeek-V4 — and it is a kernel limit, not a missing patch

**Revised 2026-08-13.** An earlier version of this finding said b12x was blocked by a
missing activation patch ([vllm#47392](https://github.com/vllm-project/vllm/pull/47392))
and would work once that landed. **That was wrong.** We then built with #47392's plumbing
present and b12x is still refused — correctly.

The clamp allowlist entry is conditional on the model's *activation*:

```python
# B12x applies the clamp only for SwiGLU-OAI and only when the installed
# FlashInfer wrapper exposes the corresponding activation parameters.
if (config.activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE
        and has_flashinfer_b12x_moe_activation()):
    NVFP4_BACKENDS_WITH_CLAMP.add(NvFp4MoeBackend.FLASHINFER_B12X)
```

In our build the second condition is satisfied (`has_flashinfer_b12x_moe_activation()`
returns `True`). The first is not — `deepseek-ai/DeepSeek-V4-Flash-0731` declares:

```json
"hidden_act": "silu",
"swiglu_limit": 10.0
```

**SILU with a clamp**, not SwiGLU-OAI. `FlashInferB12xExperts` states the constraint
itself: *"FlashInferB12xExperts only applies swiglu_limit with the swigluoai_uninterleave
activation."* So vLLM raises rather than silently skipping the clamp, which is the right
behaviour.

```
ValueError: Model sets swiglu_limit=10.0, but the explicitly requested
moe_backend='flashinfer_b12x' does not apply the SwiGLU clamp.
```

**Consequence:** b12x is unreachable for this model until the FlashInfer b12x kernel
implements the clamp for SILU. That is kernel work, not vLLM plumbing, and no currently
open vLLM PR changes it.

**This also puts a question mark over a widely repeated community claim** — that
`VLLM_USE_B12X_MOE=1` is "the entire speed difference" for DeepSeek-V4 on DGX Spark.
Either those deployments run a checkpoint with a different activation, or they bypass this
guard; the latter would skip the clamp entirely.

### 3. ~~eugr's b12x SwiGLU patch no longer applies to current vLLM main~~ RETRACTED

**This finding was wrong and is retracted.** We hit
`expected one B12x activation support predicate source anchor, found 0` against a clone of
`eugr/spark-vllm-docker` pinned at `3ad5610` (2026-08-11). It was fixed the very next day in
[`21aa9948`](https://github.com/eugr/spark-vllm-docker/commit/21aa9948), which adds an anchor
variant for the newer `_supports_activation` shape that includes `MoEActivation.GELU_TANH`.

Their patch tracks current `main` correctly. We did not re-check before writing it up.
Recorded here rather than deleted, because "verify the upstream state before reporting a
downstream bug" is the actual lesson.

---

## Measured performance

Using **MiaAI-Lab's own `benchmark-0731.py`**, so the comparison is like-for-like.
Single-stream decode, tok/s:

| prompt | MiaAI published (b12x + nvfp4_ds_mla) | ours, baseline | ours, tuned |
|---|---|---|---|
| 256 | **75.4** | 51.5 | 50.1 |
| 2048 | **68.8** | 50.9 | **56.9** |
| 8192 | — | 51.2 | 54.2 |

**We are ~17% behind at p2048 and ~33% behind at p256.** Their config differs in three
ways at once (b12x MoE, `nvfp4_ds_mla` KV, Anemll 0.25 runtime), so the gap cannot be
attributed to the kernel alone.

Their widely-quoted **82.4 / 134.6 tok/s** figures are from a different GUI harness
(2048 completion tokens, 168 ms TTFT) and are **not** comparable to the above. Their own
reproducible JSON — from the script we ran — reports 75.4 / 68.8.

### Where tuning did help: KV capacity

Four upstream-compatible flags, no code changes:

| | before | after |
|---|---|---|
| Available KV memory | 13.9 GiB | **21.15 GiB** |
| GPU KV cache size | 482,343 tok | **1,308,983 tok** (+171%) |
| Max concurrency @262K ctx | 1.84× | **4.99×** |
| Engine init | 158 s | 71 s |

```
--gpu-memory-utilization 0.85            (was 0.80)
--max-num-batched-tokens 4096            (was 8192)
--max-cudagraph-capture-size <seqs×(k+1)>
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
```

Credit: these came from the eugr/MiaAI/Keys community docs and the NVIDIA forum thread;
we verified them, we didn't invent them.

---

## What did NOT work

| Attempt | Outcome |
|---|---|
| `--moe-backend flashinfer_b12x` (MXFP4 weights) | rejected — not in the MXFP4 oracle |
| `--moe-backend deep_gemm` | loads, dies in kernel: `Unknown SF transformation` |
| `--moe-backend triton` | "kernel does not support current device" |
| `--moe-backend marlin` (MXFP4) | selected, loads, **same DeepGEMM error** → proved MoE was never the fault |
| `VLLM_USE_DEEP_GEMM=0` | got past load, then `hyperconnection.hpp: Unsupported architecture` |
| NVFP4 checkpoint + b12x | SwiGLU clamp (finding #2) |
| NVFP4 checkpoint + auto → MARLIN | weight-shape `AssertionError` in the loader |
| `VLLM_USE_BREAKABLE_CUDAGRAPH=0` | **no decode gain** — the published +28.6% was measured against Anemll's opposite default; vLLM-lineage default is already `0` |

---

## Measurement notes — please read before quoting any tok/s

See [`docs/MEASUREMENT-NOTES.md`](docs/MEASUREMENT-NOTES.md). Short version: published
figures for this model vary wildly because harnesses measure different things.

- **Decode-with-TTFT vs decode-only** differ by ~1.6× on identical runs.
- **Prefix caching**: a deterministic prompt nonce made prefill read ~1500 tok/s; a random
  nonce on the same config read ~200 tok/s. Both "real", totally different quantities.
- **Output-length variance** wrecks aggregate figures — a straggler tanks a whole cell.
- **Spec decoding delivers ~12 tokens per SSE chunk**, so chunk-timing methods (including
  our own `bench/bench_fixed.py`) produce unstable decode numbers. **That harness is
  published as a cautionary artifact, not a recommendation** — it produced 117, then 24.5,
  then 503 tok/s for near-identical configs.

---

## Reproducing

```bash
# 1. build vLLM with the DeepGEMM sm120 repin (uses eugr's harness)
git clone https://github.com/eugr/spark-vllm-docker && cd spark-vllm-docker
# make the stale b12x swigluoai patch non-fatal (see finding #3), then:
sudo ./build-and-copy.sh \
  --vllm-repo https://github.com/Mirrdhyn/vllm.git \
  --vllm-ref gb10/deepgemm-sm12x \
  --torch-version 2.13.0 --gpu-arch 12.1a -t vllm-node-pr51959

# 2. configure
cp deploy/.env.example .env      # fill in your RoCE IPs, ranks, HF cache path
#    worker sets NODE_RANK=1 and HEADLESS=1; head sets NODE_RANK=0, HEADLESS=

# 3. boot WORKER FIRST, then head
docker compose -f deploy/docker-compose.ds4.yml --env-file .env up -d

# 4. verify
bash deploy/scripts/smoke_ds4.sh
```

`--headless` on the worker is **required** — the CLI help says it's for "multi-node data
parallel", which is misleading; `entrypoints/cli/serve.py:213` reads *"Run headless
workers (for multi-node PP/TP)"*.

---

## Environment

- 2× NVIDIA DGX Spark, GB10 Blackwell **sm_121**, 128 GB unified LPDDR5X each
- 200G ConnectX-7 QSFP56 DAC, RoCEv2 — measured 111 Gb/s `ib_write_bw`, ~13.6 GB/s NCCL all-reduce
- DGX OS (Ubuntu 24.04), kernel 6.17-nvidia, CUDA 13.0, driver 580.173.02
- vLLM `main` @ `43c9335b4` (PR #51959 head), torch 2.13.0+cu130, FlashInfer 0.6.18, Triton 3.7.1
- `TORCH_CUDA_ARCH_LIST=12.1a`, NCCL gencode `sm_121`, NCCL 2.30.7

## Credits

The tuning knowledge here is the community's, not ours. Verified against and indebted to
[MiaAI-Lab](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark),
[eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker),
[drowzeys ("Keys")](https://github.com/drowzeys), tonyd2wild, and the
[NVIDIA developer forum thread](https://forums.developer.nvidia.com/t/instructions-for-running-deepseek-v4-flash-with-dspark-using-eugrs-repo/376220).
`bench/results/benchmark-0731.py` outputs come from MiaAI-Lab's script.

## Licence

MIT for the scripts and docs here. vLLM, DeepGEMM and FlashInfer retain their own
(Apache-2.0) licences; model weights retain theirs.

---

## Rebuild on current main (2026-08-13)

Rebuilt against vLLM `main` @ `0.27.2rc1.dev48+g64ca614fe` — i.e. **after** #52035 merged,
with DeepGEMM pinned at `8b1392b9` (nv_dev tip) rather than the `a6b593d` we originally
validated.

**The merged pin works on sm_121.** `transform_sf_into_required_layout` passes on device
capability (12, 1); the model loads (79.25 GiB / 217 s) and serves.

**Performance is unchanged** — same-script single-stream decode, tok/s:

| prompt | `a6b593d` (PR #51959) | `8b1392b9` (merged #52035) |
|---|---|---|
| 256 | 50.1 | 52.1 |
| 2048 | 56.9 | 54.1 |
| 8192 | 54.2 | 57.0 |

Mixed signs, single sample per cell — treat as no measurable difference. KV pool
1,293,774 vs 1,296,109 tokens; both select `DEEPGEMM_MXFP4`.
