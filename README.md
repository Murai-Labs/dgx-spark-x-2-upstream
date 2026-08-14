# DeepSeek-V4-Flash on 2× DGX Spark, on upstream vLLM

Running **DeepSeek-V4-Flash-0731** across two NVIDIA DGX Sparks (GB10, **sm_121**, TP=2
over a 200G RoCE link) on **upstream-lineage vLLM** — not the community 0.25 port that
every published Spark recipe depends on.

Everything here was measured on real GB10 hardware on 2026-08-12/13. Where a number is
uncertain or a comparison isn't like-for-like, it says so. Several findings in this repo
are **corrections of earlier versions of this same repo**; those are kept rather than
deleted, because the corrections are the useful part.

---

## TL;DR

| | |
|---|---|
| **It works** | vLLM `main` (post-[#52035](https://github.com/vllm-project/vllm/pull/52035)), built for `sm_121` |
| **Weights** | `deepseek-ai/DeepSeek-V4-Flash-0731` @ `9e165c30…`, ~79.4 GiB/node |
| **MoE backend** | **`b12x` — works, after locally disabling an unsafe FC2 tile upgrade (finding 4)** |
| **KV cache** | `fp8_ds_mla` @ `--gpu-memory-utilization 0.88`, 1,389,891 tokens |
| **Single-stream decode** | **63.7 tok/s** @ 2048-token prompt — **+23.6%** over our DEEPGEMM baseline (n=5, their script, like-for-like) |
| **Still behind** | MiaAI-Lab publish **68.8** at the same point. We are ~7.4% short. |

Stock `vllm/vllm-openai:v0.27.1-aarch64` **cannot** serve this model on GB10. It fails
in DeepGEMM with `Unknown SF transformation`, then `Unsupported architecture`. Both are
fixed upstream by [#52035](https://github.com/vllm-project/vllm/pull/52035).

---

## What is actually ours

This repo sits on top of a lot of other people's work. To make that boundary explicit:

**New here — less than we first claimed:**

- **The `b12x` FC2 tile round-trip failure** (finding 4). We hit it in the field, traced
  it, and filed it. We did **not** discover it: two open upstream PRs
  ([#39](https://github.com/local-inference-lab/b12x/pull/39),
  [#146](https://github.com/local-inference-lab/b12x/pull/146)) already fix it, both
  predating our report, and both with a better fix than ours. What is genuinely ours is
  the field repro and the shape (`N/K=4096/1024`, `moe_block_size=8`) that neither PR
  names. Our first version of this section claimed the root cause as new; that was wrong.

**Validation of other people's changes on hardware they couldn't reach:**

- [vllm#51959](https://github.com/vllm-project/vllm/pull/51959) on real **sm_121** — the PR
  had *zero* human validation when we tested it.
- The **merged** [#52035](https://github.com/vllm-project/vllm/pull/52035) pin (`8b1392b9`)
  on sm_121 — a different commit from the one we first validated.
- [DeepGEMM#403](https://github.com/deepseek-ai/DeepGEMM/pull/403)'s minimal repro on
  sm_121; its author could only test sm_120.
- b12x serving DeepSeek-V4-Flash across two nodes. We have not seen this reported
  elsewhere, but we make no claim to being first.

**Negative results**, in the [What did NOT work](#what-did-not-work) table — eight dead
ends with the exact errors, plus two measured refutations: `k=7` *reduces* throughput, and
`VLLM_USE_BREAKABLE_CUDAGRAPH=0` yields **no gain** on a vLLM-lineage build (the published
+28.6% was measured against Anemll's opposite default, so it does not transfer).

**Measurement methodology** — [`docs/MEASUREMENT-NOTES.md`](docs/MEASUREMENT-NOTES.md).
Published tok/s for this model spans more than 3×, almost entirely from methodology. This
may be the most reusable part of the repo.

**Corrections to the public record**, three of them, two to our own earlier claims —
including one posted on a vLLM PR and corrected there. They are kept in place below rather
than quietly edited out.

**Not ours:** the recipe and `benchmark-0731.py` (MiaAI-Lab), the sm_12x build harness and
b12x image (eugr), the DeepGEMM sm120 repin (its PR authors), the KV/memory flag set
(community). See [Credits](#credits).

**A second, weaker candidate** — `NvFp4MoeBackend.B12X` is missing from vLLM's
`NVFP4_BACKENDS_WITH_CLAMP` although b12x does honour `swiglu_limit` for SILU
([`deploy/patches/nvfp4_clamp_allowlist.py`](deploy/patches/nvfp4_clamp_allowlist.py)).
Patching it changes backend selection as intended, but we could not exercise it
end-to-end because the NVFP4 checkpoint fails in the weight loader first. **We verified
the plumbing, not the arithmetic.** Treat it as a lead, not a validated fix.

### Filed upstream

**Finding 4 is filed as
[local-inference-lab/b12x#182](https://github.com/local-inference-lab/b12x/issues/182).**

b12x lives at [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x)
(Apache-2.0, author Luke Alonso). The *installed distribution metadata* carries no homepage
or project URL, which is why earlier revisions of this README said the venue was unknown.

The defect is present on current `master`, not only in the 1.2.3 vendored by
`eugr/spark-vllm-b12x` — checked against the repository before filing.

**It is a duplicate.** [#39](https://github.com/local-inference-lab/b12x/pull/39) (Jul 18)
and [#146](https://github.com/local-inference-lab/b12x/pull/146) (Aug 12) both fix it and
both predate our report; we did not find them before filing. A maintainer review of #182
also corrected three claims in our original report — see finding 4.

**Timing note.** b12x is being integrated into vLLM
([vllm-project/vllm#40082](https://github.com/vllm-project/vllm/pull/40082)). If it lands
before either PR does, every GB10 user hits this on first boot — after a full weight load
— with an error that names the tile but not the round-trip that produced it.

The clamp-allowlist item is **not** filed. It is a lead, not a validated fix (see above),
and we would rather test it end-to-end first than file something we cannot stand behind.

---

## Findings

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

The work has since merged as [#52035](https://github.com/vllm-project/vllm/pull/52035),
pinning `8b1392b9` (nv_dev tip) rather than the `a6b593d` we first validated. **We
subsequently validated the merged pin too** — it loads and serves on sm_121, with no
measurable performance difference from `a6b593d`.

**Caveat:** our build stacks this on eugr's sm_12x patches (`preserve_sm12x_target`,
`sm120_cooperative_topk`). We can say the repin is *necessary and works in that
combination*; we cannot say it is *sufficient alone*.

### 2. `flashinfer_b12x` cannot serve DeepSeek-V4 — a kernel limit, not a missing patch

**This finding is about `flashinfer_b12x` on the NVFP4 path. It is a different backend
from the `b12x` MoE backend in finding 4, which does work.** Conflating the two is easy
and we did it ourselves; see finding 4.

An earlier version of this finding said `flashinfer_b12x` was blocked by a missing
activation patch ([vllm#47392](https://github.com/vllm-project/vllm/pull/47392)) and would
work once that landed. **That was wrong.** We built with #47392's plumbing present and
b12x is still refused — correctly.

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
activation."* So vLLM raises rather than silently skipping the clamp, which is correct
behaviour.

```
ValueError: Model sets swiglu_limit=10.0, but the explicitly requested
moe_backend='flashinfer_b12x' does not apply the SwiGLU clamp.
```

Unblocking *this* path needs the FlashInfer b12x kernel to implement the clamp for SILU —
kernel work, not vLLM plumbing.

### 3. ~~eugr's b12x SwiGLU patch no longer applies to current vLLM main~~ RETRACTED

**This finding was wrong and is retracted.** We hit
`expected one B12x activation support predicate source anchor, found 0` against a clone of
`eugr/spark-vllm-docker` pinned at `3ad5610` (2026-08-11). It was fixed the very next day in
[`21aa9948`](https://github.com/eugr/spark-vllm-docker/commit/21aa9948), which adds an anchor
variant for the newer `_supports_activation` shape that includes `MoEActivation.GELU_TANH`.

Their patch tracks current `main` correctly. We did not re-check before writing it up.
Recorded rather than deleted, because "verify the upstream state before reporting a
downstream bug" is the actual lesson.

### 4. `b12x`'s FC2 tile upgrade cannot survive its own re-pin (revised twice)

**This section has been corrected twice and is the more useful for it.** It supersedes an
earlier claim that b12x "cannot serve DeepSeek-V4 at all" (that was `flashinfer_b12x` —
finding 2), and a **maintainer review of our upstream report then corrected three claims
we made here**. Both the original and the corrections are kept below.

The `b12x` MoE backend on the MXFP4 path reaches the model fine — it clears the clamp
guard entirely — and then dies:

```
ValueError: force_tile_config fc2 tile (tile_k=32, tile_n=512)
            does not fit problem N/K=4096/1024 at moe_block_size=8
```

**Root cause.** In `b12x/moe/_shared/kernels/w4a16/kernel.py`, an opportunistic "ultra"
path widens the FC2 output tile from `(64, 256)` to `(32, 512)` to wave-balance FC2
against FC1. Those tiles then make a round trip the selector doesn't account for:
`:13308-13311` packs them into `launch_tail`; `:11006-11009` re-pins them as
`force_tile_config` across the Torch custom-op boundary ("the custom-op boundary cannot
carry the compiled launch object"); the second compile re-validates them with
`_candidate_tile_fits()`, which rejects them, and raises.

**The defect stated precisely: the auto-selector emits a tile config that cannot survive
its own re-pin.** Any tile the auto path can select must pass the forced re-pin validator;
this one doesn't.

#### Three things we got wrong

**It is not a shared-memory problem.** `_candidate_tile_fits` rejects the tile at
`kernel.py:465`, before any footprint math:

```python
if int(tile_n) < 64 or int(tile_k) < 64 or int(cta_threads) < 128:
    return False
smem_bytes = _shared_memory_footprint(...)   # not reached for tile_k=32
```

`tile_k=32` fails that floor unconditionally. Our hardware table below is correct output
and the wrong explanation — those rows would read `False` on a B200 with 227 KB too.

| problem K | `(32,512)` ultra | `(64,256)` default |
|---|---|---|
| 1024 | ✗ rejected | ✓ fits |
| 2048 | ✗ rejected | ✓ fits |

We used the K=2048 row to rule out TP sharding, which it does. We then read it as proving
a shared-memory-class problem, which it does not — it is equally explained by the floor,
and the floor is what the code actually does. **The GB10-vs-B200 framing (101,376 B vs
~227 KB) was wrong and is withdrawn.** GB10 is where we ran it, not why it broke.

**It is not an oversight.** We claimed the FC2 branch forgot what the neighbouring FC1
branch remembers. `kernel.py:9909-9911` says otherwise, in as many words: *"A 512-wide N
tile needs tile_k=32 to keep cta_threads=256 (512*32/64); that is below the generic
tile_k>=64 fits-floor, so we validate the footprint directly here."* The bypass is
deliberate. The FC1 branch we held up as the counter-example uses `tile_k=64`, which
clears the floor — which is why it can use the generic predicate and FC2 cannot.

**Our patch does not fix it — it disables the optimization.** Given that floor, adding
`_candidate_tile_fits(..., tile_k=32, ...)` to the upgrade's guard returns `False` on
every device, every shape, every scale format. There is no "when it won't fit": it never
fits, so the ultra path becomes dead code. We described it as falling back to `(64,256)`
"when the wide tile won't fit"; that conditional does not exist.

#### What we actually ship, and what should land upstream

[`deploy/patches/b12x_ultra_tile_fit.py`](deploy/patches/b12x_ultra_tile_fit.py) is a
**local unblock that disables the ultra FC2 tile**, and is labelled as such. It gets b12x
serving on GB10. It is not the right upstream fix and we have withdrawn it as a proposal.

The right fix admits the tile rather than removing the upgrade, which is what both
upstream PRs do — [#146](https://github.com/local-inference-lab/b12x/pull/146) whitelists
the exact `(tile_k=32, tile_n=512, cta_threads=256)` geometry in the predicate, and
[#39](https://github.com/local-inference-lab/b12x/pull/39) adds a dedicated predicate plus
an end-to-end test over the auto-select → re-pin round trip. If you are patching b12x
yourself, prefer either of those to ours.

**Result: b12x serves DeepSeek-V4 on GB10** with the ultra path disabled, and output is
verified correct by inspection, not just by the absence of a crash. The throughput gain
in the next section comes from **using b12x instead of DeepGEMM**, not from this patch —
see the note under the table.

---

## Measured performance

Using **MiaAI-Lab's own `benchmark-0731.py`**, unmodified, so the comparison is
like-for-like. Single-stream (c1) decode, tok/s. **All figures below are medians of n=5
full runs**, with per-run samples in `bench/results/`.

| prompt | ours, DEEPGEMM backend | ours, **b12x backend** (patched) | gain | MiaAI-Lab published |
|---|---|---|---|---|
| 256 | 53.92 | **65.67** | **+21.8%** | **75.4** |
| 2048 | 51.52 | **63.70** | **+23.6%** | **68.8** |
| 8192 | 53.59 | **65.04** | **+21.4%** | — |

**What this measures.** The gain column is **DEEPGEMM vs b12x — a backend switch**. It is
not the effect of our patch, and earlier versions of this README implied it was. The
patch's own contribution cannot be measured from here: there is no unpatched-b12x row,
because unpatched b12x fails init in this configuration. Per finding 4 the patch disables
an optimization, so its own contribution is at best zero relative to a working ultra path.

Both of our columns are n=5 medians on the same hardware, same script, same
session. Spread: baseline sd 1.2–3.8; b12x sd 2.0–6.4 (the p2048 spread is
inflated by one cold first run of 48.4 against a 63–66 cluster). Raw per-run values are in
[`bench/results/n5/retune_sweep.log`](bench/results/n5/retune_sweep.log).

Best configuration: `--moe-backend b12x` on the patched image, `fp8_ds_mla` KV,
`--gpu-memory-utilization 0.88`, k=5. KV pool 1,389,891 tokens.

**We remain ~7.4% behind MiaAI-Lab at p2048 and ~13% at p256.** Moving to the b12x
backend closes most of the gap that existed before it, and does not close all of it.
Their remaining advantage is not mysterious — see below.

### Why we are still behind: `nvfp4_ds_mla` is unreachable on this lineage

MiaAI-Lab run `nvfp4_ds_mla` KV cache. We cannot, and the reason is structural
rather than a tuning miss:

```
AssertionError: DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache,
                got nvfp4_ds_mla
```

The layout is fixed by the model's attention class
(`models/deepseek_v4/attention.py`, `use_fp8_ds_mla_layout: ClassVar[bool] = True`),
which `get_attn_backend()` returns directly. `VLLM_ATTENTION_BACKEND=B12X_MLA_SPARSE`
does **not** override it, even though `b12x_mla_sparse.py` itself accepts
`nvfp4_ds_mla`. So on upstream-lineage vLLM this KV dtype is not selectable for
DeepSeek-V4 without patching backend selection.

We also tried the NVFP4 checkpoint (`utarn/DeepSeek-V4-Flash-0731-NVFP4`). It
reaches the loader and fails there:

```
parameter.py:176, in load_merged_column_weight
assert param_data.shape == loaded_weight.shape
```

A smaller KV element on a bandwidth-bound decode is a plausible part of the
residual gap, but we have not measured it and are not claiming a magnitude.

### Why n=5 matters here

Our first b12x measurement was a single sample and read **65.5 tok/s** at p2048. Five runs
put the median at **60.4** (sd 1.4). The single sample was near the top of the range and
would have overstated the gain by ~8%. Earlier tables in this repo's history are n=1 and
should be read as indicative only.

Note that **60.4 is a different arm from the 63.70 in the table above**: it is the
`rep_b12x` set at default `--gpu-memory-utilization`, while the headline table is the
`b12x_hi` arm at `0.88` (`bench/results/n5/`, both n=5). We are not quoting the better of
two runs of the same configuration — but the two arms should not be read as one, and an
earlier version of this section did not distinguish them.

### Where tuning helped: KV capacity

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

Credit: these came from the eugr / MiaAI-Lab / drowzeys community docs and the NVIDIA
forum thread; we verified them, we didn't invent them.

### Speculative decoding: acceptance is workload-dependent

Worth stating because we briefly got it wrong. On **our warm-up prompts** (short, highly
predictable) DSpark reaches ~80% draft acceptance and ~5.0 accepted length at k=5. On the
**benchmark workload** the same server reports ~45% and ~3.2, with per-position acceptance
decaying `0.81, 0.56, 0.37, 0.26, 0.17`.

Quoting the first as if it were the second suggested acceptance had saturated and that a
larger `k` would pay off. It hasn't, and it doesn't — the 5th draft position lands under
20% of the time, which matches our earlier measurement that k=7 *reduced* throughput.
**Always read acceptance metrics from the window matching the workload you're quoting.**

---

## What did NOT work

| Attempt | Outcome |
|---|---|
| `--moe-backend flashinfer_b12x` | SwiGLU clamp guard (finding 2) |
| `--moe-backend b12x`, unpatched | tile-fit `ValueError` (finding 4) — **unblocked locally by disabling the ultra FC2 tile**; upstream fix is [#39](https://github.com/local-inference-lab/b12x/pull/39)/[#146](https://github.com/local-inference-lab/b12x/pull/146) |
| `--moe-backend deep_gemm` | loads, dies in kernel: `Unknown SF transformation` |
| `--moe-backend triton` | "kernel does not support current device" |
| `--moe-backend marlin` (MXFP4) | selected, loads, **same DeepGEMM error** → proved MoE was never the fault |
| `VLLM_USE_DEEP_GEMM=0` | got past load, then `hyperconnection.hpp: Unsupported architecture` |
| NVFP4 checkpoint + auto → MARLIN | weight-shape `AssertionError` in the loader |
| `--kv-cache-dtype nvfp4_ds_mla` (MXFP4 **or** NVFP4 weights) | `DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache` — layout is fixed by the model's attention class, not by the checkpoint |
| `VLLM_ATTENTION_BACKEND=B12X_MLA_SPARSE` to reach that layout | ignored — `get_attn_backend()` returns a hardcoded `backend_cls` |
| NVFP4 checkpoint + `b12x` + clamp-allowlist patch | backend **selected** (`Using 'B12X' NvFp4 MoE backend`), then loader `assert param_data.shape == loaded_weight.shape` |
| `num_speculative_tokens=7` | accept length 2.88 → 3.12, but throughput **regressed** — reverted to 5 |
| `VLLM_USE_BREAKABLE_CUDAGRAPH=0` | **no decode gain** — the published +28.6% was measured against Anemll's opposite default; vLLM-lineage default is already `0` |

---

## Measurement notes — please read before quoting any tok/s

See [`docs/MEASUREMENT-NOTES.md`](docs/MEASUREMENT-NOTES.md). Short version: published
figures for this model vary by more than 3× because harnesses measure different things.

- **Decode-with-TTFT vs decode-only** differ by ~1.6× on identical runs.
- **Prefix caching**: a deterministic prompt nonce made prefill read ~1500 tok/s; a random
  nonce on the same config read ~200 tok/s. Both "real", totally different quantities.
- **Single samples mislead** — see the n=1 vs n=5 gap above.
- **Cold start pollutes the first cell** — our first p256 case reported 6.79 s TTFT and
  41 tok/s prefill against 1.23 s / 1682 tok/s at p2048. Warm before measuring.
- **Spec decoding delivers ~12 tokens per SSE chunk**, so chunk-timing methods (including
  our own `bench/bench_fixed.py`) produce unstable decode numbers. **That harness is
  published as a cautionary artifact, not a recommendation** — it produced 117, then 24.5,
  then 503 tok/s for near-identical configs.

---

## Reproducing

```bash
# 1. get a b12x-capable image (eugr publishes one prebuilt)
docker pull eugr/spark-vllm-b12x:latest
docker tag  eugr/spark-vllm-b12x:latest vllm-node-b12x:latest

# 2. disable the unsafe ultra FC2 tile (finding 4) — required on GB10 until #39/#146 land
cat > Dockerfile <<'EOF'
FROM vllm-node-b12x:latest
COPY b12x_ultra_tile_fit.py /tmp/
RUN python3 /tmp/b12x_ultra_tile_fit.py
EOF
cp deploy/patches/b12x_ultra_tile_fit.py .
docker build -t vllm-node-b12x-fix:latest .
#    build this on BOTH nodes — the layer is tiny, far faster than shipping the image

# 3. configure
cp deploy/.env.example .env      # fill in your RoCE IPs, ranks, HF cache path
#    worker sets NODE_RANK=1 and HEADLESS=1; head sets NODE_RANK=0, HEADLESS=

# 4. boot WORKER FIRST, then head
DS4_IMAGE=vllm-node-b12x-fix:latest MOE_BACKEND=b12x \
  docker compose -f deploy/docker-compose.ds4.yml --env-file .env up -d

# 5. verify
bash deploy/scripts/smoke_ds4.sh
```

`--headless` on the worker is **required** — the CLI help says it's for "multi-node data
parallel", which is misleading; `entrypoints/cli/serve.py:213` reads *"Run headless
workers (for multi-node PP/TP)"*.

---

## Environment

- 2× NVIDIA DGX Spark, GB10 Blackwell **sm_121**, 128 GB unified LPDDR5X each
  (101,376 B opt-in shared memory per block, 48 SMs — the SM count gates the ultra branch
  in finding 4; the shared-memory figure turned out **not** to be why it fails)
- 200G ConnectX-7 QSFP56 DAC, RoCEv2 — measured 111 Gb/s `ib_write_bw`, ~13.6 GB/s NCCL all-reduce
- DGX OS (Ubuntu 24.04), kernel 6.17-nvidia, CUDA 13.0, driver 580.173.02
- vLLM `main` @ `0.27.2rc1.dev48+g64ca614fe`, torch 2.13.0+cu130, FlashInfer 0.6.18, Triton 3.7.1
- `TORCH_CUDA_ARCH_LIST=12.1a`, NCCL gencode `sm_121`, NCCL 2.30.7

## Credits

**Essentially all of the tuning knowledge here is the community's, not ours.** This repo
is a validation-and-correction layer on top of other people's work, and it would not
exist without any of the following:

- **[MiaAI-Lab](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)** —
  the original 2× DGX Spark recipe for this model, and `benchmark-0731.py`, which is the
  harness every number in this repo was produced with. We ran their script unmodified
  precisely so our figures could be checked against theirs. Their published numbers set
  the bar we measured ourselves against, and their configuration choices — b12x for MoE,
  `nvfp4_ds_mla` for KV — are what pointed us at the b12x path in the first place.
  **Their numbers remain ahead of ours** at the time of writing; see the results table.
- **[eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)** — the sm_12x
  build harness (`preserve_sm12x_target`, `sm120_cooperative_topk`) that makes vLLM
  buildable for GB10 at all, and the prebuilt `eugr/spark-vllm-b12x` image that finding 4
  is based on. Our patch is a small fix *inside* a library their image packages; the hard
  part — getting a working b12x image onto this architecture — was theirs.
- **[drowzeys ("Keys")](https://github.com/drowzeys)** and **tonyd2wild** — Spark
  deployment notes and the memory/KV flag set we verified in the tuning table.
- **[The NVIDIA developer forum thread](https://forums.developer.nvidia.com/t/instructions-for-running-deepseek-v4-flash-with-dspark-using-eugrs-repo/376220)**
  — collective debugging that saved us considerable time.
- **[yichengj0](https://github.com/vllm-project/vllm/pull/47392)** for the b12x activation
  plumbing PR, and the author of
  [vllm#51959](https://github.com/vllm-project/vllm/pull/51959) /
  [#52035](https://github.com/vllm-project/vllm/pull/52035) for the DeepGEMM sm120 repin —
  the change that makes this model run on sm_121 at all.
- **DeepSeek** for the weights, and the **vLLM**, **DeepGEMM** and **FlashInfer** teams.

If we have mis-stated anyone's work here, please open an issue — we have already had to
correct this repo three times, and would rather be corrected again than leave it wrong.

## Licence

**MIT** for the packaging, scripts and documentation in this repository — see
[`LICENSE`](LICENSE).

This is a third-party contribution. It is **not** an official DeepSeek, vLLM, NVIDIA or
FlashInfer release, and no endorsement is implied. **Model weights are not included here.**

The two patch scripts in [`deploy/patches/`](deploy/patches/) modify third-party code
*inside a locally built container image*; no upstream source is vendored into this
repository. vLLM is Apache-2.0; DeepGEMM, FlashInfer, PyTorch and Triton retain their own
licences; model weights retain theirs.

`b12x` is **Apache-2.0** ([local-inference-lab/b12x](https://github.com/local-inference-lab/b12x)).
We do not redistribute it — it reaches this stack only as a component of the third-party
`eugr/spark-vllm-b12x` image, and our patch modifies it inside a locally built image.

Full component-by-component attribution, with the exact versions and revisions under test:
**[`NOTICE`](NOTICE)** and **[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)**.
