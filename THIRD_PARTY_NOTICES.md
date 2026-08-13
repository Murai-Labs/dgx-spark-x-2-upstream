# Third-party notices

This repository contains no third-party source code in bulk. It contains
deployment configuration, benchmark outputs, documentation, and **two small
patch scripts** that modify third-party code *at container build time*:

| Patch | Modifies | Upstream licence of the modified file |
|---|---|---|
| [`deploy/patches/b12x_ultra_tile_fit.py`](deploy/patches/b12x_ultra_tile_fit.py) | `b12x` (`moe/_shared/kernels/w4a16/kernel.py`) | see **b12x** below |
| [`deploy/patches/nvfp4_clamp_allowlist.py`](deploy/patches/nvfp4_clamp_allowlist.py) | vLLM (`model_executor/layers/fused_moe/oracle/nvfp4.py`) | Apache-2.0 |

Neither patch vendors upstream source into this repository — each is a script
that edits an installed package inside a container image. The patched projects
retain their own licences.

---

## DeepSeek AI

- Model: `deepseek-ai/DeepSeek-V4-Flash-0731`
- Revision under test: `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- Also referenced: `utarn/DeepSeek-V4-Flash-0731-NVFP4` @ `ca20bac907e9711b759fcebd214a2e58ba7bd857`
  (a third-party NVFP4 requantisation; **it did not load** on this stack — see README)
- Model weights and tokenizer are **not** included in this repository
- Governed by DeepSeek's published model licence/terms for that revision

## vLLM

- https://github.com/vllm-project/vllm
- Licence: **Apache License 2.0**
- Version under test: `0.27.2rc1.dev48+g64ca614fe` (upstream `main`, post-#52035)
- Relevant upstream changes validated here:
  [#51959](https://github.com/vllm-project/vllm/pull/51959) (branch pin `a6b593d`),
  [#52035](https://github.com/vllm-project/vllm/pull/52035) (merged, pin `8b1392b9`),
  [#47392](https://github.com/vllm-project/vllm/pull/47392) (b12x activation plumbing)
- `deploy/patches/nvfp4_clamp_allowlist.py` modifies an Apache-2.0 licensed vLLM
  source file at image build time; the modification is offered back upstream
  rather than redistributed

## DeepGEMM

- https://github.com/deepseek-ai/DeepGEMM — pins `a6b593d` (validated) and
  `8b1392b9` (merged upstream pin, also validated on sm_121)
- https://github.com/vllm-project/DeepGEMM — pin `e21c821` (the pin that
  **lacks** `sm120_*` kernels; documented here as the failure mode)
- Licence: upstream DeepGEMM terms
- [DeepGEMM#403](https://github.com/deepseek-ai/DeepGEMM/pull/403) repro reproduced on sm_121

## FlashInfer

- https://github.com/flashinfer-ai/flashinfer
- Version under test: `0.6.18`
- Sparse MLA and `flashinfer_b12x` MoE paths exercised at runtime
- Licence/copyright: upstream FlashInfer authors

## b12x

- https://github.com/local-inference-lab/b12x
- PyPI package `b12x`, version **1.2.3**, author **Luke Alonso**
- Licence: **Apache License 2.0**
- SM120/SM121 CuTe DSL kernel library for NVFP4 GEMM and MoE, targeting DGX
  Spark and Blackwell RTX parts
- Note: the *installed distribution metadata* carries no homepage, project URL
  or licence field, which is why earlier revisions of this file could not state
  the licence. The repository does — it is Apache-2.0.
- `deploy/patches/b12x_ultra_tile_fit.py` modifies one function in this package
  **inside a locally built container image**. No b12x source is copied into this
  repository, and it reaches our stack only as a component of the third-party
  container image `eugr/spark-vllm-b12x`.
- The underlying defect is filed upstream as
  [local-inference-lab/b12x#182](https://github.com/local-inference-lab/b12x/issues/182).

## eugr/spark-vllm-docker

- https://github.com/eugr/spark-vllm-docker
- Provides the sm_12x build harness (`preserve_sm12x_target`,
  `sm120_cooperative_topk`) and the prebuilt image `eugr/spark-vllm-b12x`, which
  is the base image for everything measured here
- Licence: upstream repository terms

## MiaAI-Lab

- https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark
- Source of the original 2x DGX Spark recipe and of `benchmark-0731.py`
- **Every tok/s figure in this repository was produced by their script, used
  unmodified**, so that our numbers are directly comparable to theirs
- Licence: upstream repository terms

## NVIDIA

- DGX Spark / GB10 platform, CUDA 13.0, driver 580.173.02, NCCL 2.30.7
- Trademarks and platform documentation remain NVIDIA property
- No NVIDIA endorsement implied

## PyTorch

- `2.13.0+cu130`
- BSD-style licence (upstream PyTorch)

## Triton

- `3.7.1`, MIT licence (upstream Triton)

---

## Corrections

If any attribution, licence statement or version above is wrong, please open an
issue. This repository has already been corrected several times — see the
retracted and revised findings in `README.md` — and we would rather be corrected
again than leave an error standing.
