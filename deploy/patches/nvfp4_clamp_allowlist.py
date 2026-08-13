#!/usr/bin/env python3
"""
Add `NvFp4MoeBackend.B12X` to vLLM's `NVFP4_BACKENDS_WITH_CLAMP`.

WHY
---
`vllm/model_executor/layers/fused_moe/oracle/nvfp4.py` refuses any MoE backend
that is not known to apply the SwiGLU clamp when the model sets `swiglu_limit`:

    NVFP4_BACKENDS_WITH_CLAMP = {
        NvFp4MoeBackend.FLASHINFER_TRTLLM,
        NvFp4MoeBackend.FLASHINFER_CUTLASS,
        NvFp4MoeBackend.MARLIN,
    }

    # added only for SwiGLU-OAI:
    if (config.activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE
            and has_flashinfer_b12x_moe_activation()):
        NVFP4_BACKENDS_WITH_CLAMP.add(NvFp4MoeBackend.FLASHINFER_B12X)

`B12X` -- which is a *distinct* backend from `FLASHINFER_B12X` -- is never added
at all. DeepSeek-V4-Flash declares `hidden_act="silu"` with `swiglu_limit=10.0`,
so requesting `--moe-backend b12x` on NVFP4 weights is rejected.

That omission appears to be over-conservative rather than protective. b12x
normalises `swiglu_limit` for *every* gated activation, SILU included:

    >>> from b12x.moe._shared.kernels.activations import (
    ...     normalize_swiglu_limit_for_activation as f)
    >>> f("silu", 10.0)
    10.0
    >>> f("swigluoai_uninterleave", 10.0)
    10.0
    >>> f("gelu_tanh", 10.0)
    ValueError: unsupported activation 'gelu_tanh'

and the MXFP4 oracle already threads the same value through to b12x as
`gemm1_clamp_limit=swiglu_limit` in eight call sites.

STATUS -- READ THIS BEFORE RELYING ON IT
----------------------------------------
Applying this patch **does** change backend selection as intended: vLLM then
logs `Using 'B12X' NvFp4 MoE backend` instead of raising.

It has **not** been validated end to end. On our stack the NVFP4 checkpoint
(`utarn/DeepSeek-V4-Flash-0731-NVFP4`, modelopt `W4A16_NVFP4`) fails later in
the weight loader:

    parameter.py:176, in load_merged_column_weight
    assert param_data.shape == loaded_weight.shape

so no output was ever generated through this path, and we therefore have **not**
confirmed numerically that the clamp is applied in the NVFP4 kernel arithmetic.
We verified the plumbing, not the maths.

Do not treat this as a validated correctness fix. It is a documented lead.

Applies to vLLM main @ 0.27.2rc1.dev48+g64ca614fe. Idempotent.
"""

import sys

P = (
    "/usr/local/lib/python3.12/dist-packages/vllm/"
    "model_executor/layers/fused_moe/oracle/nvfp4.py"
)

OLD = """    NVFP4_BACKENDS_WITH_CLAMP = {
        NvFp4MoeBackend.FLASHINFER_TRTLLM,
        NvFp4MoeBackend.FLASHINFER_CUTLASS,
        NvFp4MoeBackend.MARLIN,
    }"""

NEW = """    NVFP4_BACKENDS_WITH_CLAMP = {
        NvFp4MoeBackend.FLASHINFER_TRTLLM,
        NvFp4MoeBackend.FLASHINFER_CUTLASS,
        NvFp4MoeBackend.MARLIN,
        # b12x normalises swiglu_limit for every gated activation, SILU
        # included -- not only SWIGLUOAI_UNINTERLEAVE. See
        # b12x/moe/_shared/kernels/activations.py::
        #   normalize_swiglu_limit_for_activation
        NvFp4MoeBackend.B12X,
    }"""


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else P
    try:
        src = open(path).read()
    except OSError as exc:
        print(f"cannot read {path}: {exc}", file=sys.stderr)
        return 2

    if NEW in src:
        print("already patched — no change")
        return 0
    if OLD not in src:
        print("anchor not found — the nvfp4 oracle has changed shape", file=sys.stderr)
        return 1

    open(path, "w").write(src.replace(OLD, NEW, 1))
    print(f"patched {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
