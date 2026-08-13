#!/usr/bin/env python3
"""
Make b12x's "ultra" FC2 tile upgrade validate against the same fit checker that
later judges it.

WHY
---
`b12x/moe/_shared/kernels/w4a16/kernel.py` opportunistically widens the FC2
output tile from (tile_k=64, tile_n=256) to (tile_k=32, tile_n=512). It gates
that upgrade on `_shared_memory_footprint(...) <= max_shared_mem - 512`.

The upgraded tiles are then passed *downstream* as an explicit
`force_tile_config` pin, where they are re-checked by `_candidate_tile_fits()` —
a stricter predicate. On parts with a small shared-memory budget the two
disagree: the upgrade fires, the pin is rejected, and engine init dies with

    ValueError: force_tile_config fc2 tile (tile_k=32, tile_n=512) does not fit
                problem N/K=4096/1024 at moe_block_size=8

NVIDIA GB10 (DGX Spark, sm_121) exposes 101,376 B of opt-in shared memory per
block, versus ~227 KB on B200-class parts. Measured on GB10:

    K=1024  tile_k=32 tile_n=512  -> fits=False
    K=1024  tile_k=64 tile_n=256  -> fits=True
    K=2048  tile_k=32 tile_n=512  -> fits=False
    K=2048  tile_k=64 tile_n=256  -> fits=True

So the wide tile never fits on this SM class, at any K — this is a
shared-memory-class issue, not an artifact of tensor-parallel sharding.

THE FIX
-------
Add `_candidate_tile_fits(...)` to the upgrade's own guard, so the upgrade only
fires when the resulting tile will survive the later check. When it would not,
selection falls through to the (64, 256) default, which fits.

This is conservative: it can only *prevent* an upgrade that was going to raise.
It cannot change any configuration that already worked.

Filed upstream as https://github.com/local-inference-lab/b12x/issues/182 --
check whether it is fixed there before applying this.

Applies cleanly to b12x 1.2.3 (as shipped in eugr/spark-vllm-b12x, and as on
local-inference-lab/b12x master as of 2026-08-13). Idempotent -- running twice
is a no-op (the anchor no longer matches).
"""

import sys

KERNEL = (
    "/usr/local/lib/python3.12/dist-packages/"
    "b12x/moe/_shared/kernels/w4a16/kernel.py"
)

OLD = """        if (
            int(intermediate_size) % ultra_fc2_tile_k == 0
            and ultra_smem <= int(max_shared_mem) - 512
        ):"""

NEW = """        if (
            int(intermediate_size) % ultra_fc2_tile_k == 0
            and ultra_smem <= int(max_shared_mem) - 512
            and _candidate_tile_fits(
                problem_n=int(hidden_size),
                problem_k=int(intermediate_size),
                cta_m_blocks=_covering_count(moe_block_size, 16),
                tile_n=512,
                tile_k=ultra_fc2_tile_k,
                cta_threads=256,
                max_shared_mem=int(max_shared_mem) - 512,
                scale_format=scale_format,
                weight_layout=weight_layout,
                weight_bits=weight_bits,
                allow_logical_tail=False,
            )
        ):"""


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else KERNEL
    try:
        src = open(path).read()
    except OSError as exc:
        print(f"cannot read {path}: {exc}", file=sys.stderr)
        return 2

    if NEW in src:
        print("already patched — no change")
        return 0
    if OLD not in src:
        print(
            "anchor not found — b12x has changed shape; re-check the ultra "
            "tile upgrade in w4a16/kernel.py before assuming this is still needed",
            file=sys.stderr,
        )
        return 1

    open(path, "w").write(src.replace(OLD, NEW, 1))
    print(f"patched {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
