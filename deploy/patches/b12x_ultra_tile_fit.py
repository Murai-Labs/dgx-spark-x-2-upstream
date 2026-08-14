#!/usr/bin/env python3
"""
DISABLE b12x's "ultra" FC2 tile upgrade.

Read the name as "disable ultra FC2", not "make the ultra tile fit". This is a
local unblock, NOT the upstream fix -- see UPSTREAM below before using it.

WHY
---
`b12x/moe/_shared/kernels/w4a16/kernel.py` opportunistically widens the FC2
output tile from (tile_k=64, tile_n=256) to (tile_k=32, tile_n=512) to
wave-balance FC2 against FC1. It validates the footprint inline, deliberately
bypassing `_candidate_tile_fits()` -- the code says so at kernel.py:9909-9911:

    A 512-wide N tile needs tile_k=32 to keep cta_threads=256 (512*32/64);
    that is below the generic tile_k>=64 fits-floor, so we validate the
    footprint directly here.

The selected tiles then make a round trip the selector does not account for:
kernel.py:13308-13311 packs them into `launch_tail`, and :11006-11009 re-pins
them as `force_tile_config` across the Torch custom-op boundary ("the custom-op
boundary cannot carry the compiled launch object"). The second compile
re-validates them with `_candidate_tile_fits()`, which does not have the
bypass, and engine init dies with

    ValueError: force_tile_config fc2 tile (tile_k=32, tile_n=512) does not fit
                problem N/K=4096/1024 at moe_block_size=8

So: the auto-selector emits a tile config that cannot survive its own re-pin.

NOT a shared-memory problem. `_candidate_tile_fits` returns False at
kernel.py:465 -- `tile_k < 64` -- before computing any footprint at all:

    if int(tile_n) < 64 or int(tile_k) < 64 or int(cta_threads) < 128:
        return False

The tile is rejected on every device at every shared-memory budget, not just
under GB10's 101,376 B. Measured on GB10:

    K=1024  tile_k=32 tile_n=512  -> fits=False
    K=1024  tile_k=64 tile_n=256  -> fits=True
    K=2048  tile_k=32 tile_n=512  -> fits=False
    K=2048  tile_k=64 tile_n=256  -> fits=True

Those rows do rule out tensor-parallel sharding as the cause. They do NOT show
a shared-memory-class issue -- they are equally explained by the tile_k floor,
which is what the code actually does. An earlier version of this file blamed
the shared-memory budget; that was wrong. GB10 is where we hit it, not why it
fails.

WHAT THIS PATCH ACTUALLY DOES
-----------------------------
It adds `_candidate_tile_fits(...)` to the upgrade's own guard. Because of the
tile_k floor above, that predicate is False for (32, 512) unconditionally --
so the effect is that the ultra branch NEVER fires and FC2 keeps the (64, 256)
default. That unblocks init, and costs whatever the wave-balance optimization
was worth on parts where it would have worked.

UPSTREAM
--------
Two open PRs fix this properly, by admitting the tile instead of removing the
upgrade. Prefer either to this patch:

  https://github.com/local-inference-lab/b12x/pull/146  (whitelists the exact
      tile_k=32 / tile_n=512 / cta_threads=256 geometry in the predicate)
  https://github.com/local-inference-lab/b12x/pull/39   (dedicated predicate
      plus an e2e test over the auto-select -> re-pin round trip)

We reported this as https://github.com/local-inference-lab/b12x/issues/182,
which is a duplicate of both. Check whether either has landed before applying
this.

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
