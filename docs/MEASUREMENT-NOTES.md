# Measurement notes

Published tok/s figures for DeepSeek-V4-Flash on DGX Spark vary by more than 3× across
sources. Most of that spread is **methodology, not hardware**. These are the traps we hit,
documented so the numbers in this repo can be interpreted — and so the next person doesn't
repeat them.

We hit every one of these ourselves. The list is a record of our own errors as much as
anything.

---

## 1. "Decode tok/s" means at least three different things

For one request generating N tokens:

| Quantity | Formula | Our measured value (same run) |
|---|---|---|
| End-to-end | N ÷ (TTFT + decode) | ~24 tok/s |
| "Decode" incl. TTFT (most harnesses) | N ÷ elapsed | ~52–57 tok/s |
| Pure decode | N ÷ decode-window | ~85 tok/s |

All three are defensible. Quoting one against another is not.

**This cost us hours.** We compared our ~52 (harness figure) against a published 82.4
(effectively pure decode, captured at 168 ms TTFT) and concluded we had a large deficit.
Measured the same way, the gap largely disappeared. Always establish what the *reference*
number measured before treating a difference as real.

## 2. Prefix caching silently inflates prefill

`benchmark-0731.py` builds its prompt with a deterministic nonce
(`f"p{prompt_tokens}-c{concurrency}-r{index}"`), so repeated invocations can be served
from the prefix cache.

On identical config we measured:

- deterministic nonce (cache hits): prefill **~1500 tok/s**
- random nonce (cold): prefill **~200 tok/s**

Neither is wrong. They answer different questions ("warm re-read" vs "cold prefill").
Any published prefill number without a stated caching policy is uninterpretable.

## 3. Output-length variance destroys aggregate figures

Aggregate is usually `total_tokens ÷ wall_clock`, so it is dominated by the slowest
request. If the model emits different token counts per run — and it will, unless you pin
`max_tokens` + `ignore_eos` — the aggregate swings wildly.

A third party reported **52.21 vs 23.03 tok/s on back-to-back identical inputs**, purely
because the model emitted 1,980 vs 837 tokens.

We saw the same shape: at `max_num_seqs=12`, p2048/c4 measured *worse* than at
`max_num_seqs=8` — physically impossible for a 4-request workload — because single-sample
cells were straggler-dominated.

## 4. Speculative decoding breaks chunk-timing harnesses

DSpark emits **~12 tokens per SSE chunk** (200 tokens arrived in 16 chunks in our runs).
This defeats two common approaches:

- **Counting SSE deltas as tokens** undercounts by ~4×. A third party measured
  *"14.7 vs 60.1 tok/s on the identical request"* this way. Always take
  `usage.completion_tokens` from the server (`stream_options: {"include_usage": true}`).
- **Timing first→last chunk** to get a decode window is unstable, because the tail can
  arrive in one burst. Our own `bench/bench_fixed.py` reported **117.3, then 24.5, then
  503.1 tok/s** for near-identical configs. It is included in this repo as a cautionary
  artifact, **not a recommendation**.

Our first version of that harness was worse still: it computed the window as
`total − TTFT`, which collapses toward zero when the tail bursts, yielding decode rates of
**1.0×10¹¹ tok/s**. Guard your denominators.

## 5. Cold measurements run ~30% low

Acceptance climbs as kernels JIT. One community measurement: acceptance
`31% → 34% → 65% → 67%` and mean accept length `2.6 → 4.6` over the first few requests,
with throughput `58.5 → 83.3 tok/s`. It also decays after ~30 minutes idle.

**Warm the endpoint with a few hundred tokens of real traffic before recording anything.**
We measured 19.9 → 31.6 → 31.8 tok/s across three consecutive warm-up requests.

## 6. Concurrent I/O contaminates GPU benchmarks

We ran a benchmark while a 173 GB model download was writing at ~63 MB/s. TTFT went from
1.43 s to **9.61 s** on the same config. Obvious in hindsight; easy to do by accident when
staging the next experiment in parallel.

---

## 7. A single sample will flatter you, and it flattered us

We measured the patched b12x backend once and read **65.5 tok/s** at p2048. Five full runs
of the same script against the same unchanged server gave:

```
p2048  median=60.43  mean=60.95  min=59.74  max=63.66  sd=1.38  n=5
       samples: 60.4, 63.7, 59.7, 60.4, 60.5
```

The single sample was near the top of the observed range and overstated the result by
~8%. Note that the spread is *not* wide — sd 1.4 on a 60 tok/s figure is tight. The
problem isn't instability; it's that one draw from a tight distribution still lands
somewhere, and "somewhere" was the high end.

This is the cheapest error to avoid on this list and the one we made most recently. Five
runs cost about fifteen minutes.

## 8. Acceptance metrics are per-workload — match the window to the claim

`SpecDecoding metrics` lines are emitted continuously and reflect **whatever traffic was
running in that window**. On the same server, same config, minutes apart:

| window | traffic | mean accept length | avg draft acceptance |
|---|---|---|---|
| 18:55:15–35 | short warm-up prompts | 4.78 → 5.01 | 75.6 → **80.2%** |
| 18:58:45–59:05 | the actual benchmark | 3.17–3.38 | **43–48%** |

Both are true statements about the server. Only one describes the workload whose tok/s
you are quoting.

We briefly read the 80.2% figure as evidence that acceptance had saturated at k=5 and
that raising `k` would therefore pay off. On the benchmark workload, per-position
acceptance decays `0.81, 0.56, 0.37, 0.26, 0.17` — the 5th draft position lands under 20%
of the time, so a larger `k` buys draft cost for almost nothing. That matches our direct
measurement that k=7 *reduced* throughput, which we would have contradicted on the
strength of a mis-scoped metric.

---

## What we'd recommend instead

For comparable numbers, report **all** of:

1. Which quantity (end-to-end / incl-TTFT / decode-only) and the formula.
2. `max_tokens`, whether `ignore_eos`/`min_tokens` were set, and observed min/max output length.
3. Prompt-nonce policy (cacheable or not) and whether prefix caching was enabled.
4. Warm-up performed.
5. Repeat count and spread — not a single sample. Report median, sd and n.
6. Token source: server `usage` vs client-side counting.
7. Concurrency, `max_num_seqs`, and whether other I/O was running.
8. For speculative decoding: the acceptance window used, and that it covers the
   benchmark traffic rather than warm-up or idle traffic.

The single most useful convention would be for the community to standardise on one
harness with fixed output length and a stated caching policy. Until then, cross-source
tok/s comparisons for this model should be treated as indicative at best.
