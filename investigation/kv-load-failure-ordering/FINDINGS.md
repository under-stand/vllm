# KV load failure ordering probe

Investigation only. No production change. No pull request.

Follow-up design note: `DESIGN.md`. Upstream main moved from `418af607` to `179ebd9896` later the same day; none of those 9 commits touch the scheduler or KV-cache manager, so the matrix below was not rerun. The block-reuse probe in `out/aba.json` was added against `418af607`.

## SHAs

| Ref | SHA | Subject |
| --- | --- | --- |
| upstream `main` | `418af607309d819bf091229bcc3d8ca8e5ce007f` | 2026-11-23 `[XPU][CI] Expand more entrypoints in Intel GPU CI (#68354)` |
| PR [#69299](https://github.com/vllm-project/vllm/pull/69299) head | `ecbebf9f7db7988532fe1164bb3dd9903080188c` | `[Bugfix][KV Connector] Evict failed synchronous KV loads from prefix cache` |
| PR base | `edccd1e39ede897504babb9a50aa8bba26089bbb` | not an ancestor of current main |

`418af607` does not contain `ecbebf9f`. Local fork `main` was `eafe6af276` and was 1134 commits behind upstream; probes ran on the upstream SHAs above, not on that fork commit.

## Environment

CPU-only scheduler integration. Real `Scheduler` / `KVCacheManager`. Connector match and `request_finished` are mocked the same way as `tests/v1/kv_connector/unit/test_invalid_blocks_correctness.py`. No GPU, no model weights, no user machine.

- Python 3.12.3, torch `2.14.0+cpu`, triton-cpu `3.8.0+cpu`
- `VLLM_TARGET_DEVICE=cpu`, `HF_HUB_OFFLINE=1` after the opt-125m config snapshot was cached
- `vllm._C` is not built. Import logs a warning and continues. This is an environment limit, not a scheduler failure.
- Package metadata `vllm==0.0.0+cpu` was installed in the venv so `vllm_version_matches_substr("cpu")` enables the Triton CPU backend. Without that, `CpuPlatform.check_and_update_config` raises `vLLM on x86 CPUs requires Triton (triton-cpu)`.
- `transformers==5.23.0` was what the resolver installed. Scheduler construction still succeeded.

## Command

```bash
export VLLM_TARGET_DEVICE=cpu VLLM_ENABLE_V1_MULTIPROCESSING=0 HF_HUB_OFFLINE=1
export VLLM_LOGGING_LEVEL=ERROR
PROBE_REF=main PROBE_SHA=418af607309d819bf091229bcc3d8ca8e5ce007f \
  .venv/bin/python investigation/kv-load-failure-ordering/probe_orphan_invalid_blocks.py
# exit 0. stdout: out/main.stdout.txt
```

PR head, from a detached worktree, with `PYTHONPATH` pointing at that tree (a script path does not put the repo root on `sys.path`; the venv `.pth` would otherwise import `/workspace`):

```bash
cd /tmp/vllm-pr69299
PYTHONPATH=/tmp/vllm-pr69299 \
PROBE_REF=pr69299 PROBE_SHA=ecbebf9f7db7988532fe1164bb3dd9903080188c \
PROBE_OUT=/workspace/investigation/kv-load-failure-ordering/out \
  /workspace/.venv/bin/python investigation/kv-load-failure-ordering/probe_orphan_invalid_blocks.py
# stdout first line: scheduler_module /tmp/vllm-pr69299/vllm/v1/core/sched/scheduler.py
```

Both runs exited 0. Controls passed, so the harness can see hits, evictions, and aborts. Raw JSON: `out/main.json`, `out/pr69299.json`.

## Setup shared by the rows

- Prompt: 8 blocks, block size 16 (128 tokens). Sync connector grant: 7 blocks (112 tokens), `load_async=False`.
- Reported invalid block: index 3.
- Invariant used for orphan rows: a later same-prefix lookup may hit at most the valid prefix, 48 tokens. A full local hit is 112 (the last prompt block is not reusable).
- `H` / `-` is `block_hash is not None` on the owner's 8 block ids, after the failure report.
- `reuse` is `get_computed_blocks` unless the row schedules a new request (abort / success / local control).
- One external grant per request. Resume does not get a second remote match, so a resume hit is local.
- Batch budget is 4096 so one `schedule()` admits the whole sync load. The helper default of 64 would chunk a 128-token prompt.

## Matrix

`held` = failed suffix is not reusable. `violated` = lookup or resume still hits past 48. Same verdict on sync and async **scheduler class** unless noted.

| Scenario | main `418af607` | #69299 `ecbebf9f` |
| --- | --- | --- |
| Control: local prefix, no external load | hit 112, `control_ok` | same |
| Control: visible owner, policy `fail` | `HHH-----`, reuse 48, `FINISHED_ERROR`, `control_ok` | same |
| Control: success then abort | reuse 112, `control_ok` (sync and async class) | same |
| Live owner, `recompute`, no abort | `HHHHHHHH`, reuse 112, owner `RUNNING` computed 48, `suffix_still_cached` | `HHH-----`, reuse 48, owner still `RUNNING` computed 48, `suffix_evicted` |
| Failure report, then abort, `fail` | `HHH-----`, reuse 48, `FINISHED_ERROR`, `held` | `held` |
| Failure report, then abort, `recompute` | `HHHHHHHH`, reuse 112, `FINISHED_ABORTED`, `violated` | `HHH-----`, reuse 48, `held` |
| Abort, then failure report, both policies | `HHHHHHHH`, reuse 112, `FINISHED_ABORTED`, `violated` | `violated` (unchanged) |
| `_preempt_request`, then failure, both policies | hash stays `HHHHHHHH`, lookup 112, resume 128, status stays `PREEMPTED`, `violated` | `violated` (unchanged) |
| `schedule()` again on a full 9-block pool, producer, both policies | preempted, hash `HHHHHHHH`, lookup/resume 112, `violated` | `violated` (unchanged) |
| Same pressure, consumer + async class (`defer_block_free=True`) | owner stays `RUNNING`, `not_reached` | `not_reached` |
| `reset_prefix_cache(True)`, consumer + async class | raises `RuntimeError` after preempt, hash `HHHHHHHH`, lookup/resume 112, `violated` | `violated` (unchanged) |
| `reset_prefix_cache(True)`, defer off (producer or consumer, sync class) | reset returns true, hash `--------`, hit 0 | same |
| Async load (`load_async=True`) | schedule hash `-------`. Abort-then-failure reuse 0. Not a prefix-cache leak | same |

Async-load `fail` while the owner is still `WAITING_FOR_REMOTE_KVS` finishes `FINISHED_ERROR`. `recompute` leaves it waiting and lowers `num_computed_tokens` (`async_handler_saw_owner`). Those blocks were never cached (`delay_cache_blocks`).

## What the traces mean

`_handle_invalid_blocks` only scans `running` and `WAITING_FOR_REMOTE_KVS` entries in `kv_holding_waiting`. On both SHAs it returns before `evict_blocks` when that scan finds nobody (`if not total_failed_requests: return set()`).

Abort calls `finish_requests` first, which frees the request and drops it out of `running`. The late `invalid_block_ids` report then matches no request. Hashes published at `allocate_slots` stay. The next same-prefix request hits 112 tokens, not 48. This is the sync scheduler path. It does not need async scheduling or a GPU. CPU platform code forces `async_scheduling=False`, and the abort rows still fail with that flag left false.

#69299 changes the visible-owner `recompute` path: it evicts the suffix even when the owner is alive, and rewinds the cache cursor. After that, aborting the owner leaves reuse at 48. That matches the PR text: failure report before abort, `recompute`. It does not record block ids for an owner that already left `running`. Abort-before-failure and preempt-before-failure stay `HHHHHHHH` / 112 on the PR head.

Producer pressure is a real second `schedule()` while the first output is outstanding and the pool has no spare block (`num_blocks=9`). The owner's next token cannot allocate, `schedule()` preempts it, and waiting admission is skipped for that step (`if not preempted_reqs`). `defer_block_free` is false for `kv_producer`, so the free happens. Consumer + async class sets `defer_block_free` and the fence refuses the free, so pressure does not preempt. That split matches the discussion on #61768 / #61912. Default CPU engine does not take this second `schedule()`; `vllm/platforms/cpu.py` forces async scheduling off. The pressure row is the GPU async-scheduling order, executed here by setting the flag after that platform check.

`reset_prefix_cache(reset_running_requests=True)` preempts first, then tries to reset. With the free fence held, reset raises `Failed to reset KV cache even when all the running requests are preempted...`. `EngineCore.reset_prefix_cache` does not catch that. The probe continues because the preempt already happened. Hashes remain and the failure report still misses the owner. Without the fence, reset succeeds and the hash mask is `--------`, so there is no stale hit to attribute to the failure report.

Direct `_preempt_request` does not pass `drop_stale_output`. The following `update_from_output` delivered 1 sampled token, and the resume delta was 128. Producer `schedule()` passes `drop_stale_output=requires_kv_delivery`, which is true for a producer, and that pressure row delivered no token; its resume hit was 112. Both are still past the 48-token valid prefix. This run does not show stale-sample delivery on the producer `schedule()` pressure path.

Policy `fail` does not finish a preempted owner. Status stays `PREEMPTED` on main and on #69299. The visible `fail` control does finish `FINISHED_ERROR`, so the miss is the scan, not the fail path itself.

## Overlap

- [#61768](https://github.com/vllm-project/vllm/issues/61768) is the abort-before-report case. Still open. [#61769](https://github.com/vllm-project/vllm/pull/61769) is closed unmerged (`2fdcff39`).
- [#61912](https://github.com/vllm-project/vllm/issues/61912) is the preempt-before-report case (pressure and reset). Still open. No merged fix found in `418af607` (`_finished_inflight` / preempt block-id record are absent).
- [#62438](https://github.com/vllm-project/vllm/pull/62438) is closed unmerged and is the producer block-reuse fence for a different report-after-realloc bug, not this eviction gap.
- [#69299](https://github.com/vllm-project/vllm/pull/69299) states that abort/preempt timing from #61768 and #61912 is out of scope. These probes agree.

## Conclusion

**Reproducible gap, not fixed on current main, and not fixed by #69299.**

#69299 does what it claims for a synchronous load whose owner is still running when the failure report is applied: under `recompute`, the failed suffix is evicted, and a later abort no longer leaves that suffix reusable. It does not evict when the owner was aborted or preempted before the report. Those rows fail the same way on `ecbebf9f` as on `418af607`.

The stale-cache symptom is the synchronous load path. An async load does not publish prefix hashes at schedule time, so abort-before-failure there did not create a hit.

No new PR is warranted from this run. The abort and preempt gaps are already tracked, and the earlier patches were not merged. A later fix, if requested, would have to remember block ids for an owner that leaves `running` before `update_from_output`, including preempt, and then evict from the first invalid block. That is outside #69299. This probe did not implement that.
