# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reject raw block-id retention as a safe late-failure design.

Uses the real scheduler. Eviction experiments call the existing
``evict_blocks`` API from the probe. They do not patch production code.

Full attention lookup stops at the first missing chained hash
(``FullAttentionManager.find_longest_cache_hit``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from vllm.v1.request import RequestStatus

from probe_orphan_invalid_blocks import (
    INVALID_BLOCK_INDEX,
    PROMPT_BLOCKS,
    _admit_owner,
    _hash_mask,
    _lookup_hit,
    _make_scheduler,
    create_model_runner_output,
    create_request,
)


def _snapshot(scheduler, block_ids: list[int]) -> dict[int, object]:
    pool = scheduler.kv_cache_manager.block_pool
    return {block_id: pool.blocks[block_id].block_hash for block_id in block_ids}


def _still_matching(scheduler, snapshot: dict[int, object]) -> list[int]:
    pool = scheduler.kv_cache_manager.block_pool
    return [
        block_id
        for block_id, old_hash in snapshot.items()
        if old_hash is not None and pool.blocks[block_id].block_hash == old_hash
    ]


def _hit_of(scheduler, request) -> int:
    _blocks, num_cached, _boundary = scheduler.kv_cache_manager.get_computed_blocks(
        request
    )
    return num_cached


def _prepare(policy: str):
    """Abort a sync load, then let a different prompt reuse the physical blocks.

    Producer role keeps ``defer_block_free`` off, including with the async
    scheduler class. That is the overlapping-step order of #61905: schedule
    the next request before the previous output is applied.
    """
    scheduler = _make_scheduler(
        policy=policy,
        role="kv_producer",
        async_scheduling=True,
        num_blocks=9,
    )
    owner, scheduled, block_ids = _admit_owner(scheduler, async_load=False)
    snapshot = _snapshot(scheduler, block_ids)
    scheduler.finish_requests(owner.request_id, RequestStatus.FINISHED_ABORTED)
    block_size = scheduler.block_size
    other = create_request(
        num_tokens=PROMPT_BLOCKS * block_size,
        common_prefix_len=0,
        block_size=block_size,
    )
    scheduler.add_request(other)
    scheduler.schedule()
    other_ids = list(scheduler.kv_cache_manager.get_block_ids(other.request_id)[0])
    return {
        "scheduler": scheduler,
        "scheduled": scheduled,
        "owner": owner,
        "block_ids": block_ids,
        "snapshot": snapshot,
        "other": other,
        "other_ids": other_ids,
        "overlap": sorted(set(block_ids) & set(other_ids)),
        "matching_after_reuse": _still_matching(scheduler, snapshot),
    }


def _row(name: str, prepared, **extra) -> dict:
    scheduler = prepared["scheduler"]
    block_ids = prepared["block_ids"]
    other = prepared["other"]
    block_size = scheduler.block_size
    return {
        "case": name,
        "policy": extra.pop("policy"),
        "overlap": prepared["overlap"],
        "matching_ids_after_reuse": prepared["matching_after_reuse"],
        "hash_mask": _hash_mask(scheduler, block_ids),
        "same_prefix_hit": _lookup_hit(scheduler, block_size),
        "reuser_prefix_hit": _hit_of(scheduler, other),
        "other_status": other.status.name,
        "other_computed": other.num_computed_tokens,
        "other_ids": prepared["other_ids"],
        "owner_ids": block_ids,
        **extra,
    }


def case_current_report(policy: str) -> dict:
    prepared = _prepare(policy)
    scheduler = prepared["scheduler"]
    invalid = {prepared["block_ids"][INVALID_BLOCK_INDEX]}
    scheduler.update_from_output(
        prepared["scheduled"],
        create_model_runner_output(
            [prepared["owner"]],
            invalid_block_ids=invalid,
        ),
    )
    row = _row("current_report_after_reuse", prepared, policy=policy)
    row["note"] = (
        "Production _handle_invalid_blocks runs after the physical ids "
        "belong to a different prompt."
    )
    return row


def case_raw_id_evict(policy: str) -> dict:
    prepared = _prepare(policy)
    prepared["scheduler"].kv_cache_manager.evict_blocks(set(prepared["block_ids"]))
    row = _row("simulated_raw_id_evict", prepared, policy=policy)
    row["note"] = (
        "Probe-only: evict every recorded physical id, including ones the "
        "new request already rehashed. Does not call update_from_output."
    )
    return row


def case_hash_match_evict(policy: str) -> dict:
    prepared = _prepare(policy)
    matching = prepared["matching_after_reuse"]
    if matching:
        prepared["scheduler"].kv_cache_manager.evict_blocks(set(matching))
    row = _row("simulated_hash_match_evict", prepared, policy=policy)
    row["evicted_ids"] = matching
    row["note"] = (
        "Probe-only: evict a recorded id only while its block_hash is still "
        "the snapshot from the failed load."
    )
    return row


def case_noreuse_evict_reported_block_only() -> dict:
    scheduler = _make_scheduler(
        policy="fail",
        role="kv_consumer",
        async_scheduling=False,
        num_blocks=64,
    )
    owner, _scheduled, block_ids = _admit_owner(scheduler, async_load=False)
    snapshot = _snapshot(scheduler, block_ids)
    scheduler.finish_requests(owner.request_id, RequestStatus.FINISHED_ABORTED)
    reported = block_ids[INVALID_BLOCK_INDEX]
    assert snapshot[reported] == scheduler.kv_cache_manager.block_pool.blocks[
        reported
    ].block_hash
    before = _lookup_hit(scheduler, scheduler.block_size)
    scheduler.kv_cache_manager.evict_blocks({reported})
    after = _lookup_hit(scheduler, scheduler.block_size)
    return {
        "case": "noreuse_evict_reported_block_only",
        "policy": "fail",
        "hash_before": _hash_mask(scheduler, block_ids),
        "same_prefix_hit_before": before,
        "same_prefix_hit_after": after,
        "note": (
            "No reallocation. Evicting only the reported block breaks the "
            "full-attention hash chain, so later blocks need not be evicted "
            "to stop the hit."
        ),
    }


def main() -> None:
    rows = [
        case_noreuse_evict_reported_block_only(),
        case_current_report("fail"),
        case_current_report("recompute"),
        case_raw_id_evict("fail"),
        case_hash_match_evict("fail"),
    ]
    out = Path(__file__).resolve().parent / "out" / "aba.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print("scheduler_sha", os.environ.get("PROBE_SHA", "workspace"))
    for row in rows:
        print(json.dumps(row, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()
