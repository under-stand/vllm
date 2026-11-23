# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU scheduler probe for orphaned KV-load failures.

Runs the real v1 Scheduler / KV cache manager. Only the connector's match
and finish answers are mocked, the same way
``tests/v1/kv_connector/unit/test_invalid_blocks_correctness.py`` does.

This file records observations. It does not patch production code.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

from vllm.v1.request import RequestStatus

# Import the in-tree scheduler fixtures from the tree under test.
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "tests/v1/kv_connector/unit"))

from utils import (  # noqa: E402
    create_mock_connector,
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)


PROMPT_BLOCKS = 8
INVALID_BLOCK_INDEX = 3
# Blocks 0..6 can be a prefix hit; the final prompt block is not.
HITTABLE_BLOCKS = PROMPT_BLOCKS - 1


@dataclass
class Row:
    scenario: str
    ref: str
    sha: str
    policy: str
    load: str
    scheduling: str
    role: str
    defer_block_free: bool
    requires_kv_delivery: bool
    block_size: int
    owner_blocks: int
    hash_before: str
    hash_after: str
    owner_status: str
    owner_computed: int | None
    reuse_hit: int | None
    resume_hit: int | None
    valid_prefix_tokens: int
    full_hit_tokens: int
    delivered: list[dict] = field(default_factory=list)
    notes: str = ""
    verdict: str = ""
    error: str = ""


def _install_connector(scheduler, matched: dict[str, tuple[int, bool]]) -> None:
    connector = create_mock_connector()
    granted: set[str] = set()

    def get_num_new_matched_tokens(request, num_computed):
        # One external grant per request. Later schedules see only the local
        # prefix cache, which is the quantity under test.
        total, is_async = matched.get(request.request_id, (0, False))
        if request.request_id in granted or num_computed:
            return 0, False
        if total:
            granted.add(request.request_id)
        return total, is_async

    connector.get_num_new_matched_tokens.side_effect = get_num_new_matched_tokens
    connector.request_finished.return_value = (False, None)
    connector.take_events.return_value = ()
    connector.get_kv_connector_stats.return_value = None
    connector.get_loaded_kv_cache_group_ids.return_value = ()
    connector.has_pending_block_frees.return_value = False
    connector.supports_divergent_local_hybrid_hits = False
    connector.may_load_async.return_value = False
    scheduler.connector = connector


def _hash_mask(scheduler, block_ids: list[int]) -> str:
    pool = scheduler.kv_cache_manager.block_pool
    return "".join(
        "H" if pool.blocks[block_id].block_hash is not None else "-"
        for block_id in block_ids
    )


def _delivered(outputs) -> list[dict]:
    rows = []
    for client_outputs in outputs.values():
        for output in client_outputs.outputs:
            token_ids = list(getattr(output, "new_token_ids", []) or [])
            rows.append(
                {
                    "request_id": output.request_id,
                    "finish_reason": (
                        None
                        if output.finish_reason is None
                        else str(output.finish_reason)
                    ),
                    "num_tokens": len(token_ids),
                    "error": output.error is not None,
                }
            )
    return rows


def _lookup_hit(scheduler, block_size: int) -> int:
    """Local prefix hit only. Does not allocate or schedule."""
    request = create_request(
        num_tokens=PROMPT_BLOCKS * block_size,
        common_prefix_len=PROMPT_BLOCKS * block_size,
        block_size=block_size,
    )
    _blocks, num_cached, _boundary = scheduler.kv_cache_manager.get_computed_blocks(
        request
    )
    return num_cached


def _reuse_hit(scheduler, block_size: int) -> int:
    """Schedule a same-prefix request that is offered no external KV."""
    request = create_request(
        num_tokens=PROMPT_BLOCKS * block_size,
        common_prefix_len=PROMPT_BLOCKS * block_size,
        block_size=block_size,
    )
    scheduler.add_request(request)
    output = scheduler.schedule()
    scheduled = output.num_scheduled_tokens.get(request.request_id, 0)
    # schedule() adds the new tokens onto the cache hit.
    return request.num_computed_tokens - scheduled


def _make_scheduler(
    *,
    policy: str,
    role: str,
    async_scheduling: bool,
    num_blocks: int,
):
    config = create_vllm_config(
        kv_connector="ExampleConnector",
        kv_role=role,
        kv_load_failure_policy=policy,
        block_size=16,
        max_num_seqs=4,
        max_num_batched_tokens=4096,
        max_model_len=4096,
    )
    # CPU platform check forces this off. Set it afterwards so AsyncScheduler
    # and the consumer free-fence can be constructed the way GPU async
    # scheduling would. The row records that distinction.
    config.scheduler_config.async_scheduling = async_scheduling
    scheduler = create_scheduler(config, num_blocks=num_blocks)
    return scheduler


def _admit_owner(scheduler, *, async_load: bool):
    block_size = scheduler.block_size
    owner = create_request(
        num_tokens=PROMPT_BLOCKS * block_size,
        common_prefix_len=PROMPT_BLOCKS * block_size,
        block_size=block_size,
    )
    external = (PROMPT_BLOCKS - 1) * block_size
    _install_connector(scheduler, {owner.request_id: (external, async_load)})
    scheduler.add_request(owner)
    scheduled = scheduler.schedule()
    block_ids = list(scheduler.kv_cache_manager.get_block_ids(owner.request_id)[0])
    return owner, scheduled, block_ids


def _report(scheduler, scheduled, owner, block_ids):
    invalid = {block_ids[INVALID_BLOCK_INDEX]}
    outputs = scheduler.update_from_output(
        scheduled,
        create_model_runner_output([owner], invalid_block_ids=invalid),
    )
    return outputs


def _base_row(scheduler, scenario, policy, load, scheduling, role, block_ids) -> Row:
    block_size = scheduler.block_size
    return Row(
        scenario=scenario,
        ref=os.environ.get("PROBE_REF", "main"),
        sha=os.environ.get("PROBE_SHA", "unknown"),
        policy=policy,
        load=load,
        scheduling=scheduling,
        role=role,
        defer_block_free=bool(scheduler.defer_block_free),
        requires_kv_delivery=bool(scheduler.requires_kv_delivery),
        block_size=block_size,
        owner_blocks=len(block_ids),
        hash_before=_hash_mask(scheduler, block_ids),
        hash_after="",
        owner_status="",
        owner_computed=None,
        reuse_hit=None,
        resume_hit=None,
        valid_prefix_tokens=INVALID_BLOCK_INDEX * block_size,
        full_hit_tokens=HITTABLE_BLOCKS * block_size,
    )


def _orphan_verdict(row: Row) -> str:
    if row.error:
        return "error"
    if row.load == "async":
        if "H" not in row.hash_before and (row.reuse_hit or 0) == 0:
            return "not_applicable_async_uncached"
        if row.reuse_hit is not None and row.reuse_hit > row.valid_prefix_tokens:
            return "violated"
        return "held"
    if row.owner_blocks < PROMPT_BLOCKS:
        return "not_reached"
    if row.reuse_hit is None:
        return "error"
    if row.reuse_hit > row.valid_prefix_tokens:
        return "violated"
    return "held"


def scenario_failure_then_abort(policy: str, scheduling: str) -> Row:
    async_scheduling = scheduling == "async"
    scheduler = _make_scheduler(
        policy=policy,
        role="kv_consumer",
        async_scheduling=async_scheduling,
        num_blocks=64,
    )
    owner, scheduled, block_ids = _admit_owner(scheduler, async_load=False)
    row = _base_row(
        scheduler,
        "failure_then_abort",
        policy,
        "sync",
        scheduling,
        "kv_consumer",
        block_ids,
    )
    outputs = _report(scheduler, scheduled, owner, block_ids)
    row.delivered = _delivered(outputs)
    if owner.status != RequestStatus.FINISHED_ERROR:
        scheduler.finish_requests(owner.request_id, RequestStatus.FINISHED_ABORTED)
    row.hash_after = _hash_mask(scheduler, block_ids)
    row.owner_status = owner.status.name
    row.owner_computed = owner.num_computed_tokens
    row.reuse_hit = _reuse_hit(scheduler, scheduler.block_size)
    row.notes = (
        "Failure report is applied while the owner is still running, then "
        "the owner is aborted if the fail policy did not already finish it."
    )
    row.verdict = _orphan_verdict(row)
    return row


def scenario_abort_then_failure(policy: str, scheduling: str) -> Row:
    async_scheduling = scheduling == "async"
    scheduler = _make_scheduler(
        policy=policy,
        role="kv_consumer",
        async_scheduling=async_scheduling,
        num_blocks=64,
    )
    owner, scheduled, block_ids = _admit_owner(scheduler, async_load=False)
    row = _base_row(
        scheduler,
        "abort_then_failure",
        policy,
        "sync",
        scheduling,
        "kv_consumer",
        block_ids,
    )
    scheduler.finish_requests(owner.request_id, RequestStatus.FINISHED_ABORTED)
    outputs = _report(scheduler, scheduled, owner, block_ids)
    row.delivered = _delivered(outputs)
    row.hash_after = _hash_mask(scheduler, block_ids)
    row.owner_status = owner.status.name
    row.owner_computed = owner.num_computed_tokens
    row.reuse_hit = _reuse_hit(scheduler, scheduler.block_size)
    row.notes = (
        "finish_requests(ABORTED) runs before update_from_output, matching "
        "EngineCore applying the abort queue before the failure report."
    )
    row.verdict = _orphan_verdict(row)
    return row


def scenario_preempt_then_failure(policy: str, scheduling: str) -> Row:
    """Pop the owner and call _preempt_request, which schedule() itself calls."""
    async_scheduling = scheduling == "async"
    scheduler = _make_scheduler(
        policy=policy,
        role="kv_producer",
        async_scheduling=async_scheduling,
        num_blocks=64,
    )
    owner, scheduled, block_ids = _admit_owner(scheduler, async_load=False)
    row = _base_row(
        scheduler,
        "preempt_direct_then_failure",
        policy,
        "sync",
        scheduling,
        "kv_producer",
        block_ids,
    )
    assert owner in scheduler.running
    scheduler.running.remove(owner)
    scheduler._preempt_request(owner, 0.0)
    outputs = _report(scheduler, scheduled, owner, block_ids)
    row.delivered = _delivered(outputs)
    row.hash_after = _hash_mask(scheduler, block_ids)
    row.owner_status = owner.status.name
    row.owner_computed = owner.num_computed_tokens
    # Lookup before resume so a later schedule cannot republish hashes first.
    row.reuse_hit = _lookup_hit(scheduler, scheduler.block_size)
    resumed = scheduler.schedule()
    resumed_scheduled = resumed.num_scheduled_tokens.get(owner.request_id, 0)
    if owner.request_id in resumed.num_scheduled_tokens:
        row.resume_hit = owner.num_computed_tokens - resumed_scheduled
    else:
        row.resume_hit = None
    row.notes = (
        "Owner is popped from running and _preempt_request is called, the "
        "same call schedule() makes, before the failure report. reuse_hit is "
        "get_computed_blocks before resume. resume_hit is that resume. No "
        "second external grant is offered."
    )
    row.verdict = _orphan_verdict(row)
    if (
        row.verdict == "held"
        and row.resume_hit is not None
        and row.resume_hit > row.valid_prefix_tokens
    ):
        row.verdict = "violated"
        row.notes += " Resume itself hit past the invalid block."
    return row


def scenario_pressure_preempt(policy: str, role: str) -> Row:
    """Second schedule() before the first output, with the pool exactly full.

    Async scheduling tries to allocate the owner's next token while the load
    step is still in flight. With 8 blocks in use and no spare block, that
    allocation preempts through schedule(), then waiting admission is skipped
    for the rest of the step. A consumer with the free fence cannot free the
    in-flight blocks, so preemption does not happen.
    """
    scheduler = _make_scheduler(
        policy=policy,
        role=role,
        async_scheduling=True,
        num_blocks=9,
    )
    owner, scheduled, block_ids = _admit_owner(scheduler, async_load=False)
    row = _base_row(
        scheduler,
        "pressure_schedule_then_failure",
        policy,
        "sync",
        "async",
        role,
        block_ids,
    )
    pressure = scheduler.schedule()
    preempted = owner.status == RequestStatus.PREEMPTED
    row.notes = (
        f"second schedule preempted={preempted} "
        f"scheduled={list(pressure.num_scheduled_tokens)} "
        f"free_blocks="
        f"{scheduler.kv_cache_manager.block_pool.free_block_queue.num_free_blocks}"
    )
    if not preempted:
        row.owner_status = owner.status.name
        row.hash_after = _hash_mask(scheduler, block_ids)
        row.verdict = "not_reached"
        return row
    outputs = _report(scheduler, scheduled, owner, block_ids)
    row.delivered = _delivered(outputs)
    row.hash_after = _hash_mask(scheduler, block_ids)
    row.owner_status = owner.status.name
    row.owner_computed = owner.num_computed_tokens
    row.reuse_hit = _lookup_hit(scheduler, scheduler.block_size)
    if owner.status == RequestStatus.PREEMPTED:
        resumed = scheduler.schedule()
        resumed_scheduled = resumed.num_scheduled_tokens.get(owner.request_id, 0)
        if owner.request_id in resumed.num_scheduled_tokens:
            row.resume_hit = owner.num_computed_tokens - resumed_scheduled
    row.verdict = _orphan_verdict(row)
    if (
        row.verdict == "held"
        and row.resume_hit is not None
        and row.resume_hit > row.valid_prefix_tokens
    ):
        row.verdict = "violated"
        row.notes += " Resume itself hit past the invalid block."
    return row


def scenario_reset_then_failure(policy: str, role: str, async_scheduling: bool) -> Row:
    scheduling = "async" if async_scheduling else "sync"
    scheduler = _make_scheduler(
        policy=policy,
        role=role,
        async_scheduling=async_scheduling,
        num_blocks=64,
    )
    owner, scheduled, block_ids = _admit_owner(scheduler, async_load=False)
    row = _base_row(
        scheduler,
        "reset_running_then_failure",
        policy,
        "sync",
        scheduling,
        role,
        block_ids,
    )
    reset_note = ""
    try:
        reset_ok = scheduler.reset_prefix_cache(reset_running_requests=True)
        reset_note = f"reset_returned={reset_ok}"
    except Exception as exc:  # noqa: BLE001
        reset_note = f"reset_raised={type(exc).__name__}: {exc}"
    row.notes = (
        "reset_prefix_cache(reset_running_requests=True) is the public "
        f"preemption path. {reset_note}. "
        "EngineCore.reset_prefix_cache does not catch this exception; "
        "preemption has already run. The probe continues, as a caller that "
        "observes the failed RPC would."
    )
    outputs = _report(scheduler, scheduled, owner, block_ids)
    row.delivered = _delivered(outputs)
    row.hash_after = _hash_mask(scheduler, block_ids)
    row.owner_status = owner.status.name
    row.owner_computed = owner.num_computed_tokens
    row.reuse_hit = _lookup_hit(scheduler, scheduler.block_size)
    if owner.status == RequestStatus.PREEMPTED:
        resumed = scheduler.schedule()
        resumed_scheduled = resumed.num_scheduled_tokens.get(owner.request_id, 0)
        if owner.request_id in resumed.num_scheduled_tokens:
            row.resume_hit = owner.num_computed_tokens - resumed_scheduled
    if "reset_raised" in reset_note and owner.status == RequestStatus.PREEMPTED:
        row.verdict = _orphan_verdict(row)
        if (
            row.verdict == "held"
            and row.resume_hit is not None
            and row.resume_hit > row.valid_prefix_tokens
        ):
            row.verdict = "violated"
    elif "reset_returned=True" in reset_note and "H" not in row.hash_after:
        row.verdict = "not_applicable_reset_cleared"
    else:
        row.verdict = _orphan_verdict(row)
    return row


def scenario_success_abort_keeps_cache(scheduling: str) -> Row:
    async_scheduling = scheduling == "async"
    scheduler = _make_scheduler(
        policy="fail",
        role="kv_consumer",
        async_scheduling=async_scheduling,
        num_blocks=64,
    )
    owner, scheduled, block_ids = _admit_owner(scheduler, async_load=False)
    row = _base_row(
        scheduler,
        "control_success_then_abort",
        "fail",
        "sync",
        scheduling,
        "kv_consumer",
        block_ids,
    )
    outputs = scheduler.update_from_output(
        scheduled, create_model_runner_output([owner])
    )
    row.delivered = _delivered(outputs)
    scheduler.finish_requests(owner.request_id, RequestStatus.FINISHED_ABORTED)
    row.hash_after = _hash_mask(scheduler, block_ids)
    row.owner_status = owner.status.name
    row.reuse_hit = _reuse_hit(scheduler, scheduler.block_size)
    row.notes = "Successful sync load, then abort. Valid KV must stay reusable."
    if row.reuse_hit == row.full_hit_tokens and row.hash_after.count("H") >= HITTABLE_BLOCKS:
        row.verdict = "control_ok"
    else:
        row.verdict = "control_failed"
    return row


def scenario_visible_fail_evicts() -> Row:
    scheduler = _make_scheduler(
        policy="fail",
        role="kv_consumer",
        async_scheduling=False,
        num_blocks=64,
    )
    owner, scheduled, block_ids = _admit_owner(scheduler, async_load=False)
    row = _base_row(
        scheduler,
        "control_visible_fail_evicts",
        "fail",
        "sync",
        "sync",
        "kv_consumer",
        block_ids,
    )
    outputs = _report(scheduler, scheduled, owner, block_ids)
    row.delivered = _delivered(outputs)
    row.hash_after = _hash_mask(scheduler, block_ids)
    row.owner_status = owner.status.name
    row.owner_computed = owner.num_computed_tokens
    row.reuse_hit = _reuse_hit(scheduler, scheduler.block_size)
    row.notes = (
        "Owner is still running when the failure report arrives and policy "
        "is fail. Existing handler should evict the suffix and fail the owner."
    )
    suffix = row.hash_after[INVALID_BLOCK_INDEX:]
    if (
        owner.status == RequestStatus.FINISHED_ERROR
        and "H" not in suffix
        and row.reuse_hit == row.valid_prefix_tokens
    ):
        row.verdict = "control_ok"
    else:
        row.verdict = "control_failed"
    return row


def scenario_visible_recompute() -> Row:
    scheduler = _make_scheduler(
        policy="recompute",
        role="kv_consumer",
        async_scheduling=False,
        num_blocks=64,
    )
    owner, scheduled, block_ids = _admit_owner(scheduler, async_load=False)
    row = _base_row(
        scheduler,
        "visible_recompute_no_abort",
        "recompute",
        "sync",
        "sync",
        "kv_consumer",
        block_ids,
    )
    outputs = _report(scheduler, scheduled, owner, block_ids)
    row.delivered = _delivered(outputs)
    row.hash_after = _hash_mask(scheduler, block_ids)
    row.owner_status = owner.status.name
    row.owner_computed = owner.num_computed_tokens
    row.reuse_hit = _lookup_hit(scheduler, scheduler.block_size)
    row.notes = (
        "Failure while the owner is alive under recompute, no abort. "
        "reuse_hit is get_computed_blocks of another same-prefix request "
        "and does not allocate. Main historically keeps the suffix hashed "
        "so a live owner can repair it; #69299 evicts it immediately."
    )
    if owner.status == RequestStatus.RUNNING and owner.num_computed_tokens == (
        INVALID_BLOCK_INDEX * scheduler.block_size
    ):
        if row.reuse_hit is not None and row.reuse_hit > row.valid_prefix_tokens:
            row.verdict = "suffix_still_cached"
        else:
            row.verdict = "suffix_evicted"
    else:
        row.verdict = "control_failed"
    return row


def scenario_async_applicability(policy: str, ordering: str) -> Row:
    scheduler = _make_scheduler(
        policy=policy,
        role="kv_consumer",
        async_scheduling=False,
        num_blocks=64,
    )
    owner, scheduled, block_ids = _admit_owner(scheduler, async_load=True)
    row = _base_row(
        scheduler,
        f"async_{ordering}",
        policy,
        "async",
        "sync",
        "kv_consumer",
        block_ids,
    )
    computed_before = owner.num_computed_tokens
    row.notes = (
        f"owner_status_after_schedule={owner.status.name} "
        f"computed_before_report={computed_before}"
    )
    if ordering == "abort_then_failure":
        scheduler.finish_requests(owner.request_id, RequestStatus.FINISHED_ABORTED)
    outputs = _report(scheduler, scheduled, owner, block_ids)
    row.delivered = _delivered(outputs)
    row.hash_after = _hash_mask(scheduler, block_ids)
    row.owner_status = owner.status.name
    row.owner_computed = owner.num_computed_tokens
    if (
        ordering == "failure_while_waiting"
        and owner.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    ):
        if owner.num_computed_tokens < computed_before:
            row.verdict = "async_handler_saw_owner"
        else:
            row.verdict = "async_handler_missed"
        row.notes += " Async blocks were not prefix-cached at schedule time."
        return row
    if not owner.is_finished():
        scheduler.finish_requests(owner.request_id, RequestStatus.FINISHED_ABORTED)
    row.reuse_hit = _reuse_hit(scheduler, scheduler.block_size)
    if ordering == "failure_while_waiting":
        if policy == "fail" and row.owner_status != RequestStatus.FINISHED_ERROR.name:
            row.verdict = "control_failed"
        elif "H" in row.hash_before:
            row.verdict = _orphan_verdict(row)
        else:
            row.verdict = "not_applicable_async_uncached"
            row.notes += (
                " Async load uses delay_cache_blocks, so schedule does not "
                "publish prefix hashes."
            )
    else:
        row.verdict = _orphan_verdict(row)
    return row


def scenario_local_prefix_control() -> Row:
    scheduler = _make_scheduler(
        policy="fail",
        role="kv_consumer",
        async_scheduling=False,
        num_blocks=64,
    )
    block_size = scheduler.block_size
    first = create_request(
        num_tokens=PROMPT_BLOCKS * block_size,
        common_prefix_len=PROMPT_BLOCKS * block_size,
        block_size=block_size,
    )
    _install_connector(scheduler, {})
    scheduler.add_request(first)
    scheduled = scheduler.schedule()
    block_ids = list(scheduler.kv_cache_manager.get_block_ids(first.request_id)[0])
    row = _base_row(
        scheduler,
        "control_local_prefix_hit",
        "fail",
        "none",
        "sync",
        "kv_consumer",
        block_ids,
    )
    scheduler.update_from_output(scheduled, create_model_runner_output([first]))
    row.hash_before = _hash_mask(scheduler, block_ids)
    row.hash_after = row.hash_before
    row.owner_status = first.status.name
    row.reuse_hit = _reuse_hit(scheduler, block_size)
    row.notes = "No external load. A second identical prompt must hit local KV."
    if row.reuse_hit == row.full_hit_tokens:
        row.verdict = "control_ok"
    else:
        row.verdict = "control_failed"
    return row


def run_all() -> list[Row]:
    rows: list[Row] = []
    builders = [
        scenario_local_prefix_control,
        scenario_visible_fail_evicts,
        scenario_visible_recompute,
        lambda: scenario_success_abort_keeps_cache("sync"),
        lambda: scenario_success_abort_keeps_cache("async"),
    ]
    for policy in ("fail", "recompute"):
        for scheduling in ("sync", "async"):
            builders.append(
                lambda policy=policy, scheduling=scheduling: (
                    scenario_failure_then_abort(policy, scheduling)
                )
            )
            builders.append(
                lambda policy=policy, scheduling=scheduling: (
                    scenario_abort_then_failure(policy, scheduling)
                )
            )
            builders.append(
                lambda policy=policy, scheduling=scheduling: (
                    scenario_preempt_then_failure(policy, scheduling)
                )
            )
        for role in ("kv_producer", "kv_consumer"):
            builders.append(
                lambda policy=policy, role=role: scenario_pressure_preempt(
                    policy, role
                )
            )
        for role, async_scheduling in (
            ("kv_consumer", True),
            ("kv_producer", False),
            ("kv_consumer", False),
        ):
            builders.append(
                lambda policy=policy, role=role, async_scheduling=async_scheduling: (
                    scenario_reset_then_failure(policy, role, async_scheduling)
                )
            )
        for ordering in ("failure_while_waiting", "abort_then_failure"):
            builders.append(
                lambda policy=policy, ordering=ordering: scenario_async_applicability(
                    policy, ordering
                )
            )

    for build in builders:
        try:
            rows.append(build())
        except Exception as exc:  # noqa: BLE001
            rows.append(
                Row(
                    scenario="unhandled",
                    ref=os.environ.get("PROBE_REF", "main"),
                    sha=os.environ.get("PROBE_SHA", "unknown"),
                    policy="",
                    load="",
                    scheduling="",
                    role="",
                    defer_block_free=False,
                    requires_kv_delivery=False,
                    block_size=0,
                    owner_blocks=0,
                    hash_before="",
                    hash_after="",
                    owner_status="",
                    owner_computed=None,
                    reuse_hit=None,
                    resume_hit=None,
                    valid_prefix_tokens=0,
                    full_hit_tokens=0,
                    notes=repr(build),
                    verdict="error",
                    error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                )
            )
    return rows


def main() -> int:
    import vllm.v1.core.sched.scheduler as sched_mod

    print("scheduler_module", sched_mod.__file__)
    rows = run_all()
    out_dir = Path(os.environ.get("PROBE_OUT", _REPO / "investigation/kv-load-failure-ordering/out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = os.environ.get("PROBE_REF", "main")
    payload = [asdict(row) for row in rows]
    out_path = out_dir / f"{ref}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"{'scenario':<36} {'pol':<9} {'load':<5} {'sched':<5} "
        f"{'role':<12} {'defer':<5} {'hash_before':<10} {'hash_after':<10} "
        f"{'status':<16} {'reuse':>5} {'resume':>6} verdict"
    )
    controls_ok = True
    for row in rows:
        print(
            f"{row.scenario:<36} {row.policy:<9} {row.load:<5} {row.scheduling:<5} "
            f"{row.role:<12} {str(row.defer_block_free):<5} {row.hash_before:<10} "
            f"{row.hash_after:<10} {row.owner_status:<16} "
            f"{str(row.reuse_hit):>5} {str(row.resume_hit):>6} {row.verdict}"
        )
        if row.notes:
            print(f"  note: {row.notes}")
        if row.error:
            print(row.error)
            controls_ok = False
        if row.verdict == "control_failed":
            controls_ok = False
    print(f"wrote {out_path}")
    return 0 if controls_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
