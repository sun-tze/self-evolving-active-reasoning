"""Pure allocation time/receipt rules, shared by wrapper and controller."""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

from .core import CONTRACT_SHA, PROTOCOL_SHA, ADDENDUM_SHA, IntegrityError

WALL_SECONDS = 7200
SAVE_RESERVE_SECONDS = 120
SHUTDOWN_RESERVE_SECONDS = 120
TOTAL_ALLOCATION_SECONDS = 14400


@dataclass(frozen=True)
class AllocationContext:
    stage: str
    job_id: str
    experiment_id: str
    allocation_start_unix: float
    controller_start_unix: float
    implementation_sha256: str
    wall_seconds: int = WALL_SECONDS
    save_reserve_seconds: int = SAVE_RESERVE_SECONDS
    shutdown_reserve_seconds: int = SHUTDOWN_RESERVE_SECONDS

    def __post_init__(self):
        if (self.stage not in ("stage-one","stage-two") or not self.job_id.isdigit() or
                not self.experiment_id or not all(math.isfinite(t) for t in
                    (self.allocation_start_unix,self.controller_start_unix)) or
                self.allocation_start_unix<=0 or self.controller_start_unix<self.allocation_start_unix or
                self.wall_seconds!=WALL_SECONDS or self.save_reserve_seconds!=SAVE_RESERVE_SECONDS or
                self.shutdown_reserve_seconds!=SHUTDOWN_RESERVE_SECONDS):
            raise IntegrityError("Invalid allocation timing/identity contract")

    @property
    def hard_deadline_unix(self):
        return self.allocation_start_unix+self.wall_seconds

    @property
    def controller_deadline_unix(self):
        return self.hard_deadline_unix-self.shutdown_reserve_seconds

    @property
    def request_deadline_unix(self):
        return self.controller_deadline_unix-self.save_reserve_seconds

    def startup_elapsed(self,now_unix=None):
        now=time.time() if now_unix is None else now_unix
        if not math.isfinite(now) or now<self.controller_start_unix:
            raise IntegrityError("Allocation clock moved backwards")
        return now-self.allocation_start_unix

    def request_monotonic_deadline(self,now_unix=None,now_monotonic=None):
        now=time.time() if now_unix is None else now_unix
        mono=time.monotonic() if now_monotonic is None else now_monotonic
        self.startup_elapsed(now)
        remaining=self.request_deadline_unix-now
        if remaining<=0:
            raise IntegrityError("Allocation request window exhausted")
        return mono+remaining

    def document(self):
        return dict(format="sun_gn_allocation_context_v1",**self.__dict__,
                    protocol_sha256=PROTOCOL_SHA,protocol_addendum_sha256=ADDENDUM_SHA,contract_sha256=CONTRACT_SHA,
                    gpu_count=1,gpu_type="l40s",hard_deadline_unix=self.hard_deadline_unix,
                    controller_deadline_unix=self.controller_deadline_unix,
                    request_deadline_unix=self.request_deadline_unix,
                    measured_pre_controller_seconds=self.controller_start_unix-self.allocation_start_unix)

    @classmethod
    def load(cls,document,stage,implementation_sha,experiment_id,job_id):
        fields={k:document[k] for k in cls.__dataclass_fields__}
        context=cls(**fields)
        if (document!=context.document() or context.stage!=stage or
                context.implementation_sha256!=implementation_sha or
                context.experiment_id!=experiment_id or context.job_id!=job_id):
            raise IntegrityError("Allocation receipt differs from current stage/release/job")
        return context


def validate_prior_allocation(receipt,experiment_id,implementation_sha,current_job_id):
    if (receipt.get("stage")!="stage-one" or receipt.get("status")!="completed" or
            receipt.get("experiment_id")!=experiment_id or receipt.get("implementation_sha256")!=implementation_sha or
            receipt.get("protocol_sha256")!=PROTOCOL_SHA or receipt.get("protocol_addendum_sha256")!=ADDENDUM_SHA or receipt.get("contract_sha256")!=CONTRACT_SHA or
            receipt.get("job_id")==current_job_id or not str(receipt.get("job_id","")).isdigit() or
            receipt.get("gpu_count")!=1 or receipt.get("gpu_type")!="l40s" or
            type(receipt.get("allocation_elapsed_seconds")) not in (int,float) or
            not math.isfinite(receipt["allocation_elapsed_seconds"]) or
            not 0<=receipt["allocation_elapsed_seconds"]<=WALL_SECONDS):
        raise IntegrityError("Prior allocation did not complete within the frozen two-job contract")
    if receipt["allocation_elapsed_seconds"]+WALL_SECONDS>TOTAL_ALLOCATION_SECONDS:
        raise IntegrityError("Total allocation budget exceeded")


def stage_two_time_check(gate,N,context,now_unix=None):
    """Recheck the frozen N with actual second-job startup; never resize it."""
    elapsed=context.startup_elapsed(now_unix)
    allowance=max(gate["allowance_seconds"],2*(elapsed+context.save_reserve_seconds+context.shutdown_reserve_seconds))
    estimate=allowance+math.ceil(4*N/8)*gate["actor_conservative_seconds"]
    if estimate>5400:
        raise IntegrityError("Actual second-stage startup no longer fits frozen N within ninety minutes")
    return dict(selected_N=N,actual_startup_seconds=elapsed,allowance_seconds=allowance,
                stage_two_estimated_seconds=estimate)
