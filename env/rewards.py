from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RewardTracker:
    """
    Incremental reward shaping.

    We award only for *new* correct progress (delta-based), so repeated actions
    don't keep accruing reward without improving the state.
    """

    extracted_tasks_correct: int = 0
    deadlines_correct: int = 0
    scheduled_correct: int = 0
    rescheduled_correct: int = 0
    missed_deadline_penalized: bool = False

    def as_dict(self) -> Dict[str, int | bool]:
        return {
            "extracted_tasks_correct": self.extracted_tasks_correct,
            "deadlines_correct": self.deadlines_correct,
            "scheduled_correct": self.scheduled_correct,
            "rescheduled_correct": self.rescheduled_correct,
            "missed_deadline_penalized": self.missed_deadline_penalized,
        }


@dataclass
class RewardResult:
    value: float
    components: Dict[str, float] = field(default_factory=dict)


def compute_incremental_reward(
    *,
    prev: RewardTracker,
    curr: RewardTracker,
    invalid_action: bool,
) -> RewardResult:
    components: Dict[str, float] = {}
    reward = 0.0

    if invalid_action:
        components["invalid_action"] = -0.2
        reward += components["invalid_action"]

    # correct task extraction → +0.3
    delta_tasks = max(0, curr.extracted_tasks_correct - prev.extracted_tasks_correct)
    if delta_tasks:
        components["task_extraction"] = 0.3 * delta_tasks
        reward += components["task_extraction"]

    # correct deadline parsing → +0.3
    delta_deadlines = max(0, curr.deadlines_correct - prev.deadlines_correct)
    if delta_deadlines:
        components["deadline_parsing"] = 0.3 * delta_deadlines
        reward += components["deadline_parsing"]

    # valid scheduling before deadline → +0.4
    delta_sched = max(0, curr.scheduled_correct - prev.scheduled_correct)
    if delta_sched:
        components["scheduling"] = 0.4 * delta_sched
        reward += components["scheduling"]

    # smart rescheduling → +0.5
    delta_resched = max(0, curr.rescheduled_correct - prev.rescheduled_correct)
    if delta_resched:
        components["rescheduling"] = 0.5 * delta_resched
        reward += components["rescheduling"]

    # missed deadline → -1.0 (only once)
    if (not prev.missed_deadline_penalized) and curr.missed_deadline_penalized:
        components["missed_deadline"] = -1.0
        reward += components["missed_deadline"]

    return RewardResult(value=reward, components=components)

