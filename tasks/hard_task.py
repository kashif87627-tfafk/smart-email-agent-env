from __future__ import annotations

from typing import Dict


def get_hard_task() -> Dict:
    """
    Multi-step task + deadline + update (reschedule scenario).
    Two emails: initial request, then deadline update.
    """
    email1 = (
        "Subject: Launch plan needed\n\n"
        "Hi,\n"
        "We need a launch plan for Project Orion by 25 April 2026.\n"
        "Please:\n"
        "1) Draft the plan\n"
        "2) Get legal review\n"
        "3) Schedule the stakeholder readout\n"
        "Thanks,\n"
        "Riley\n"
    )
    email2 = (
        "Subject: Update: earlier deadline\n\n"
        "Update: The deadline moved up. We now need the launch plan by 22 April 2026.\n"
        "Please adjust the schedule accordingly.\n"
        "- Riley\n"
    )

    return {
        "task_id": "hard",
        "name": "Hard: multi-step + deadline update + reschedule",
        "start_date": "2026-04-15",
        "emails": [email1, email2],
        # The environment expects the initial parsed deadline (email1) to be 2026-04-25,
        # then the updated deadline (email2) to be 2026-04-22.
        "initial_due_date": "2026-04-25",
        "updated_due_date": "2026-04-22",
        "expected_tasks": [
            {"title": "launch plan for project orion", "due_date": "2026-04-22"},
        ],
        "expected_subtasks": {
            "launch plan for project orion": [
                "draft the plan",
                "get legal review",
                "schedule the stakeholder readout",
            ]
        },
        "expected_calendar": [
            {"title": "launch plan for project orion", "due_date": "2026-04-22"},
        ],
        "update": {
            "email_index": 1,
            "new_due_date": "2026-04-22",
        },
    }

