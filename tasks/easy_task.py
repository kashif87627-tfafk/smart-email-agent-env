from __future__ import annotations

from typing import Dict, List


def get_easy_task() -> Dict:
    """
    Simple email with one task and one date.
    """
    email = (
        "Subject: Quick request\n\n"
        "Hi team,\n"
        "Please submit the expense report by 15 April 2026.\n"
        "Thanks,\n"
        "Morgan\n"
    )

    return {
        "task_id": "easy",
        "name": "Easy: one task + one date",
        "start_date": "2026-04-10",
        "emails": [email],
        "expected_tasks": [
            {"title": "submit the expense report", "due_date": "2026-04-15"},
        ],
        "expected_calendar": [
            {"title": "submit the expense report", "due_date": "2026-04-15"},
        ],
        "update": None,
    }

