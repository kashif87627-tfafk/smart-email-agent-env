from __future__ import annotations

from typing import Dict


def get_medium_task() -> Dict:
    """
    Multiple tasks with deadlines.
    """
    email = (
        "Subject: Sprint admin\n\n"
        "Hi,\n"
        "- Prepare the Q2 roadmap draft by 20 April 2026.\n"
        "- Book a meeting with Finance by 18 April 2026.\n"
        "- Send the vendor follow-up email by 16 April 2026.\n"
        "Best,\n"
        "Avery\n"
    )

    return {
        "task_id": "medium",
        "name": "Medium: multiple tasks + deadlines",
        "start_date": "2026-04-14",
        "emails": [email],
        "expected_tasks": [
            {"title": "prepare the q2 roadmap draft", "due_date": "2026-04-20"},
            {"title": "book a meeting with finance", "due_date": "2026-04-18"},
            {"title": "send the vendor follow-up email", "due_date": "2026-04-16"},
        ],
        "expected_calendar": [
            {"title": "prepare the q2 roadmap draft", "due_date": "2026-04-20"},
            {"title": "book a meeting with finance", "due_date": "2026-04-18"},
            {"title": "send the vendor follow-up email", "due_date": "2026-04-16"},
        ],
        "update": None,
    }

