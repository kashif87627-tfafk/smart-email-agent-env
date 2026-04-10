from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def _norm(s: str) -> str:
    return " ".join(s.lower().strip().split())


def _get_titles_from_tasks(tasks: List[Dict]) -> List[str]:
    out: List[str] = []
    for t in tasks:
        title = t.get("title")
        if isinstance(title, str):
            out.append(_norm(title))
    return out


def _get_due_from_tasks(tasks: List[Dict]) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for t in tasks:
        title = t.get("title")
        if isinstance(title, str):
            out[_norm(title)] = t.get("due_date")
    return out


def _get_calendar_due(calendar: List[Dict]) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for e in calendar:
        title = e.get("title")
        if isinstance(title, str):
            out[_norm(title)] = e.get("due_date")
    return out


def _score_task_and_due(
    *,
    expected: List[Dict],
    got_tasks: List[Dict],
    got_calendar: List[Dict],
) -> float:
    """
    Deterministic score in [0,1].
    We reward:
      - correct task titles present
      - correct due dates for those titles
      - calendar has same titles + due dates
    """
    if not expected:
        return 1.0

    expected_titles = [_norm(x["title"]) for x in expected]
    expected_due = { _norm(x["title"]): x.get("due_date") for x in expected }

    got_titles = set(_get_titles_from_tasks(got_tasks))
    got_due = _get_due_from_tasks(got_tasks)
    cal_due = _get_calendar_due(got_calendar)

    # title coverage
    title_hits = sum(1 for t in expected_titles if t in got_titles)
    title_score = title_hits / len(expected_titles)

    # due date correctness (only for titles that exist)
    due_hits = 0
    for t in expected_titles:
        if t in got_titles and got_due.get(t) == expected_due.get(t):
            due_hits += 1
    due_score = due_hits / len(expected_titles)

    # calendar correctness
    cal_hits = 0
    for t in expected_titles:
        if cal_due.get(t) == expected_due.get(t):
            cal_hits += 1
    cal_score = cal_hits / len(expected_titles)

    # weighted
    score = (0.4 * title_score) + (0.3 * due_score) + (0.3 * cal_score)
    return max(0.01, min(0.99, score))


def grade_easy(final_state: Dict) -> float:
    exp = final_state["task_spec"]["expected_tasks"]
    return _score_task_and_due(
        expected=exp,
        got_tasks=final_state.get("tasks", []),
        got_calendar=final_state.get("calendar", []),
    )


def grade_medium(final_state: Dict) -> float:
    exp = final_state["task_spec"]["expected_tasks"]
    return _score_task_and_due(
        expected=exp,
        got_tasks=final_state.get("tasks", []),
        got_calendar=final_state.get("calendar", []),
    )


def grade_hard(final_state: Dict) -> float:
    spec = final_state["task_spec"]
    exp = spec["expected_tasks"]
    base = _score_task_and_due(
        expected=exp,
        got_tasks=final_state.get("tasks", []),
        got_calendar=final_state.get("calendar", []),
    )

    # extra: subtasks split present
    expected_subtasks = spec.get("expected_subtasks") or {}
    got_tasks = final_state.get("tasks", [])
    got_subtasks_by_title = { _norm(t.get("title", "")): t.get("subtasks", []) for t in got_tasks }

    sub_score = 1.0
    if expected_subtasks:
        hits = 0
        total = 0
        for title, subs in expected_subtasks.items():
            total += 1
            got_subs = got_subtasks_by_title.get(_norm(title), [])
            got_norm = set(_norm(x) for x in got_subs if isinstance(x, str))
            want_norm = set(_norm(x) for x in subs)
            if want_norm.issubset(got_norm):
                hits += 1
        sub_score = hits / max(1, total)

    # extra: update handled (agent has seen second email)
    update_seen = bool(final_state.get("email_index", 0) >= 1)
    upd_score = 1.0 if update_seen else 0.0

    score = (0.7 * base) + (0.2 * sub_score) + (0.1 * upd_score)
    return max(0.01, min(0.99, score))

