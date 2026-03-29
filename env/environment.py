from __future__ import annotations

import re
import uuid
from dataclasses import asdict
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from env.models import Action, ActionType, CalendarEvent, ExtractedTask, Observation, Reward
from env.rewards import RewardTracker, compute_incremental_reward
from tasks.easy_task import get_easy_task
from tasks.hard_task import get_hard_task
from tasks.medium_task import get_medium_task


MONTHS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)

_MONTH_TO_NUM = {m: i + 1 for i, m in enumerate(MONTHS)}
_MONTH_ABBR_TO_NUM = {m[:3]: i + 1 for i, m in enumerate(MONTHS)}


def _norm(s: str) -> str:
    return " ".join(s.lower().strip().split())


def _iso(d: date) -> str:
    return d.isoformat()


def _parse_date_candidates(text: str, *, default_year: int) -> List[str]:
    """
    Extract ISO date candidates from free-form email text.
    Supports patterns like:
      - "15 April 2026"
      - "Apr 15, 2026"
      - "15 Apr"
    """
    out: List[str] = []
    # Explicit date-like phrases (keep simple & deterministic)
    # 1) "15 April 2026"
    pat1 = re.compile(
        r"\b([0-3]?\d)\s+(" + "|".join(MONTHS) + r")\s+(20\d{2})\b",
        re.IGNORECASE,
    )
    for m in pat1.finditer(text):
        day_s, month_s, year_s = m.group(1), m.group(2), m.group(3)
        try:
            day = int(day_s)
            month = _MONTH_TO_NUM[_norm(month_s)]
            year = int(year_s)
            out.append(date(year, month, day).isoformat())
        except Exception:
            continue

    # 2) "Apr 15, 2026"
    pat2 = re.compile(
        r"\b(" + "|".join(m[:3] for m in MONTHS) + r")\s+([0-3]?\d),\s*(20\d{2})\b",
        re.IGNORECASE,
    )
    for m in pat2.finditer(text):
        month_s, day_s, year_s = m.group(1), m.group(2), m.group(3)
        try:
            day = int(day_s)
            month = _MONTH_ABBR_TO_NUM[_norm(month_s)[:3]]
            year = int(year_s)
            out.append(date(year, month, day).isoformat())
        except Exception:
            continue

    # 3) "15 April" (assume default year)
    pat3 = re.compile(
        r"\b([0-3]?\d)\s+(" + "|".join(MONTHS) + r")\b",
        re.IGNORECASE,
    )
    for m in pat3.finditer(text):
        day_s, month_s = m.group(1), m.group(2)
        # avoid duplicating matches that already included a year
        span_text = text[m.start() : m.end() + 6]
        if re.search(r"\b20\d{2}\b", span_text):
            continue
        try:
            day = int(day_s)
            month = _MONTH_TO_NUM[_norm(month_s)]
            out.append(date(int(default_year), month, day).isoformat())
        except Exception:
            continue

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for d in out:
        if d not in seen:
            seen.add(d)
            deduped.append(d)
    return deduped


def _extract_task_phrases(text: str) -> List[str]:
    """
    Heuristic task extraction:
      - bullet lines starting with '-' or '*'
      - imperative lines containing 'please' / 'need' / 'we need'
      - numbered checklist items "1) ..."
    """
    tasks: List[str] = []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        if re.match(r"^[-*]\s+", ln):
            tasks.append(re.sub(r"^[-*]\s+", "", ln))
            continue
        # Numbered items are treated as potential *subtasks* (created via split_task),
        # not top-level tasks.
        if ln.lower().startswith("please "):
            tasks.append(ln[7:])
            continue
        if "please:" in ln.lower():
            continue
        if re.search(r"\bplease\b", ln, re.IGNORECASE) and any(v in ln.lower() for v in ["submit", "send", "book", "prepare", "draft", "schedule", "get "]):
            # strip leading context
            cleaned = re.sub(r"^.*\bplease\b[:,]?\s*", "", ln, flags=re.IGNORECASE).strip()
            tasks.append(cleaned)
            continue
        if re.search(r"\bwe need\b", ln, re.IGNORECASE):
            cleaned = re.sub(r"^.*\bwe need\b\s*", "", ln, flags=re.IGNORECASE).strip()
            tasks.append(cleaned)
            continue

    # Normalize & prune obvious non-tasks
    cleaned_tasks: List[str] = []
    for t in tasks:
        t2 = t.strip().strip(".")
        t2 = re.sub(r"\bby\s+.*$", "", t2, flags=re.IGNORECASE).strip()
        t2 = re.sub(r"\bon\s+\d{4}-\d{2}-\d{2}\b.*$", "", t2, flags=re.IGNORECASE).strip()
        # Normalize leading articles for more stable grading
        t2 = re.sub(r"^(a|an|the)\s+", "", t2, flags=re.IGNORECASE).strip()
        if len(t2) < 4:
            continue
        cleaned_tasks.append(t2)

    # Deduplicate (case-insensitive)
    seen = set()
    out: List[str] = []
    for t in cleaned_tasks:
        n = _norm(t)
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


class SmartEmailTaskCalendarEnv:
    """
    OpenEnv-compatible environment:
      - reset() -> Observation
      - step(action) -> (Observation, Reward, done, info)
      - state() -> dict (full internal state)
    """

    def __init__(self) -> None:
        self._task_specs = {
            "easy": get_easy_task(),
            "medium": get_medium_task(),
            "hard": get_hard_task(),
        }
        self._task_order = ["easy", "medium", "hard"]
        self._task_ptr = 0

        self.task_spec: Dict[str, Any] = {}
        self.email_index: int = 0
        self.email_text: str = ""
        self.current_date: date = date.today()

        self.tasks: List[ExtractedTask] = []
        self.deadlines: List[Dict[str, Any]] = []
        self.calendar: List[CalendarEvent] = []

        self.last_action: str = ""
        self.last_action_error: bool = False

        self.reward_tracker = RewardTracker()

        # For incremental reward (compare to expected)
        self._expected_titles: List[str] = []
        self._expected_due_by_title: Dict[str, Optional[str]] = {}

    def reset(self, task_id: Optional[str] = None) -> Observation:
        if task_id is None:
            task_id = self._task_order[self._task_ptr % len(self._task_order)]
            self._task_ptr += 1
        if task_id not in self._task_specs:
            raise ValueError(f"Unknown task_id: {task_id}")

        self.task_spec = self._task_specs[task_id]
        self.email_index = 0
        self.email_text = self.task_spec["emails"][0]
        self.current_date = datetime.strptime(self.task_spec["start_date"], "%Y-%m-%d").date()

        self.tasks = []
        self.deadlines = []
        self.calendar = []
        self.last_action = ""
        self.last_action_error = False
        self.reward_tracker = RewardTracker()

        self._expected_titles = [_norm(x["title"]) for x in self.task_spec.get("expected_tasks", [])]
        self._expected_due_by_title = self._current_expected_due_map()

        return self._observation()

    def state(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_spec.get("task_id"),
            "task_spec": self.task_spec,
            "email_index": self.email_index,
            "email_text": self.email_text,
            "current_date": _iso(self.current_date),
            "tasks": [t.model_dump() for t in self.tasks],
            "deadlines": self.deadlines,
            "calendar": [e.model_dump() for e in self.calendar],
            "reward_tracker": self.reward_tracker.as_dict(),
            "last_action": self.last_action,
            "last_action_error": self.last_action_error,
        }

    def step(self, action: Action | Dict[str, Any]) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if isinstance(action, dict):
            action = Action.model_validate(action)

        prev_tracker = RewardTracker(**asdict(self.reward_tracker))
        self.last_action = action.action.value
        self.last_action_error = False
        invalid_action = False
        info: Dict[str, Any] = {}

        try:
            if action.action == ActionType.parse_email:
                self._act_parse_email()
            elif action.action == ActionType.extract_deadline:
                self._act_extract_deadline()
            elif action.action == ActionType.create_task:
                self._act_create_task()
            elif action.action == ActionType.split_task:
                self._act_split_task(action.params)
            elif action.action == ActionType.schedule_task:
                self._act_schedule_task(action.params)
            elif action.action == ActionType.reschedule_task:
                self._act_reschedule_task(action.params)
            elif action.action == ActionType.noop:
                pass
            else:
                invalid_action = True
                self.last_action_error = True
        except Exception as e:
            invalid_action = True
            self.last_action_error = True
            info["error"] = str(e)

        # time moves forward slightly each step (keeps realism; deterministic)
        self.current_date = self.current_date + timedelta(days=1)

        # Update reward tracker based on current state vs expected
        self._update_reward_tracker()
        reward_res = compute_incremental_reward(prev=prev_tracker, curr=self.reward_tracker, invalid_action=invalid_action)
        reward = Reward(value=reward_res.value, components=reward_res.components)

        done = self._is_done()
        info.update(
            {
                "task_id": self.task_spec.get("task_id"),
                "reward_components": reward.components,
            }
        )
        return (self._observation(), reward, done, info)

    def _observation(self) -> Observation:
        return Observation(
            email_text=self.email_text,
            extracted_tasks=[t.model_dump() for t in self.tasks],
            deadlines=self.deadlines,
            calendar=[e.model_dump() for e in self.calendar],
            current_date=_iso(self.current_date),
            last_action=self.last_action,
            last_action_error=self.last_action_error,
        )

    def _act_parse_email(self) -> None:
        """
        Advances to the next email if one exists; otherwise no-op.
        """
        if not self.task_spec:
            raise RuntimeError("Environment not reset")
        if self.email_index + 1 < len(self.task_spec["emails"]):
            self.email_index += 1
            self.email_text = self.task_spec["emails"][self.email_index]
            # Deadline expectations can change after an update email (hard task).
            self._expected_due_by_title = self._current_expected_due_map()

    def _act_extract_deadline(self) -> None:
        default_year = self.current_date.year
        dates = _parse_date_candidates(self.email_text, default_year=default_year)
        # Attach a deadline record for each expected task if possible.
        # Heuristic: if only one expected task, use the first date candidate.
        self.deadlines = []
        if not dates:
            return

        if len(self._expected_titles) == 1:
            title = self._expected_titles[0]
            self.deadlines.append({"title": title, "due_date": dates[0], "source": "email"})
            self._set_task_due_if_exists(title, dates[0])
            return

        # For multiple tasks: try per-line association; fall back to sequential
        lines = [ln.strip() for ln in self.email_text.splitlines() if ln.strip()]
        candidates: List[Tuple[str, str]] = []
        for ln in lines:
            ln_dates = _parse_date_candidates(ln, default_year=default_year)
            if not ln_dates:
                continue
            task_phrases = _extract_task_phrases(ln)
            if task_phrases:
                candidates.append((task_phrases[0], ln_dates[0]))

        used = set()
        for t, d in candidates:
            nt = _norm(t)
            # snap to expected title if close match (simple contains check)
            matched = None
            for exp in self._expected_titles:
                if exp in nt or nt in exp:
                    matched = exp
                    break
            if matched and matched not in used:
                used.add(matched)
                self.deadlines.append({"title": matched, "due_date": d, "source": "email"})
                self._set_task_due_if_exists(matched, d)

        # If still missing, assign remaining expected titles sequentially
        remaining = [t for t in self._expected_titles if t not in used]
        date_idx = 0
        for t in remaining:
            if date_idx >= len(dates):
                break
            self.deadlines.append({"title": t, "due_date": dates[date_idx], "source": "email"})
            self._set_task_due_if_exists(t, dates[date_idx])
            date_idx += 1

    def _act_create_task(self) -> None:
        titles = _extract_task_phrases(self.email_text)
        for title in titles:
            if any(_norm(t.title) == title for t in self.tasks):
                continue
            tid = str(uuid.uuid4())[:8]
            # Tasks should not "magically" know the due date; it must be extracted from the email.
            self.tasks.append(ExtractedTask(task_id=tid, title=title, due_date=None))

    def _current_expected_due_map(self) -> Dict[str, Optional[str]]:
        """
        Expected due dates for reward shaping.
        For the hard task, the expected due date changes after the update email is parsed.
        """
        base = { _norm(x["title"]): x.get("due_date") for x in self.task_spec.get("expected_tasks", []) }
        if self.task_spec.get("task_id") != "hard":
            return base
        # During the first email, we expect the original deadline to be extracted.
        if self.email_index == 0:
            original = self.task_spec.get("initial_due_date")
            if original and self._expected_titles:
                base[self._expected_titles[0]] = original
        # After parsing the update email, we expect the updated deadline.
        if self.email_index >= 1:
            updated = self.task_spec.get("updated_due_date")
            if updated and self._expected_titles:
                base[self._expected_titles[0]] = updated
        return base

    def _act_split_task(self, params: Dict[str, Any]) -> None:
        """
        params:
          - task_title: str (preferred) OR task_id: str
        """
        task = self._find_task(params)
        if task is None:
            raise ValueError("Task not found for split_task")

        # If the email has explicit numbered subtasks, take those; else split by "and"
        lines = [ln.strip() for ln in self.email_text.splitlines() if ln.strip()]
        subs = []
        for ln in lines:
            if re.match(r"^\d+\)\s+", ln):
                subs.append(_norm(re.sub(r"^\d+\)\s+", "", ln).strip().strip(".")))
        if not subs:
            parts = re.split(r"\band\b|,", task.title, flags=re.IGNORECASE)
            subs = [_norm(p.strip().strip(".")) for p in parts if _norm(p.strip().strip("."))]

        # dedupe preserve
        seen = set()
        out = []
        for s in subs:
            if s not in seen:
                seen.add(s)
                out.append(s)
        task.subtasks = out
        task.status = "split"

    def _act_schedule_task(self, params: Dict[str, Any]) -> None:
        """
        params:
          - task_title or task_id
          - start_date (optional, ISO)
          - end_date (optional, ISO)
        If start/end not provided, schedule as a 1-day event on current_date.
        """
        task = self._find_task(params)
        if task is None:
            raise ValueError("Task not found for schedule_task")

        start_s = params.get("start_date")
        end_s = params.get("end_date")
        if start_s:
            start = datetime.strptime(start_s, "%Y-%m-%d").date()
        else:
            start = self.current_date
        if end_s:
            end = datetime.strptime(end_s, "%Y-%m-%d").date()
        else:
            end = start

        eid = str(uuid.uuid4())[:8]
        self.calendar.append(
            CalendarEvent(
                event_id=eid,
                task_id=task.task_id,
                title=task.title,
                start_date=_iso(start),
                end_date=_iso(end),
                due_date=task.due_date,
            )
        )
        task.status = "scheduled"

    def _act_reschedule_task(self, params: Dict[str, Any]) -> None:
        """
        params:
          - task_title or task_id
          - new_due_date: ISO date (optional)
          - new_start_date / new_end_date (optional)
        """
        task = self._find_task(params)
        if task is None:
            raise ValueError("Task not found for reschedule_task")

        new_due = params.get("new_due_date")
        if new_due:
            task.due_date = new_due
            # also update deadline record
            found = False
            for d in self.deadlines:
                if _norm(d.get("title", "")) == _norm(task.title):
                    d["due_date"] = new_due
                    d["source"] = "update"
                    found = True
            if not found:
                self.deadlines.append({"title": _norm(task.title), "due_date": new_due, "source": "update"})

        new_start_s = params.get("new_start_date")
        new_end_s = params.get("new_end_date")
        if new_start_s:
            new_start = datetime.strptime(new_start_s, "%Y-%m-%d").date()
        else:
            new_start = self.current_date
        if new_end_s:
            new_end = datetime.strptime(new_end_s, "%Y-%m-%d").date()
        else:
            new_end = new_start

        # Update existing event if present, else create
        event = None
        for e in self.calendar:
            if e.task_id == task.task_id:
                event = e
                break
        if event is None:
            eid = str(uuid.uuid4())[:8]
            self.calendar.append(
                CalendarEvent(
                    event_id=eid,
                    task_id=task.task_id,
                    title=task.title,
                    start_date=_iso(new_start),
                    end_date=_iso(new_end),
                    due_date=task.due_date,
                )
            )
        else:
            # Replace the event object to avoid any assignment edge-cases.
            idx = self.calendar.index(event)
            self.calendar[idx] = CalendarEvent(
                event_id=event.event_id,
                task_id=event.task_id,
                title=event.title,
                start_date=_iso(new_start),
                end_date=_iso(new_end),
                due_date=task.due_date,
            )

    def _find_task(self, params: Dict[str, Any]) -> Optional[ExtractedTask]:
        title = params.get("task_title")
        tid = params.get("task_id")
        if isinstance(tid, str):
            for t in self.tasks:
                if t.task_id == tid:
                    return t
        if isinstance(title, str):
            nt = _norm(title)
            for t in self.tasks:
                if _norm(t.title) == nt:
                    return t
            # allow contains matching
            for t in self.tasks:
                if nt in _norm(t.title) or _norm(t.title) in nt:
                    return t
        return self.tasks[0] if self.tasks and params.get("fallback_first") else None

    def _set_task_due_if_exists(self, title_norm: str, due_iso: str) -> None:
        for t in self.tasks:
            if _norm(t.title) == title_norm:
                t.due_date = due_iso

    def _update_reward_tracker(self) -> None:
        # Extracted task correctness: count how many expected titles are present in tasks
        got_titles = set(_norm(t.title) for t in self.tasks)
        self.reward_tracker.extracted_tasks_correct = sum(1 for t in self._expected_titles if t in got_titles)

        # Deadline correctness: title has due_date exactly matching expected
        correct_deadlines = 0
        for t in self.tasks:
            nt = _norm(t.title)
            if nt in self._expected_due_by_title and t.due_date == self._expected_due_by_title.get(nt):
                correct_deadlines += 1
        self.reward_tracker.deadlines_correct = correct_deadlines

        # Scheduling correctness: calendar has event for title and end_date <= due_date
        scheduled = 0
        for exp_title in self._expected_titles:
            exp_due = self._expected_due_by_title.get(exp_title)
            ev = next((e for e in self.calendar if _norm(e.title) == exp_title), None)
            if ev and exp_due:
                try:
                    end = datetime.strptime(ev.end_date, "%Y-%m-%d").date()
                    due = datetime.strptime(exp_due, "%Y-%m-%d").date()
                    # Require both: scheduled on/before due AND calendar's due_date reflects the task's due date.
                    if end <= due and ev.due_date == exp_due:
                        scheduled += 1
                except Exception:
                    pass
        self.reward_tracker.scheduled_correct = scheduled

        # Rescheduling correctness: for hard task, if update email seen and due date matches new expected
        if self.task_spec.get("task_id") == "hard":
            if self.email_index >= 1:
                # due correctness is already tracked; we specifically treat it as reschedule
                # if any expected title now has due date matching expected (post-update)
                res = 0
                for t in self.tasks:
                    nt = _norm(t.title)
                    if nt in self._expected_due_by_title and t.due_date == self._expected_due_by_title.get(nt):
                        res += 1
                self.reward_tracker.rescheduled_correct = res

        # Missed deadline: current date passed due and task not scheduled correctly
        missed = False
        for exp_title in self._expected_titles:
            exp_due = self._expected_due_by_title.get(exp_title)
            if not exp_due:
                continue
            due = datetime.strptime(exp_due, "%Y-%m-%d").date()
            if self.current_date > due:
                ev = next((e for e in self.calendar if _norm(e.title) == exp_title), None)
                if ev is None:
                    missed = True
                else:
                    try:
                        end = datetime.strptime(ev.end_date, "%Y-%m-%d").date()
                        if end > due:
                            missed = True
                    except Exception:
                        missed = True
        self.reward_tracker.missed_deadline_penalized = self.reward_tracker.missed_deadline_penalized or missed

    def _is_done(self) -> bool:
        # done when all expected titles appear, have correct due, and are scheduled before due
        if not self.task_spec:
            return True
        if self.reward_tracker.extracted_tasks_correct < len(self._expected_titles):
            return False
        if self.reward_tracker.deadlines_correct < len(self._expected_titles):
            return False
        if self.reward_tracker.scheduled_correct < len(self._expected_titles):
            return False
        # For hard: ensure update email has been seen (forces reschedule scenario)
        if self.task_spec.get("task_id") == "hard" and self.email_index < 1:
            return False
        return True

