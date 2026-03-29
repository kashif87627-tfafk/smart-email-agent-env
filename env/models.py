from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    parse_email = "parse_email"
    extract_deadline = "extract_deadline"
    create_task = "create_task"
    split_task = "split_task"
    schedule_task = "schedule_task"
    reschedule_task = "reschedule_task"
    noop = "noop"


class CalendarEvent(BaseModel):
    event_id: str
    task_id: str
    title: str
    start_date: str  # ISO date YYYY-MM-DD
    end_date: str  # ISO date YYYY-MM-DD (inclusive)
    due_date: Optional[str] = None  # ISO date YYYY-MM-DD


class ExtractedTask(BaseModel):
    task_id: str
    title: str
    status: Literal["new", "split", "scheduled", "done"] = "new"
    subtasks: List[str] = Field(default_factory=list)
    due_date: Optional[str] = None  # ISO date YYYY-MM-DD


class Observation(BaseModel):
    email_text: str
    extracted_tasks: List[Dict[str, Any]] = Field(default_factory=list)
    deadlines: List[Dict[str, Any]] = Field(default_factory=list)
    calendar: List[Dict[str, Any]] = Field(default_factory=list)
    current_date: str
    last_action: str = ""
    last_action_error: bool = False


class Action(BaseModel):
    action: ActionType
    params: Dict[str, Any] = Field(default_factory=dict)


class Reward(BaseModel):
    value: float = 0.0
    components: Dict[str, float] = Field(default_factory=dict)
