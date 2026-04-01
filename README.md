---
title: Smart Email Agent Env
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

## Smart Email Task & Calendar Agent Environment

This is a **fully runnable OpenEnv-compatible environment** for a hackathon. It simulates how an agent:

- reads email text
- extracts tasks
- parses deadlines (e.g., “15 April 2026”)
- splits complex tasks into subtasks
- schedules tasks onto a calendar
- reacts to deadline updates by rescheduling

It ships with:

- **FastAPI server**: `POST /reset`, `POST /step`, `GET /state` (plus `GET /` and `GET /healthz`)
- **3 tasks**: easy / medium / hard
- **deterministic graders** (score \(0.0\)–\(1.0\))
- **inference runner** (`inference.py`) that can use the OpenAI SDK or a deterministic baseline

If you’re a judge: start at **“Quick demo”** and **“API usage”** below.

## Environment design

The environment is implemented in `env/environment.py` as `SmartEmailTaskCalendarEnv` and follows the OpenEnv-style API:

- `reset(task_id: Optional[str]) -> Observation`
- `step(action) -> (observation, reward, done, info)`
- `state() -> dict`

Internal state tracked:

- **tasks**: extracted tasks (with optional subtasks + due dates)
- **deadlines**: parsed per-task deadlines (source: email/update)
- **calendar**: scheduled calendar events

Time advances deterministically by **+1 day per step** to enable deadline-miss penalties (and to make grading deterministic).

## Quick demo (local)

From `smart-email-agent-env/`:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
python inference.py --api-base http://127.0.0.1:8000
```

You should see printed scores for `easy`, `medium`, and `hard`.

## Action space

Actions are strings (Pydantic enum in `env/models.py`):

- `parse_email`
- `extract_deadline`
- `create_task`
- `split_task`
- `schedule_task`
- `reschedule_task`
- `noop`

Each action accepts a `params` dict.

- **Important**: `schedule_task` and `reschedule_task` need a target task via `params.task_id` or `params.task_title`.

### What each action does (in this environment)

- **`create_task`**: extracts top-level tasks from the current email (this is “task extraction”)
- **`extract_deadline`**: parses deadline(s) from the current email and attaches them to tasks
- **`split_task`**: creates subtasks for a task (hard task uses numbered items in the email)
- **`schedule_task`**: adds a calendar event for the given task (defaults to a 1-day event on the current date)
- **`parse_email`**: moves to the **next email** in the task (used for the hard task’s update email)
- **`reschedule_task`**: updates due date and calendar event timing after a deadline update
- **`noop`**: no operation

## Observation space

Observation model (Pydantic in `env/models.py`) includes:

- `email_text: str`
- `extracted_tasks: list`
- `deadlines: list`
- `calendar: list`
- `current_date: str`
- `last_action: str`
- `last_action_error: bool`

## Reward shaping (incremental)

Reward is incremental (delta-based), not just final:

- correct task extraction → **+0.3**
- correct deadline parsing → **+0.3**
- valid scheduling before deadline → **+0.4**
- smart rescheduling → **+0.5**
- invalid action → **-0.2**
- missed deadline → **-1.0** (once)

Reward logic lives in `env/rewards.py`.

## Tasks

Task specs live in:

- `tasks/easy_task.py`
- `tasks/medium_task.py`
- `tasks/hard_task.py`

The **hard** task contains two emails: an initial request and a later email that **moves the deadline earlier**, requiring rescheduling.

## Graders

Graders are deterministic and return a score in \([0.0, 1.0]\):

- `tasks/graders.py`: `grade_easy`, `grade_medium`, `grade_hard`

Scoring checks:

- expected task titles present
- due dates correct
- calendar contains scheduled items with correct due dates
- hard task additionally checks subtasks + that the update email was seen

## API usage (FastAPI)

Open `/docs` for interactive testing.

Typical sequence (medium task):

- `POST /reset` with `{ "task_id": "medium" }`
- `POST /step` with `{ "action": "create_task", "params": {} }`
- `POST /step` with `{ "action": "extract_deadline", "params": {} }`
- `POST /step` with `{ "action": "schedule_task", "params": { "task_id": "<id from observation.extracted_tasks>" } }` (repeat per task)

Hard task update sequence adds:

- `POST /step` with `{ "action": "parse_email", "params": {} }` to load the update email
- `POST /step` with `{ "action": "extract_deadline", "params": {} }`
- `POST /step` with `{ "action": "reschedule_task", "params": { "task_id": "<id>", "new_due_date": "2026-04-22" } }`

## Running locally

See **Quick demo** above.

### OpenAI-powered inference (optional)

`inference.py` reads:

- `API_BASE_URL` (for the environment server; `--api-base` overrides this)
- `MODEL_NAME`
- `HF_TOKEN` (read for compatibility; not required by this baseline)
- `OPENAI_API_KEY`

If `OPENAI_API_KEY` is not set, `inference.py` automatically uses a deterministic baseline policy so everything still runs.

## Docker

```bash
docker build -t smart-email-agent-env .
docker run -p 7860:7860 -e PORT=7860 smart-email-agent-env
```

## Baseline results

The built-in baseline policy is designed to solve all three tasks, so you should typically see near-perfect scores when running:

```bash
python inference.py --api-base http://127.0.0.1:8000
```

