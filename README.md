# Smart Email Task & Calendar Agent Environment

An **OpenEnv-compatible** environment for a hackathon where an agent:

- reads email text
- extracts actionable tasks
- parses deadlines
- splits tasks into subtasks
- schedules tasks onto a calendar
- handles deadline updates via rescheduling

This repo includes a FastAPI server (`/reset`, `/step`, `/state`), three tasks (easy/medium/hard), deterministic graders, and an `inference.py` runner that can use the OpenAI client (with a baseline fallback so the project is runnable without keys).

## Environment design

The environment is implemented in `env/environment.py` as `SmartEmailTaskCalendarEnv` and follows the OpenEnv-style API:

- `reset(task_id: Optional[str]) -> Observation`
- `step(action) -> (observation, reward, done, info)`
- `state() -> dict`

Internal state tracked:

- **tasks**: extracted tasks (with optional subtasks + due dates)
- **deadlines**: parsed per-task deadlines (source: email/update)
- **calendar**: scheduled calendar events

Time advances deterministically by **+1 day per step** to enable deadline-miss penalties.

## Action space

Actions are strings (Pydantic enum in `env/models.py`):

- `parse_email`
- `extract_deadline`
- `create_task`
- `split_task`
- `schedule_task`
- `reschedule_task`
- `noop`

Each action accepts an optional `params` dict (e.g. `task_id`, `task_title`, `new_due_date`, `start_date`).

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

## Running locally

From `smart-email-agent-env/`:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn server.app:app --reload
```

In a second terminal:

```bash
python inference.py
```

### OpenAI-powered inference (optional)

`inference.py` reads:

- `API_BASE_URL` (also used for OpenAI base_url if provided)
- `MODEL_NAME`
- `HF_TOKEN` (read for compatibility; not required by this baseline)
- `OPENAI_API_KEY`

If `OPENAI_API_KEY` is not set, `inference.py` automatically uses a deterministic baseline policy so everything still runs.

## Docker

```bash
docker build -t smart-email-agent-env .
docker run -p 8000:8000 smart-email-agent-env
```

## Baseline results

The built-in baseline policy is designed to solve all three tasks, so you should typically see near-perfect scores when running:

```bash
python inference.py
```

