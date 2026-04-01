---
title: Smart Email Agent Env
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---
<<<<<<< HEAD

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
=======
---

title: Smart Email Task & Calendar Agent Environment
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
-------------
>>>>>>> 51df5b5fad006e096acc27555afafe051efb83f7

# Smart Email Task & Calendar Agent Environment

An OpenEnv-compatible environment that simulates how an AI agent processes emails to extract tasks, understand deadlines, and schedule them on a calendar.

---

## Overview

This project models a realistic workflow where users receive emails containing actionable items. The environment allows an AI agent to:

<<<<<<< HEAD
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
=======
* Read and interpret email content
* Extract tasks and associated deadlines
* Break tasks into smaller steps
* Schedule them on a calendar
* Adapt when deadlines change
>>>>>>> 51df5b5fad006e096acc27555afafe051efb83f7

The goal is to provide a structured setup where agent behavior can be evaluated consistently.

---

## Environment Design

<<<<<<< HEAD
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
=======
The environment follows the OpenEnv interface:
>>>>>>> 51df5b5fad006e096acc27555afafe051efb83f7

* `reset()` initializes a task scenario
* `step(action)` applies an action and returns updated state, reward, and status
* `state()` returns the full internal state

The system maintains:

* Extracted tasks
* Parsed deadlines
* Calendar entries
* Current simulated date

---

## Observation Space

Each interaction returns an observation with:

* `email_text`: the current email content
* `extracted_tasks`: identified tasks
* `deadlines`: parsed deadlines
* `calendar`: scheduled events
* `current_date`: simulation date
* `last_action`: last action taken
* `last_action_error`: whether the last action failed

---

## Action Space

The agent can perform the following actions:

* `parse_email`
* `extract_deadline`
* `create_task`
* `split_task`
* `schedule_task`
* `reschedule_task`
* `noop`

Actions are applied sequentially, and the environment enforces logical dependencies between them.

---

## Reward Function

The environment provides incremental rewards:

* Correct task extraction: +0.3
* Correct deadline parsing: +0.3
* Valid scheduling before deadline: +0.4
* Effective rescheduling: +0.5
* Invalid action: -0.2
* Missed deadline: -1.0

This encourages step-by-step reasoning rather than single-step solutions.

---

## Tasks

Three deterministic task scenarios are included:

* **Easy**: Single task with a single deadline
* **Medium**: Multiple tasks with different deadlines
* **Hard**: Multi-step tasks with a deadline update requiring rescheduling

Each task is predefined to ensure reproducible evaluation.

---

## Evaluation

Each scenario has a deterministic grader that produces a score between 0.0 and 1.0 based on:

* Task extraction accuracy
* Deadline correctness
* Calendar scheduling correctness

This ensures consistent benchmarking across different agents.

<<<<<<< HEAD
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
=======
---

## Agent Interaction

The interaction loop follows a standard pattern:

```
reset → observe → act → step → reward → repeat
```

The agent must choose appropriate actions at each step to maximize reward.

---

## Running the Environment

### Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
uvicorn server.app:app --reload
```

---

### Inference

```bash
python inference.py --api-base https://<your-space>.hf.space
```
>>>>>>> 51df5b5fad006e096acc27555afafe051efb83f7

The script uses the OpenAI client if credentials are provided, otherwise it falls back to a deterministic baseline.

---

<<<<<<< HEAD
- `API_BASE_URL` (for the environment server; `--api-base` overrides this)
- `MODEL_NAME`
- `HF_TOKEN` (read for compatibility; not required by this baseline)
- `OPENAI_API_KEY`
=======
## API Endpoints
>>>>>>> 51df5b5fad006e096acc27555afafe051efb83f7

* `POST /reset`
* `POST /step`
* `GET /state`

You can interact with the environment using the `/docs` interface.

---

## Deployment

The project is containerized and deployed on Hugging Face Spaces using Docker.

```bash
<<<<<<< HEAD
docker build -t smart-email-agent-env .
docker run -p 7860:7860 -e PORT=7860 smart-email-agent-env
=======
docker build -t smart-email-env .
docker run -p 7860:7860 smart-email-env
>>>>>>> 51df5b5fad006e096acc27555afafe051efb83f7
```

---

## Baseline Results

<<<<<<< HEAD
```bash
python inference.py --api-base http://127.0.0.1:8000
=======
```
easy:   1.0
medium: 1.0
hard:   1.0
>>>>>>> 51df5b5fad006e096acc27555afafe051efb83f7
```

---

## Key Features

* Realistic email-based task simulation
* Multi-step agent interaction
* Deadline reasoning and adaptation
* Reward shaping for incremental progress
* Deterministic evaluation for reproducibility

---

## Conclusion

This environment provides a structured and realistic framework for evaluating AI agents on email-driven productivity tasks. It balances simplicity with real-world relevance, making it suitable for both experimentation and benchmarking.

---

Built for OpenEnv Hackathon — Round 1

