---

title: Smart Email Task & Calendar Agent Environment
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---
-------------

# Smart Email Task & Calendar Agent Environment

A fully runnable OpenEnv-compatible environment that simulates how an AI agent processes emails to extract tasks, understand deadlines, and schedule them on a calendar.



## Overview

This project models a realistic workflow where users receive emails containing actionable items. The environment enables an AI agent to:

* Read and interpret email content
* Extract actionable tasks
* Parse deadlines (e.g., “15 April 2026”)
* Break complex tasks into subtasks
* Schedule tasks on a calendar
* Adapt to deadline updates through rescheduling

The goal is to provide a structured and reproducible setup for evaluating agent behavior.

---

## Environment Design

The environment follows the OpenEnv interface:

* `reset(task_id)` → initializes a task scenario
* `step(action)` → applies an action and returns `(observation, reward, done, info)`
* `state()` → returns the full internal state

### Internal State

* Extracted tasks
* Parsed deadlines
* Calendar entries
* Current simulated date

Time advances deterministically by **+1 day per step**, enabling deadline-based evaluation.

---

## Observation Space

Each step returns:

* `email_text`: current email content
* `extracted_tasks`: identified tasks
* `deadlines`: parsed deadlines
* `calendar`: scheduled events
* `current_date`: simulation date
* `last_action`: last action taken
* `last_action_error`: whether the last action failed

---

## Action Space

The agent can perform:

* `parse_email`
* `extract_deadline`
* `create_task`
* `split_task`
* `schedule_task`
* `reschedule_task`
* `noop`

### Action Notes

* `create_task` performs task extraction from the email
* `extract_deadline` attaches deadlines to tasks
* `split_task` generates subtasks (used in complex scenarios)
* `schedule_task` requires a target task (`task_id` or `task_title`)
* `parse_email` loads the next email (used in update scenarios)
* `reschedule_task` updates task timing after deadline changes

---

## Reward Function

The environment provides incremental rewards:

* Task extraction: +0.3
* Deadline parsing: +0.3
* Valid scheduling: +0.4
* Effective rescheduling: +0.5
* Invalid action: -0.2
* Missed deadline: -1.0

This encourages structured, step-by-step reasoning.

---

## Tasks

Three deterministic scenarios are provided:

* **Easy**: Single task and deadline
* **Medium**: Multiple tasks with independent deadlines
* **Hard**: Multi-step tasks with a deadline update requiring rescheduling

Each scenario is predefined to ensure reproducible evaluation.

---

## Evaluation

Each task includes a deterministic grader returning a score between 0.0 and 1.0 based on:

* Task extraction accuracy
* Deadline correctness
* Calendar scheduling correctness

---

## Agent Interaction

The agent operates in a loop:

```
reset → observe → act → step → reward → repeat
```

The objective is to maximize cumulative reward by selecting appropriate actions.

---

## API Usage

You can interact with the environment via Swagger UI:

```
/docs
```

### Example Flow (Medium Task)

1. `POST /reset`

```json
{ "task_id": "medium" }
```

2. Extract tasks

```json
{ "action": "create_task" }
```

3. Extract deadlines

```json
{ "action": "extract_deadline" }
```

4. Schedule tasks

```json
{
  "action": "schedule_task",
  "params": { "task_id": "<task_id>" }
}
```

---

### Hard Task (Deadline Update Flow)

1. Process initial email
2. Load update email:

```json
{ "action": "parse_email" }
```

3. Re-extract deadline:

```json
{ "action": "extract_deadline" }
```

4. Reschedule:

```json
{
  "action": "reschedule_task",
  "params": {
    "task_id": "<task_id>",
    "new_due_date": "2026-04-22"
  }
}
```

---

## Running Locally

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
uvicorn server.app:app --host 127.0.0.1 --port 8000
```

---

## Inference

```bash
python inference.py --api-base http://127.0.0.1:8000
```

The script supports OpenAI-based inference and falls back to a deterministic baseline if no API key is provided.

---

## API Endpoints

* `POST /reset`
* `POST /step`
* `GET /state`

---

## Deployment

The project is containerized for Hugging Face Spaces:

```bash
docker build -t smart-email-agent-env .
docker run -p 7860:7860 -e PORT=7860 smart-email-agent-env
```

---

## Baseline Results

```
easy:   1.0
medium: 1.0
hard:   1.0
```

---

## Key Features

* Real-world email workflow simulation
* Multi-step agent reasoning
* Deadline understanding and adaptation
* Reward shaping for incremental progress
* Deterministic evaluation and reproducibility

---

## Conclusion

This environment provides a structured and realistic framework for evaluating AI agents on email-driven productivity tasks. It balances real-world relevance with controlled evaluation, making it suitable for both experimentation and benchmarking.

---

Built for OpenEnv Hackathon — Round 1


