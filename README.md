---
title: Smart Email Agent Env
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---
---

title: Smart Email Task & Calendar Agent Environment
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
-------------

# Smart Email Task & Calendar Agent Environment

An OpenEnv-compatible environment that simulates how an AI agent processes emails to extract tasks, understand deadlines, and schedule them on a calendar.

---

## Overview

This project models a realistic workflow where users receive emails containing actionable items. The environment allows an AI agent to:

* Read and interpret email content
* Extract tasks and associated deadlines
* Break tasks into smaller steps
* Schedule them on a calendar
* Adapt when deadlines change

The goal is to provide a structured setup where agent behavior can be evaluated consistently.

---

## Environment Design

The environment follows the OpenEnv interface:

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

The script uses the OpenAI client if credentials are provided, otherwise it falls back to a deterministic baseline.

---

## API Endpoints

* `POST /reset`
* `POST /step`
* `GET /state`

You can interact with the environment using the `/docs` interface.

---

## Deployment

The project is containerized and deployed on Hugging Face Spaces using Docker.

```bash
docker build -t smart-email-env .
docker run -p 7860:7860 smart-email-env
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

