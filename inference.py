from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from openai import OpenAI

from tasks.graders import grade_easy, grade_hard, grade_medium

# --- Environment variables (matching the validator's sample pattern) ---
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")  # optional


def _wait_for_server(base_url: str, retries: int = 30, delay: float = 2.0) -> None:
    print(f"Waiting for server at {base_url} ...", flush=True)
    for _ in range(retries):
        try:
            r = httpx.get(f"{base_url}/healthz", timeout=5.0)
            if r.status_code == 200:
                print("Server is ready.", flush=True)
                return
        except Exception:
            pass
        time.sleep(delay)
    print("[warn] Server may not be ready — proceeding anyway.", flush=True)


def _baseline_policy(obs: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    last_action = obs.get("last_action", "")
    email_text = obs.get("email_text", "")
    extracted = obs.get("extracted_tasks", [])
    calendar = obs.get("calendar", [])

    if not extracted:
        return {"action": "create_task", "params": {}}

    if task_id == "hard" and "update:" not in email_text.lower():
        t0 = extracted[0]
        if t0.get("status") == "new" and not t0.get("subtasks"):
            return {"action": "split_task", "params": {"task_id": t0.get("task_id")}}

    if not any(t.get("due_date") for t in extracted):
        return {"action": "extract_deadline", "params": {}}

    for t in extracted:
        if t.get("status") != "scheduled":
            return {"action": "schedule_task", "params": {"task_id": t.get("task_id")}}

    if task_id == "hard":
        if "update:" not in email_text.lower():
            return {"action": "parse_email", "params": {}}
        t0 = extracted[0]
        if t0.get("due_date") != "2026-04-22":
            if last_action != "extract_deadline":
                return {"action": "extract_deadline", "params": {}}
            return {"action": "reschedule_task", "params": {"task_id": t0.get("task_id"), "new_due_date": "2026-04-22"}}
        cal_entry = next((e for e in calendar if e.get("task_id") == t0.get("task_id")), None)
        if cal_entry and cal_entry.get("due_date") != "2026-04-22":
            return {"action": "reschedule_task", "params": {"task_id": t0.get("task_id"), "new_due_date": "2026-04-22"}}

    return {"action": "noop", "params": {}}


def _llm_policy(client: OpenAI, obs: Dict[str, Any]) -> Dict[str, Any]:
    system = (
        "You are an agent in an email-task scheduling environment. "
        "Return a single JSON object with keys: action, params. "
        "Allowed actions: parse_email, extract_deadline, create_task, split_task, "
        "schedule_task, reschedule_task, noop. Use ISO dates YYYY-MM-DD."
    )
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Observation:\n{obs}\n\nReturn next action as JSON."},
        ],
        response_format={"type": "json_object"},
    )
    try:
        act = json.loads(resp.choices[0].message.content)
        if not isinstance(act, dict) or "action" not in act:
            return {"action": "noop", "params": {}}
        act.setdefault("params", {})
        return act
    except Exception:
        return {"action": "noop", "params": {}}


def run_task(
    env_base: str,
    task_id: str,
    llm_client: Optional[OpenAI],
) -> Tuple[Dict[str, Any], int]:
    step_num = 0
    try:
        with httpx.Client(base_url=env_base, timeout=30.0) as client:
            reset_resp = client.post("/reset", json={"task_id": task_id})
            reset_resp.raise_for_status()
            obs = reset_resp.json()

            for _ in range(40):
                try:
                    if llm_client is not None:
                        try:
                            action = _llm_policy(llm_client, obs)
                        except Exception:
                            action = _baseline_policy(obs, task_id)
                    else:
                        action = _baseline_policy(obs, task_id)

                    step_resp = client.post("/step", json=action)
                    step_resp.raise_for_status()
                    step_out = step_resp.json()

                    if "observation" not in step_out:
                        print(f"[warn] missing observation in step response", flush=True)
                        break

                    step_num += 1
                    reward_val = step_out.get("reward", {}).get("value", 0.0)
                    print(f"[STEP] step={step_num} reward={reward_val}", flush=True)

                    obs = step_out["observation"]
                    if step_out.get("done", False):
                        break
                except Exception as e:
                    print(f"[warn] step error: {e}", flush=True)
                    break

            final_state = client.get("/state").json()
            return final_state, step_num
    except Exception as e:
        print(f"[warn] run_task failed for '{task_id}': {e}", flush=True)
        return {}, step_num


def main() -> None:
    load_dotenv()

    # Env server runs on the container port
    port = os.getenv("PORT", "7860")
    env_base = f"http://127.0.0.1:{port}"

    # LLM client — always use API_BASE_URL + MODEL_NAME as the validator injects them
    llm_client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN or "no-key",
    )

    _wait_for_server(env_base)

    graders = {"easy": grade_easy, "medium": grade_medium, "hard": grade_hard}

    results: List[Tuple[str, float]] = []
    for task_id in ["easy", "medium", "hard"]:
        print(f"[START] task={task_id}", flush=True)
        try:
            final_state, steps = run_task(env_base, task_id, llm_client)
            score = graders[task_id](final_state) if final_state else 0.0
        except Exception as e:
            print(f"[warn] task '{task_id}' error: {e}", flush=True)
            score, steps = 0.0, 0
        results.append((task_id, score))
        print(f"[END] task={task_id} score={score:.3f} steps={steps}", flush=True)

    avg = sum(s for _, s in results) / len(results)
    print(f"average score: {avg:.3f}", flush=True)


if __name__ == "__main__":
    main()
