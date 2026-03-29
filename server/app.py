from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException

from env.environment import SmartEmailTaskCalendarEnv
from env.models import Action, Observation

app = FastAPI(title="Smart Email Task & Calendar Agent Environment")
env = SmartEmailTaskCalendarEnv()


@app.post("/reset", response_model=Observation)
def reset(payload: dict | None = None):
    task_id = payload.get("task_id") if payload else None
    try:
        return env.reset(task_id=task_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    return env.state()


def main() -> None:
    """
    Entrypoint for OpenEnv / Spaces style deployments.
    """
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()