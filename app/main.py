"""
Perspecta - Single-Process FastAPI Multi-Agent Orchestration & Benchmarking Engine
Routes:
- GET /                     -> Serves static/index.html
- GET /api/health           -> Returns active sandbox backend (docker|subprocess) and system health
- POST /api/run/{scenario} -> Starts scenario run and returns run_id + ws_url
- GET /api/benchmarks      -> Returns aggregated benchmark KPIs computed from JSONL logs
- WS /ws/{run_id}          -> Streams live server-pushed WSEvent sequence
"""

import asyncio
import time
import os
import uuid
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import HealthResponse, ScenarioRunResponse, BenchmarkKPIs
from app.sandbox import check_docker_availability
from app.memory import FastMemoryStore, VectorMemoryStore
from app.scenarios import (
    RunEventQueue,
    run_collab_engineering,
    run_consensus_debate,
    run_long_horizon_memory,
    run_dynamic_orchestration
)
from app.marl_rewards import run_marl_scenario
from app.benchmarking import log_run_metrics, get_aggregated_benchmarks

app = FastAPI(
    title="Perspecta",
    description="FYP Multi-Agent Coordination Mechanisms Verification Platform",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active WebSocket event queues indexed by run_id
ACTIVE_QUEUES: Dict[str, RunEventQueue] = {}


@app.get("/api/health", response_model=HealthResponse)
async def get_health():
    """
    Returns active sandbox backend status (docker vs subprocess fallback) and component availability.
    """
    docker_ok, docker_msg = check_docker_availability()
    backend = "docker" if docker_ok else "subprocess"
    isolation = "strong" if docker_ok else "weak"

    # Fast memory redis check
    fm = FastMemoryStore()
    redis_ok = (fm.backend == "redis")

    # Chroma DB check
    try:
        vm = VectorMemoryStore()
        chroma_ok = True
    except Exception:
        chroma_ok = False

    return HealthResponse(
        status="ok",
        active_backend=backend,
        isolation=isolation,
        provider="mock",
        docker_available=docker_ok,
        redis_available=redis_ok,
        chroma_available=chroma_ok
    )


@app.get("/api/benchmarks", response_model=BenchmarkKPIs)
async def get_benchmarks():
    """
    Exposes aggregated KPIs calculated from real execution log history.
    """
    return get_aggregated_benchmarks()


async def execute_scenario_task(run_id: str, scenario_name: str, event_queue: RunEventQueue):
    """
    Background worker task executing scenario and streaming events.
    """
    start_ts = time.time()
    result_metrics: Dict[str, Any] = {}
    simulated_events = 0
    real_events = 0

    try:
        if scenario_name == "collab-engineering":
            result_metrics = await run_collab_engineering(run_id, event_queue)
        elif scenario_name == "consensus-debate":
            result_metrics = await run_consensus_debate(run_id, event_queue)
        elif scenario_name == "long-horizon-memory":
            result_metrics = await run_long_horizon_memory(run_id, event_queue)
        elif scenario_name == "dynamic-orchestration":
            result_metrics = await run_dynamic_orchestration(run_id, event_queue)
        elif scenario_name == "marl-reward-shaping":
            result_metrics = await run_marl_scenario(run_id, event_queue)
        else:
            await event_queue.emit("error", {"message": f"Unknown scenario: {scenario_name}"}, simulated=False)
            return

        duration = time.time() - start_ts

        # Log metrics to JSON lines file
        log_data = {
            "run_id": run_id,
            "scenario": scenario_name,
            "duration_sec": round(duration, 3),
            "iterations": result_metrics.get("iterations", 1),
            "sandbox_pass_count": result_metrics.get("sandbox_pass_count", 0),
            "sandbox_fail_count": result_metrics.get("sandbox_fail_count", 0),
            "simulated_events": simulated_events,
            "real_events": real_events,
            "extra": result_metrics
        }
        log_run_metrics(log_data)

        # Emit aggregated benchmark update event live
        updated_kpis = get_aggregated_benchmarks()
        await event_queue.emit("benchmark_update", updated_kpis.model_dump(), simulated=False)

        # Emit run complete event
        await event_queue.emit(
            "run_complete",
            {
                "run_id": run_id,
                "scenario": scenario_name,
                "duration_sec": round(duration, 3),
                "status": "COMPLETED"
            },
            simulated=False
        )

    except Exception as ex:
        await event_queue.emit("error", {"message": f"Scenario execution failed: {str(ex)}"}, simulated=False)
    finally:
        # Keep queue alive briefly for WS client consumption
        await asyncio.sleep(5)
        ACTIVE_QUEUES.pop(run_id, None)


@app.post("/api/run/{scenario_name}", response_model=ScenarioRunResponse)
async def start_scenario_run(scenario_name: str):
    """
    Launches a scenario execution and returns run_id + WebSocket streaming URL.
    """
    valid_scenarios = [
        "collab-engineering",
        "consensus-debate",
        "long-horizon-memory",
        "dynamic-orchestration",
        "marl-reward-shaping"
    ]
    if scenario_name not in valid_scenarios:
        raise HTTPException(status_code=400, detail=f"Invalid scenario name. Must be one of {valid_scenarios}")

    run_id = f"run-{uuid.uuid4().hex[:8]}"
    event_queue = RunEventQueue(run_id=run_id)
    ACTIVE_QUEUES[run_id] = event_queue

    # Start scenario execution in background
    asyncio.create_task(execute_scenario_task(run_id, scenario_name, event_queue))

    return ScenarioRunResponse(
        run_id=run_id,
        scenario=scenario_name,
        status="started",
        ws_url=f"/ws/{run_id}"
    )


@app.websocket("/ws/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: str):
    """
    Server-pushed WebSocket event endpoint for real-time scenario telemetry.
    """
    await websocket.accept()
    queue_wrapper = ACTIVE_QUEUES.get(run_id)

    if not queue_wrapper:
        await websocket.send_json({
            "seq": 0,
            "ts": time.time(),
            "run_id": run_id,
            "simulated": False,
            "type": "error",
            "payload": {"error": "Run ID not found or already completed."}
        })
        await websocket.close()
        return

    try:
        while True:
            # Wait for next event from queue
            event = await queue_wrapper.queue.get()
            await websocket.send_json(event.model_dump())
            if event.type == "run_complete" or event.type == "error":
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({
                "seq": 9999,
                "ts": time.time(),
                "run_id": run_id,
                "simulated": False,
                "type": "error",
                "payload": {"error": str(e)}
            })
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# Mount static assets
static_path = os.path.join(os.path.dirname(__file__), "..", "static")
os.makedirs(static_path, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_path), name="static")


@app.get("/")
def read_root():
    """
    Serves the main single-page web UI dashboard.
    """
    index_file = os.path.join(static_path, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Perspecta API Server running. static/index.html not found."}
