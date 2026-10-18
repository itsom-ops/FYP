# Perspecta — Multi-Agent Coordination Mechanisms & Verification Platform

Perspecta is a single-process FastAPI web application demonstrating the core coordination, verification, memory, consensus, and benchmarking mechanisms behind an academic FYP (Final Year Project) proposal.

> [!NOTE]
> This repository is a review-1 demonstration of **coordination mechanisms** and execution primitives. It is designed to run 100% offline with zero external API key requirements.

---

## Objective Coverage Table

| # | FYP Objective | Scenario | Status shown in UI |
|---|---|---|---|
| 1 | Dynamic orchestration, real-time role recruitment | dynamic-orchestration | Real |
| 2 | SOPs / shared message pool, stop cascading hallucinations | collab-engineering | Real |
| 3 | Long-horizon tiered memory | long-horizon-memory | Real |
| 4 | Theory of Mind / belief tracking | (embedded in collab-engineering + dynamic-orchestration) | Real graph state, templated language |
| 5 | Multi-round consensus, heterogeneous models, confidence scaling | consensus-debate | Simulated confidence signal |
| 6 | LLM-generated dense rewards for MARL sample efficiency | marl-reward-shaping | Real toy measurement, illustrative scale |
| 7 | Benchmarking framework, milestone KPIs | benchmark dashboard | Real, computed from logs |

---

## Technical Architecture & Real vs Simulated Primitives

### Real Primitives (Executed & Measured Live)
1. **Code Sandbox Execution (`app/sandbox.py`)**: Spawns isolated Docker containers (`network_mode="none"`, `mem_limit="256m"`, `pids_limit=64`, `read_only=True`, non-root, 10s timeout) to evaluate python code and capture real exit codes. If Docker daemon is unavailable, falls back gracefully to `SubprocessSandbox` emitting `isolation="weak"` (AMBER badge in UI).
2. **Tiered Long-Horizon Memory (`app/memory.py`)**: Stores and retrieves constraints using ChromaDB (vector similarity disk persistence) vs Fast Memory (Redis or local low-latency dict fallback). Real retrieval duration in milliseconds is measured and displayed.
3. **Dynamic Graph Recompilation (`app/scenarios.py`)**: LangGraph workflow execution with runtime supervisor recompilation. Upon injected failure, the supervisor mutates node routing, adds a recovery node, and resumes state cleanly across the compiled graph boundaries.
4. **Toy MARL Grid-world (`app/marl_rewards.py`)**: 5x5 tabular Q-learning grid-world with 2 agents. Evaluates sparse goal rewards vs LLM potential-based dense reward shaping. Measures real episodes-to-convergence and calculates true % improvement.
5. **JSON-L Benchmarking Dashboard (`app/benchmarking.py`)**: Logged run telemetry is written to disk as structured JSON lines (`data/benchmark_logs.jsonl`). Aggregated KPIs are calculated directly from run history logs.
6. **Server-Pushed WebSocket Stream (`app/main.py`)**: Bounded `asyncio.Queue(1000)` per client run streaming live events. Drops oldest events with explicit error notifications on buffer overflow.

### Simulated Primitives (Tagged & Labeled)
1. **Agent Personas (`MockProvider`)**: Mock LLM personas ("Llama", "ERNIE", "Coder", "Reviewer", "Tester"). Every generated event includes `simulated: true` and a **SIM** tag in the console.
2. **Confidence-Weighted Debate**: Simulated confidence scores attached to personas. The UI and code explicitly disclaim:
   *`"Simulated confidence signal — hosted LLM APIs (Anthropic/OpenAI) do not expose logits; a logit-derived version would require a self-hosted open-weight backend (e.g. vLLM)."`*
3. **Theory of Mind / Belief Tracking**: Belief states rendered via structured templates backed by real graph runtime state.
4. **Sample Efficiency Baseline**: The 185.9% efficiency figure from literature is **not hardcoded or claimed**. The UI displays the real measured toy gridworld percentage improvement with the disclaimer:
   *`"Illustrative toy result on a synthetic gridworld — not the benchmark referenced in the literature review."`*

---

## Deviations & Limitations
1. **LangGraph Graph Mutation**: LangGraph compiled `StateGraph` objects are immutable once compiled. Runtime node dynamic injection is achieved by state extraction and re-compiling a modified supervisor graph topology across step boundaries.
2. **Docker vs Subprocess Sandbox**: When Docker daemon is not active on host machine, execution falls back to `SubprocessSandbox` with resource guards, marking isolation level as `weak` (amber badge).
3. **Redis Fallback**: In environments without a running Redis server, `FastMemoryStore` automatically uses a high-precision, in-memory dictionary cache while preserving real latency measurements in milliseconds.
4. **Milestone KPIs Scope**: The benchmarking suite computes metrics over actual execution logs (task success, debate iterations, memory latency, graph mutations, MARL convergence). Full competitive/emergent multi-agent benchmarks are left for future work.

---

## Quick Start / How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Perspecta Server
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 3. Open Web Dashboard
Open your browser and navigate to:
```
http://localhost:8000
```

No external API keys or internet connection required.
