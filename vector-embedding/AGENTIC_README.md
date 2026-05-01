# Agentic AI Demo — Architecture

## Overview

This section of the workspace shows how a minimal **agentic AI application** works:
a goal-driven loop that selects tools, calls them, accumulates observations in
working memory, and synthesises a final answer.

---

## Architecture Diagram

```mermaid
flowchart TD
    User(["User / Caller"])
    Demo["agentic_ai_demo.py<br>Entrypoint<br>- prints tool list<br>- iterates demo tasks"]
    Agent["agentic_core.py : ShoppingAgent<br>Agent Loop<br>- run(task)<br>- _decide(task, working_memory)<br>- _execute(action, action_input)<br>- _compose_answer(working_memory)"]

    subgraph WorkingMemory["Working Memory (dict)"]
        WM1["search_knowledge_base observation"]
        WM2["search_products observation"]
        WM3["calculate_shipping observation"]
    end

    subgraph Tools["agentic_tools.py - Tool Layer"]
        T1["search_knowledge_base(query)<br>Knowledge Base lookup<br>shipping, returns, bags, headphones"]
        T2["search_products(category, max_price)<br>Product catalogue filter<br>bags: CityLite 14, TravelPro 15, Executive 16<br>headphones: StudioPods"]
        T3["calculate_shipping(weight_kg, expedited)<br>Cost and ETA estimator"]
    end

    FinalAnswer(["Final Answer"])

    User -->|task string| Demo
    Demo -->|task| Agent

    Agent -->|Step 1| T1
    T1 -->|matches| WM1
    WM1 -->|observation stored| Agent

    Agent -->|Step 2| T2
    T2 -->|products| WM2
    WM2 -->|observation stored| Agent

    Agent -->|Step 3 - only if weight in task| T3
    T3 -->|cost + ETA| WM3
    WM3 -->|observation stored| Agent

    Agent -->|action == finish| FinalAnswer
    FinalAnswer --> User
```

---

## File Map

| File | Role |
|---|---|
| `agentic_ai_demo.py` | Entrypoint — runs two demo tasks and prints step-by-step traces |
| `agentic_core.py` | Agent loop — `ShoppingAgent` with `_decide -> _execute -> working memory` cycle |
| `agentic_tools.py` | Tool layer — `search_knowledge_base`, `search_products`, `calculate_shipping` |

---

## Agent Loop — Step by Step

```
+------------------------------------------------------+
|                    Agent Loop                        |
|                                                      |
|  1. Receive task (natural-language goal string)      |
|                                                      |
|  +-----------------------------------------------+  |
|  |  THINK  _decide(task, working_memory)          |  |
|  |   - Inspect what has already been observed    |  |
|  |   - Choose the next tool OR "finish"          |  |
|  +-------------------+--------------------------+  |
|                      |                             |
|  +-------------------v--------------------------+  |
|  |  ACT    _execute(action, action_input)        |  |
|  |   - Call the selected tool function          |  |
|  |   - Returns a structured observation dict   |  |
|  +-------------------+--------------------------+  |
|                      |                             |
|  +-------------------v--------------------------+  |
|  |  OBSERVE  working_memory[action] = result    |  |
|  |   - Store the observation for future steps  |  |
|  +-------------------+--------------------------+  |
|                      |                             |
|        loop back to THINK until action == "finish" |
|                      |                             |
|  +-------------------v--------------------------+  |
|  |  ANSWER  _compose_answer(working_memory)     |  |
|  |   - Synthesise final response from all obs  |  |
|  +----------------------------------------------+  |
+------------------------------------------------------+
```

---

## Running the Demo

```bash
python3 agentic_ai_demo.py
```

No API key or external model is required — the planner uses a rule-based
policy so every step is fully deterministic and inspectable.
