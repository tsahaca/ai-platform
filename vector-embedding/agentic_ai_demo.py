"""Runnable demo showing how an agentic AI application works.

This file prints the main pieces of an agent workflow:
1. goal intake
2. reasoning / decision step
3. tool use
4. observation storage
5. final answer synthesis
"""

from __future__ import annotations

from pprint import pformat

from agentic_core import ShoppingAgent
from agentic_tools import available_tools


DEMO_TASKS = [
    "Recommend a laptop bag under $80 and estimate expedited shipping for 2 kg.",
    "Find headphones under $120.",
]


def print_tools() -> None:
    print("Available tools:")
    for tool in available_tools():
        print(f"- {tool.name}: {tool.description}")


def run_demo(task: str) -> None:
    print("=" * 72)
    print("Agentic AI Demo")
    print("=" * 72)
    print(f"Task: {task}\n")

    agent = ShoppingAgent()
    run = agent.run(task)

    for index, step in enumerate(run.steps, start=1):
        print(f"[Step {index}] Thought")
        print(f"  {step.thought}")
        print(f"[Step {index}] Action")
        print(f"  {step.action}({pformat(step.action_input)})")
        print(f"[Step {index}] Observation")
        print(f"  {pformat(step.observation)}\n")

    print("Final answer")
    print("-" * 72)
    print(run.final_answer)
    print()


def main() -> None:
    print_tools()
    print()
    for task in DEMO_TASKS:
        run_demo(task)


if __name__ == "__main__":
    main()