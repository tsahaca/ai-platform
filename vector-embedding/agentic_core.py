"""A tiny agent loop that demonstrates agentic AI structure.

This is deliberately simple:
- inspect the user goal
- decide the next tool to call
- store observations in working memory
- produce a final answer after enough evidence is gathered

It uses a rule-based policy so the example stays runnable without an API key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from agentic_tools import calculate_shipping, search_knowledge_base, search_products


@dataclass
class AgentStep:
    thought: str
    action: str
    action_input: dict[str, Any]
    observation: dict[str, Any]


@dataclass
class AgentRun:
    task: str
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str = ""


class ShoppingAgent:
    def run(self, task: str) -> AgentRun:
        run = AgentRun(task=task)
        working_memory: dict[str, Any] = {}

        while True:
            next_action = self._decide(task, working_memory)
            if next_action["action"] == "finish":
                run.final_answer = self._compose_answer(task, working_memory)
                return run

            observation = self._execute(next_action["action"], next_action["action_input"])
            run.steps.append(
                AgentStep(
                    thought=next_action["thought"],
                    action=next_action["action"],
                    action_input=next_action["action_input"],
                    observation=observation,
                )
            )
            working_memory[next_action["action"]] = observation

    def _decide(self, task: str, working_memory: dict[str, Any]) -> dict[str, Any]:
        if "search_knowledge_base" not in working_memory:
            return {
                "thought": "I should gather domain guidance before choosing a product.",
                "action": "search_knowledge_base",
                "action_input": {"query": task},
            }

        if "search_products" not in working_memory:
            return {
                "thought": "I now have constraints, so I should find matching products.",
                "action": "search_products",
                "action_input": {
                    "category": self._infer_category(task),
                    "max_price": self._extract_budget(task),
                },
            }

        if self._extract_weight(task) is not None and "calculate_shipping" not in working_memory:
            return {
                "thought": "The user asked about shipping, so I should estimate delivery cost.",
                "action": "calculate_shipping",
                "action_input": {
                    "weight_kg": self._extract_weight(task),
                    "expedited": self._wants_expedited(task),
                },
            }

        return {
            "thought": "I have enough evidence to answer.",
            "action": "finish",
            "action_input": {},
        }

    def _execute(self, action: str, action_input: dict[str, Any]) -> dict[str, Any]:
        if action == "search_knowledge_base":
            return search_knowledge_base(**action_input)
        if action == "search_products":
            return search_products(**action_input)
        if action == "calculate_shipping":
            return calculate_shipping(**action_input)
        raise ValueError(f"Unknown action: {action}")

    def _compose_answer(self, task: str, working_memory: dict[str, Any]) -> str:
        products = working_memory["search_products"]["products"]
        if not products:
            return "I could not find a matching product under the current constraints. Try raising the budget or broadening the category."

        best_product = products[0]
        answer_lines = [
            f"Recommended product: {best_product['name']} at ${best_product['price']:.2f}.",
            f"Why: it fits the inferred category '{best_product['category']}' and is the lowest-priced match.",
            f"Features: {', '.join(best_product['features'])}.",
        ]

        kb_result = working_memory.get("search_knowledge_base", {})
        kb_matches = kb_result.get("matches", [])
        if kb_matches:
            answer_lines.append(f"Policy guidance: {kb_matches[0]['summary']}")

        shipping = working_memory.get("calculate_shipping")
        if shipping:
            answer_lines.append(
                "Shipping estimate: "
                f"${shipping['shipping_cost_usd']:.2f}, {shipping['estimated_delivery']}."
            )

        answer_lines.append(f"Original request: {task}")
        return "\n".join(answer_lines)

    def _extract_budget(self, task: str) -> float | None:
        match = re.search(r"\$(\d+(?:\.\d+)?)", task)
        return float(match.group(1)) if match else None

    def _extract_weight(self, task: str) -> float | None:
        match = re.search(r"(\d+(?:\.\d+)?)\s*kg", task.lower())
        return float(match.group(1)) if match else None

    def _infer_category(self, task: str) -> str:
        lowered = task.lower()
        if "bag" in lowered or "backpack" in lowered:
            return "bag"
        if "headphone" in lowered:
            return "headphones"
        return "bag"

    def _wants_expedited(self, task: str) -> bool:
        lowered = task.lower()
        return "expedited" in lowered or "fast shipping" in lowered