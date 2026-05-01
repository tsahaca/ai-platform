"""Small toolset used by the agentic AI demo.

The goal here is not to simulate a full production tool layer.
It is to provide a few concrete actions an agent can choose from:
search knowledge, search products, and compute shipping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str


PRODUCTS = [
    {
        "name": "CityLite 14",
        "category": "bag",
        "price": 49.0,
        "weight_kg": 0.9,
        "features": ["lightweight", "laptop", "commute"],
    },
    {
        "name": "TravelPro 15",
        "category": "bag",
        "price": 79.0,
        "weight_kg": 1.2,
        "features": ["laptop", "travel", "water-resistant"],
    },
    {
        "name": "Executive 16",
        "category": "bag",
        "price": 129.0,
        "weight_kg": 1.5,
        "features": ["premium", "laptop", "leather"],
    },
    {
        "name": "StudioPods",
        "category": "headphones",
        "price": 99.0,
        "weight_kg": 0.4,
        "features": ["wireless", "music", "noise-cancelling"],
    },
]

KNOWLEDGE_BASE = {
    "shipping": "Standard shipping takes 3-5 business days. Expedited shipping takes 1-2 business days.",
    "returns": "Unused products can be returned within 30 days for a full refund.",
    "bags": "Laptop bags are best filtered by budget, laptop size, and travel or commute needs.",
    "headphones": "Headphone recommendations usually depend on budget, portability, and whether noise cancellation matters.",
}

TOPIC_KEYWORDS = {
    "shipping": {"shipping", "delivery", "expedited"},
    "returns": {"return", "returns", "refund"},
    "bags": {"bag", "bags", "backpack", "laptop"},
    "headphones": {"headphone", "headphones", "audio", "music"},
}


def available_tools() -> list[ToolSpec]:
    return [
        ToolSpec("search_knowledge_base", "Look up policy or buying guidance."),
        ToolSpec("search_products", "Find products that match category and budget constraints."),
        ToolSpec("calculate_shipping", "Estimate shipping cost from weight and speed."),
    ]


def search_knowledge_base(query: str) -> dict[str, Any]:
    query_words = {word.strip(".,?!").lower() for word in query.split()}
    matches = []
    for topic, text in KNOWLEDGE_BASE.items():
        if query_words & TOPIC_KEYWORDS.get(topic, set()):
            matches.append({"topic": topic, "summary": text})
    if not matches:
        matches.append({"topic": "general", "summary": "No exact article matched, so rely on product search and explicit constraints."})
    return {"matches": matches}


def search_products(category: str, max_price: float | None = None) -> dict[str, Any]:
    results = [product for product in PRODUCTS if product["category"] == category]
    if max_price is not None:
        results = [product for product in results if product["price"] <= max_price]
    results.sort(key=lambda item: (item["price"], item["weight_kg"]))
    return {"count": len(results), "products": results}


def calculate_shipping(weight_kg: float, expedited: bool = False) -> dict[str, Any]:
    base_cost = 4.99 + (weight_kg * 2.25)
    if expedited:
        base_cost += 6.5
        eta = "1-2 business days"
    else:
        eta = "3-5 business days"
    return {
        "shipping_cost_usd": round(base_cost, 2),
        "estimated_delivery": eta,
        "expedited": expedited,
    }