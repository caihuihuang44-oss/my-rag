"""Intent and complexity routing nodes."""

from __future__ import annotations

import time

from app.agents.nodes.intent import IntentClassifier
from app.agents.state import AgentState
from app.core.enums import QueryComplexity, RetrievalStrategy
from app.retrieval.adaptive import AdaptiveRouter
from app.schemas.agent import ComplexityResult
from app.schemas.common import LatencyBreakdown


def _latency(state: AgentState) -> LatencyBreakdown:
    return state.get("latency") or LatencyBreakdown()


def _current_query(state: AgentState) -> str:
    """Follow-ups are classified, routed and retrieved on the resolved query."""
    return state.get("resolved_query") or state["user_query"]


async def intent_node(state: AgentState, classifier: IntentClassifier) -> dict:
    started = time.perf_counter()
    intent = await classifier.classify(_current_query(state))
    latency = _latency(state)
    latency.intent_ms = (time.perf_counter() - started) * 1000
    return {"intent": intent, "latency": latency}


async def router_node(state: AgentState, router: AdaptiveRouter) -> dict:
    started = time.perf_counter()
    intent = state["intent"]
    assert intent is not None
    complexity = (
        ComplexityResult(
            complexity=QueryComplexity.TOOL_REQUIRED,
            strategy=RetrievalStrategy.AGENT,
            confidence=1.0,
            rationale="confirmed_action",
        )
        if state.get("confirmed_action")
        else await router.route(_current_query(state), intent)
    )
    latency = _latency(state)
    latency.routing_ms = (time.perf_counter() - started) * 1000
    return {
        "complexity": complexity,
        "strategy": complexity.strategy,
        "query_complexity": complexity.complexity,
        "latency": latency,
    }
