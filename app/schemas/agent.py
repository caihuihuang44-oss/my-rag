"""Agent planning, intent, complexity and tool-result schemas."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core.enums import IntentLabel, QueryComplexity, RetrievalStrategy, ToolRisk


class ResolvedQuery(BaseModel):
    """Self-contained query produced by the follow-up resolver.

    ``source`` records how the query was produced so routing decisions stay
    debuggable: ``none`` means the raw turn was already complete.
    """

    query: str
    rewritten: bool = False
    source: Literal["none", "llm", "rule"] = "none"
    referenced_turns: int = 0


class IntentResult(BaseModel):
    label: IntentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    prompt_version: str


class ComplexityResult(BaseModel):
    complexity: QueryComplexity
    strategy: RetrievalStrategy
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    name: str
    success: bool
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    risk: ToolRisk = ToolRisk.READ
    truncated: bool = False


class PlanStep(BaseModel):
    step_id: int
    goal: str
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    needs_retrieval: bool = False


class AgentPlan(BaseModel):
    steps: list[PlanStep]
    rationale: str = ""
