"""Follow-up resolution: turn a context-dependent question into a self-contained one.

Why this node exists: before it, every downstream node read ``user_query`` only, so a
short follow-up ("那运费谁出") lost the topic of the previous turn. The resolver keeps
``user_query`` untouched (audit + write-intent detection) and exposes
``resolved_query`` for intent, routing, retrieval, planning and generation.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from app.agents.state import AgentState
from app.core.constants import PROMPT_VERSION_FOLLOWUP
from app.core.exceptions import LLMOutputError, LLMProviderError
from app.prompts.followup import FOLLOWUP_SYSTEM, FOLLOWUP_USER
from app.providers.base import LLMProvider
from app.providers.fake import OfflineLLMProvider
from app.providers.json_parser import parse_model
from app.schemas.agent import ResolvedQuery
from app.schemas.chat import ChatMessage

logger = logging.getLogger(__name__)

MAX_FOLLOWUP_CHARS = 12
HISTORY_TURNS = 6
FOLLOWUP_CUES = (
    "那",
    "这个",
    "那个",
    "它",
    "它们",
    "还有",
    "呢",
    "换成",
    "改成",
    "改一下",
    "另一个",
    "同样",
    "上面",
    "刚才",
    "这款",
    "这件",
)


class _RewritePayload(BaseModel):
    query: str


def rule_based_resolution(query: str, history: list[ChatMessage]) -> ResolvedQuery:
    """Conservative offline fallback for short follow-ups.

    It never guesses slots: it only re-attaches the previous user turn so BM25 and
    dense retrieval still see the topic. Longer questions are left untouched, which
    keeps precision high when there is no LLM to rewrite with.
    """
    previous = next((m.content for m in reversed(history) if m.role == "user"), "")
    text = query.strip()
    if not previous or previous.strip() == text:
        return ResolvedQuery(query=query)
    if len(text) <= MAX_FOLLOWUP_CHARS and any(cue in text for cue in FOLLOWUP_CUES):
        return ResolvedQuery(
            query=f"{previous.strip()}；{text}",
            rewritten=True,
            source="rule",
            referenced_turns=1,
        )
    return ResolvedQuery(query=query)


def _history_block(history: list[ChatMessage]) -> str:
    turns = [m for m in history if m.role in {"user", "assistant"}][-HISTORY_TURNS:]
    return "\n".join(f"{m.role}: {m.content[:200]}" for m in turns) or "(无)"


async def resolve_query(
    query: str,
    *,
    history: list[ChatMessage],
    summary: str,
    llm: LLMProvider,
) -> ResolvedQuery:
    """Rewrite a follow-up into a standalone query, or return it unchanged.

    First turn costs no LLM call at all. Any provider/parse failure degrades to the
    rule fallback instead of dropping the turn.
    """
    if not history:
        return ResolvedQuery(query=query)
    if isinstance(llm, OfflineLLMProvider):
        return rule_based_resolution(query, history)
    try:
        response = await llm.generate(
            [
                {"role": "system", "content": FOLLOWUP_SYSTEM},
                {
                    "role": "user",
                    "content": FOLLOWUP_USER.format(
                        prompt_version=PROMPT_VERSION_FOLLOWUP,
                        summary=(summary or "无")[:600],
                        history=_history_block(history),
                        query=query,
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        rewritten = parse_model(response.content, _RewritePayload).query.strip()
        if len(rewritten) >= 2:
            return ResolvedQuery(
                query=rewritten,
                rewritten=rewritten != query.strip(),
                source="llm",
                referenced_turns=1,
            )
        logger.warning("followup_rewrite_blank")
    except (LLMProviderError, LLMOutputError):
        logger.warning("followup_rewrite_failed")
    return rule_based_resolution(query, history)


async def prepare_context_node(state: AgentState, *, llm: LLMProvider) -> dict:
    """Resolve the current turn against the loaded conversation history."""
    raw = state["user_query"]
    if state.get("confirm_action_id") or state.get("confirmed_action"):
        # A confirmation must execute exactly the proposal the user saw.
        return {"resolved_query": raw}
    resolved = await resolve_query(
        raw,
        history=list(state.get("messages") or []),
        summary=state.get("summary") or "",
        llm=llm,
    )
    if resolved.rewritten:
        logger.info(
            "followup_resolved",
            extra={
                "trace_id": state.get("trace_id"),
                "source": resolved.source,
                "resolved_chars": len(resolved.query),
            },
        )
    return {"resolved_query": resolved.query}
