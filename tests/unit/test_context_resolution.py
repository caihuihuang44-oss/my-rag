"""Follow-up resolution keeps multi-turn conversations on topic."""

from app.agents.nodes.context import (
    prepare_context_node,
    resolve_query,
    rule_based_resolution,
)
from app.providers.fake import FakeLLMProvider, OfflineLLMProvider
from app.schemas.chat import ChatMessage

_PREV = [
    ChatMessage(role="user", content="定制马克杯能七天无理由退货吗"),
    ChatMessage(role="assistant", content="定制商品不适用七天无理由。"),
]


async def test_first_turn_costs_no_rewrite() -> None:
    resolved = await resolve_query("退款多久到账", history=[], summary="", llm=FakeLLMProvider())
    assert resolved.query == "退款多久到账"
    assert resolved.rewritten is False
    assert resolved.source == "none"


async def test_llm_rewrites_followup_with_history_and_summary() -> None:
    llm = FakeLLMProvider(canned={"运费谁出": '{"query": "定制马克杯退货的运费谁出"}'})
    resolved = await resolve_query("运费谁出", history=_PREV, summary="用户在问定制杯退货", llm=llm)
    assert resolved.query == "定制马克杯退货的运费谁出"
    assert resolved.rewritten is True
    assert resolved.source == "llm"
    sent = "\n".join(message["content"] for message in llm.calls[-1])
    assert "定制马克杯能七天无理由退货吗" in sent
    assert "用户在问定制杯退货" in sent


async def test_malformed_llm_json_degrades_to_rule_fallback() -> None:
    # The default FakeLLMProvider reply carries no "query" key: parse fails, rule runs.
    resolved = await resolve_query("那运费呢", history=_PREV, summary="", llm=FakeLLMProvider())
    assert resolved.source == "rule"
    assert resolved.query.startswith("定制马克杯能七天无理由退货吗")
    assert "那运费呢" in resolved.query


async def test_offline_provider_rewrites_short_followup_only() -> None:
    offline = OfflineLLMProvider()
    short = await resolve_query("那运费呢", history=_PREV, summary="", llm=offline)
    assert short.source == "rule"
    long_query = "请问定制商品在质量问题情况下是否仍可申请退货并退还运费"
    untouched = await resolve_query(long_query, history=_PREV, summary="", llm=offline)
    assert untouched.query == long_query
    assert untouched.rewritten is False


def test_rule_resolution_needs_a_previous_user_turn() -> None:
    resolved = rule_based_resolution("那运费呢", [])
    assert resolved.query == "那运费呢"
    assert resolved.source == "none"


async def test_prepare_context_keeps_raw_query_for_write_detection() -> None:
    llm = FakeLLMProvider(canned={"运费谁出": '{"query": "定制马克杯退货的运费谁出"}'})
    state = {
        "trace_id": "tr_test",
        "session_id": "sess_test",
        "user_id": "u_demo",
        "user_query": "运费谁出",
        "messages": list(_PREV),
        "summary": "",
    }
    update = await prepare_context_node(state, llm=llm)  # type: ignore[arg-type]
    assert update["resolved_query"] == "定制马克杯退货的运费谁出"
    assert state["user_query"] == "运费谁出"


async def test_confirmation_turn_is_never_rewritten() -> None:
    llm = FakeLLMProvider(canned={"确认": '{"query": "不该出现"}'})
    state = {
        "user_query": "确认",
        "messages": list(_PREV),
        "confirm_action_id": "act_1",
    }
    update = await prepare_context_node(state, llm=llm)  # type: ignore[arg-type]
    assert update["resolved_query"] == "确认"
