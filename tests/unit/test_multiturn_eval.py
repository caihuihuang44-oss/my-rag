"""Multi-turn follow-up resolution: dataset contract and offline baseline."""

from eval.runner import MULTITURN_PATH, eval_multiturn, load_jsonl


def test_dataset_contract() -> None:
    rows = load_jsonl(MULTITURN_PATH)
    assert len(rows) == 20
    assert len({row["id"] for row in rows}) == len(rows)
    for row in rows:
        assert row["query"].strip()
        assert row["relevant_doc_ids"], row["id"]
        assert row["history"], row["id"]
        for turn in row["history"]:
            assert turn["role"] in {"user", "assistant"}
            assert turn["content"].strip()
        assert row.get("expect_resolved_contains") or row.get("expect_no_rewrite"), row["id"]


def test_offline_rules_leave_complete_questions_untouched() -> None:
    rows = load_jsonl(MULTITURN_PATH)
    no_rewrite_rows = [row for row in rows if row.get("expect_no_rewrite")]
    assert no_rewrite_rows, "dataset must keep over-rewrite controls"
    # The rule fallback only touches short follow-ups, so controls stay verbatim.
    for row in no_rewrite_rows:
        assert len(row["query"]) > 12 or not any(
            cue in row["query"] for cue in ("那", "这个", "那个", "它", "还有", "呢")
        )


async def test_offline_multiturn_report_shape() -> None:
    report = await eval_multiturn(use_llm=False)
    assert report["n"] == 20
    assert report["resolver"] == "rules"
    assert 0.0 <= report["resolve_accuracy"] <= 1.0
    arms = report["retrieval_bm25"]
    for arm in ("raw_followup", "resolved_query"):
        for metric in ("precision_at_1", "recall_at_3", "mrr"):
            assert 0.0 <= arms[arm][metric] <= 1.0
    assert report["no_rewrite"]["rate"] == 1.0
    assert report["history_leak"]["n"] == 2
    assert report["history_leak"]["rate"] == 0.0
    assert report["answer_keyword_hit"] is None
    assert len(report["cases"]) == 20
    assert "BM25-only" in report["note"]


async def test_rule_baseline_leaves_the_gap_the_llm_closes() -> None:
    """Documents why the rule fallback is not the answer: it cannot carry slots.

    With rules only, a short follow-up keeps the raw query, so entity/order-id
    carry-over is lost. The LLM resolver is what closes this gap.
    """
    report = await eval_multiturn(use_llm=False)
    assert report["slot_carry"]["rate"] is not None
    assert report["slot_carry"]["rate"] < 1.0


class _StubLLM:
    """Rewrite first, then answer: covers the `--answers` arm without any network."""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    async def ping(self) -> bool:
        return True

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: dict[str, str] | None = None,
    ):
        from app.providers.base import LLMResponse
        from app.schemas.common import TokenUsage

        self.calls.append(messages)
        if "追问改写器" in messages[0]["content"]:
            content = '{"query": "定制马克杯退货的运费谁承担"}'
        else:
            content = (
                '{"answer": "质量问题运费由平台承担。", "citation_chunk_ids": [], "refused": false}'
            )
        return LLMResponse(content=content, usage=TokenUsage(total_tokens=8), model="stub")


async def test_answers_arm_is_covered_with_a_stub_llm() -> None:
    report = await eval_multiturn(use_llm=True, answers=True, llm=_StubLLM())
    assert report["resolver"].startswith("injected/")
    assert report["n"] == 20
    assert report["answer_keyword_hit"] is not None
    # The stub rewrites every turn to the same query, so injection is visible in controls.
    assert report["history_leak"]["rate"] is not None
