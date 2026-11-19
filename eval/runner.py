"""Offline eval harness for resume-quantifiable metrics.

python -m eval.runner --task all --llm
python -m eval.runner --task intent --llm
python -m eval.runner --task retrieval
python -m eval.runner --task generation --llm
python -m eval.runner --task e2e --llm
python -m eval.runner --task multiturn --llm
python -m eval.runner --task multiturn --llm --answers
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.metrics import (
    accuracy,
    classification_report,
    hit_at_k,
    mrr,
    percentile,
    precision_at_k,
    recall_at_k,
)

ROOT = Path(__file__).resolve().parents[1]
INTENT_PATH = ROOT / "eval/dataset/intent_100.jsonl"
RAG_PATH = ROOT / "eval/dataset/rag_eval.jsonl"
E2E_PATH = ROOT / "eval/dataset/e2e_eval.jsonl"
MULTITURN_PATH = ROOT / "eval/dataset/multiturn_eval.jsonl"
REPORT_DIR = ROOT / "eval/reports"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _doc_stem(document_id: str) -> str:
    return document_id.rsplit("_", 1)[0]


def _unique_stems(chunks) -> list[str]:
    stems: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        stem = _doc_stem(chunk.document_id)
        if stem not in seen:
            seen.add(stem)
            stems.append(stem)
    return stems


def _summarize_scores(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round(sum(values) / len(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def _keyword_hit(answer: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    return all(word in answer for word in keywords)


def eval_intent_rules() -> dict:
    from app.agents.nodes.intent import rule_based_intent

    rows = load_jsonl(INTENT_PATH)
    pairs = [(row["intent"], rule_based_intent(row["query"]).label.value) for row in rows]
    return {
        "backend": "rules",
        "prompt_version": "intent_rules_v1",
        "n": len(pairs),
        "accuracy": round(accuracy(pairs), 4),
        "per_label": {k: round(v, 4) for k, v in classification_report(pairs).items()},
    }


async def eval_intent_llm(*, version: str, overlay: bool) -> dict:
    from app.agents.nodes.intent import IntentClassifier
    from app.config.settings import get_settings
    from app.prompts.intent import INTENT_SYSTEM, INTENT_SYSTEM_V1, INTENT_SYSTEM_V2
    from app.providers.deepseek import DeepSeekProvider

    settings = get_settings()
    if not settings.llm_configured:
        raise SystemExit("DEEPSEEK_API_KEY missing")
    if version == "intent_v1":
        system = INTENT_SYSTEM_V1
    elif version == "intent_v2":
        system = INTENT_SYSTEM_V2
    else:
        system = INTENT_SYSTEM
    classifier = IntentClassifier(
        DeepSeekProvider(settings),
        system_prompt=system,
        prompt_version=version,
        apply_overlay=overlay,
    )
    rows = load_jsonl(INTENT_PATH)
    semaphore = asyncio.Semaphore(4)
    errors = 0

    async def _one(row: dict) -> tuple[str, str]:
        nonlocal errors
        async with semaphore:
            try:
                pred = (await classifier.classify(row["query"])).label.value
                return row["intent"], pred
            except Exception:
                errors += 1
                return row["intent"], "ERROR"

    pairs = list(await asyncio.gather(*[_one(row) for row in rows]))
    mismatches = [
        {"query": rows[i]["query"], "gold": gold, "pred": pred}
        for i, (gold, pred) in enumerate(pairs)
        if gold != pred
    ]
    return {
        "backend": "deepseek",
        "prompt_version": version,
        "overlay": overlay,
        "n": len(pairs),
        "errors": errors,
        "accuracy": round(accuracy(pairs), 4),
        "per_label": {k: round(v, 4) for k, v in classification_report(pairs).items()},
        "mismatches": mismatches,
    }


def eval_intent_overlay() -> dict:
    from app.agents.nodes.intent_overlay import high_precision_intent

    rows = load_jsonl(INTENT_PATH)
    pairs = []
    uncovered = 0
    for row in rows:
        pred = high_precision_intent(row["query"])
        if pred is None:
            uncovered += 1
            pairs.append((row["intent"], "NONE"))
        else:
            pairs.append((row["intent"], pred.value))
    return {
        "backend": "overlay",
        "prompt_version": "intent_overlay_v1",
        "n": len(pairs),
        "uncovered": uncovered,
        "accuracy": round(accuracy(pairs), 4),
        "per_label": {k: round(v, 4) for k, v in classification_report(pairs).items()},
    }


async def eval_retrieval() -> dict:
    from app.config.settings import get_settings
    from app.providers.embeddings import build_embedding_provider
    from app.providers.reranker import CrossEncoderRerankProvider, LexicalRerankProvider
    from app.retrieval.hybrid import HybridRetriever
    from app.retrieval.ingest import ingest_to_memory
    from app.retrieval.reranker import DocumentReranker
    from app.retrieval.vector import VectorRetriever

    settings = get_settings()
    embedder = build_embedding_provider(settings)
    dense_backend = getattr(embedder, "backend", "vector")
    vector_label = "hash-vector" if dense_backend == "hash" else dense_backend
    hybrid_label = f"bm25+{vector_label}+rrf"
    bm25, store, _ = await ingest_to_memory("knowledge", embedder)
    vector = VectorRetriever(store, embedder)
    hybrid = HybridRetriever(bm25, vector, rrf_k=60)
    rows = load_jsonl(RAG_PATH)

    async def _run(name: str, retrieve) -> dict:
        precs1, recalls3, hits1, hits3, mrrs = [], [], [], [], []
        for row in rows:
            chunks = await retrieve(row["query"])
            stems = _unique_stems(chunks)
            relevant = set(row["relevant_doc_ids"])
            precs1.append(precision_at_k(relevant, stems, 1))
            recalls3.append(recall_at_k(relevant, stems, 3))
            hits1.append(hit_at_k(relevant, stems, 1))
            hits3.append(hit_at_k(relevant, stems, 3))
            mrrs.append(mrr(relevant, stems))
        return {
            "backend": name,
            "n": len(rows),
            "precision_at_1": _summarize_scores(precs1),
            "recall_at_3": _summarize_scores(recalls3),
            "hit_at_1": _summarize_scores(hits1),
            "hit_at_3": _summarize_scores(hits3),
            "mrr": _summarize_scores(mrrs),
        }

    async def _bm25(query: str):
        return bm25.search(query, top_k=5)

    async def _vector(query: str):
        return await vector.search(query, top_k=5)

    async def _hybrid(query: str):
        return await hybrid.retrieve(query, top_k=5)

    systems = [
        await _run("bm25", _bm25),
        await _run(vector_label, _vector),
        await _run(hybrid_label, _hybrid),
    ]
    ce_provider = CrossEncoderRerankProvider(settings)
    if ce_provider.available:
        reranker = DocumentReranker(ce_provider)

        async def _hybrid_cross_encoder(query: str):
            fused = await hybrid.retrieve(query, top_k=8)
            return await reranker.rerank(query, fused, top_n=5)

        systems.append(await _run("hybrid+cross-encoder", _hybrid_cross_encoder))
        note = (
            f"Rerank is local Cross-Encoder ({settings.cross_encoder_model}). "
            f"Dense side is {vector_label}."
        )
    else:
        fallback = DocumentReranker(LexicalRerankProvider())

        async def _hybrid_lexical(query: str):
            fused = await hybrid.retrieve(query, top_k=8)
            return await fallback.rerank(query, fused, top_n=5)

        systems.append(await _run("hybrid+lexical-fallback", _hybrid_lexical))
        # Kept out of the f-string on purpose: a backslash inside an f-string
        # expression is a SyntaxError on Python 3.11 (PEP 701 landed in 3.12).
        rerank_hint = ce_provider.load_error or 'pip install -e ".[rerank]"'
        note = (
            "Cross-Encoder not loaded; hybrid+cross-encoder skipped. "
            f"{rerank_hint}. "
            "Lexical fallback is CI-only, not a Cross-Encoder."
        )
    return {
        "n": len(rows),
        "dense_backend": vector_label,
        "local_embedding_configured": settings.local_embedding_configured,
        "cross_encoder_configured": settings.cross_encoder_configured,
        "cross_encoder_available": ce_provider.available,
        "cross_encoder_model": settings.cross_encoder_model,
        "note": note,
        "systems": systems,
    }


async def _generate_answer(llm, query: str, evidence: str, *, summary: str = "") -> str:
    from app.agents.nodes.generate import AnswerPayload
    from app.prompts.answer import ANSWER_SYSTEM, ANSWER_USER
    from app.providers.json_parser import parse_model

    response = await llm.generate(
        [
            {"role": "system", "content": ANSWER_SYSTEM},
            {
                "role": "user",
                "content": ANSWER_USER.format(
                    query=query,
                    intent="faq_policy",
                    strategy="single",
                    summary=summary or "（无历史，这是本轮对话的第一句）",
                    evidence=evidence,
                    tool_results="(无)",
                ),
            },
        ],
        temperature=0.0,
        max_tokens=500,
        response_format={"type": "json_object"},
    )
    return parse_model(response.content, AnswerPayload).answer


async def eval_generation() -> dict:
    from app.config.settings import get_settings
    from app.guardrails.hallucination import FaithfulnessChecker
    from app.providers.deepseek import DeepSeekProvider
    from app.providers.embeddings import build_embedding_provider
    from app.retrieval.hybrid import HybridRetriever
    from app.retrieval.ingest import ingest_to_memory
    from app.retrieval.vector import VectorRetriever

    settings = get_settings()
    if not settings.llm_configured:
        raise SystemExit("DEEPSEEK_API_KEY missing")
    llm = DeepSeekProvider(settings)
    checker = FaithfulnessChecker(llm)
    embedder = build_embedding_provider(settings)
    bm25, store, _ = await ingest_to_memory("knowledge", embedder)
    retriever = HybridRetriever(bm25, VectorRetriever(store, embedder), rrf_k=60)
    rows = load_jsonl(RAG_PATH)
    semaphore = asyncio.Semaphore(3)

    naive_supported = 0
    rag_supported = 0
    checked_supported = 0
    naive_keyword = 0
    rag_keyword = 0
    checked_keyword = 0
    naive_hallucinated = 0
    rag_hallucinated = 0
    checked_hallucinated = 0

    async def _one(row: dict) -> None:
        nonlocal naive_supported, rag_supported, checked_supported
        nonlocal naive_keyword, rag_keyword, checked_keyword
        nonlocal naive_hallucinated, rag_hallucinated, checked_hallucinated
        async with semaphore:
            chunks = await retriever.retrieve(row["query"], top_k=5)
            evidence = (
                "\n\n".join(f"[{c.chunk_id}] {c.content}" for c in chunks[:6]) or "(无检索结果)"
            )
            naive = await _generate_answer(llm, row["query"], "(无检索结果)")
            rag = await _generate_answer(llm, row["query"], evidence)
            naive_judge = await checker.check(
                query=row["query"], answer=naive, chunks=chunks, tool_results_text=""
            )
            rag_judge = await checker.check(
                query=row["query"], answer=rag, chunks=chunks, tool_results_text=""
            )
            checked = rag
            checked_judge = rag_judge
            if not rag_judge.supported:
                from app.agents.nodes.faithfulness import self_correct_answer

                checked, checked_judge, _ = await self_correct_answer(
                    llm=llm,
                    checker=checker,
                    query=row["query"],
                    answer=rag,
                    chunks=chunks,
                    tool_results_text="",
                    first_result=rag_judge,
                )
            keywords = row.get("gold_answer_contains") or []
            if naive_judge.supported:
                naive_supported += 1
            else:
                naive_hallucinated += 1
            if rag_judge.supported:
                rag_supported += 1
            else:
                rag_hallucinated += 1
            if checked_judge.supported:
                checked_supported += 1
            else:
                checked_hallucinated += 1
            if _keyword_hit(naive, keywords):
                naive_keyword += 1
            if _keyword_hit(rag, keywords):
                rag_keyword += 1
            if _keyword_hit(checked, keywords):
                checked_keyword += 1

    await asyncio.gather(*[_one(row) for row in rows])
    n = len(rows)
    naive_h = naive_hallucinated / n
    checked_h = checked_hallucinated / n
    drop = ((naive_h - checked_h) / naive_h) if naive_h else 0.0
    return {
        "n": n,
        "judge": "deepseek-faithfulness_v1",
        "naive_no_context": {
            "faithfulness": round(naive_supported / n, 4),
            "hallucination_rate": round(naive_h, 4),
            "keyword_hit": round(naive_keyword / n, 4),
        },
        "rag": {
            "faithfulness": round(rag_supported / n, 4),
            "hallucination_rate": round(rag_hallucinated / n, 4),
            "keyword_hit": round(rag_keyword / n, 4),
        },
        "rag_plus_self_check": {
            "faithfulness": round(checked_supported / n, 4),
            "hallucination_rate": round(checked_h, 4),
            "keyword_hit": round(checked_keyword / n, 4),
        },
        "hallucination_relative_drop": round(drop, 4),
        "note": "Hallucination = LLM judge supported=false. Relative drop is naive vs RAG+self-check.",
    }


async def eval_e2e() -> dict:
    import os

    os.environ["RATE_LIMIT_PER_MINUTE"] = "1000"
    from app.config.settings import clear_settings_cache, get_settings
    from app.schemas.chat import ChatRequest
    from app.services.container import build_container

    clear_settings_cache()
    settings = get_settings()
    if not settings.llm_configured:
        raise SystemExit("DEEPSEEK_API_KEY missing")
    container = await build_container(settings, seed=True)
    rows = load_jsonl(E2E_PATH)
    latencies: list[float] = []
    intent_ms: list[float] = []
    retrieval_ms: list[float] = []
    llm_ms: list[float] = []
    tool_ms: list[float] = []
    refuse_ok = 0
    refuse_n = 0
    keyword_ok = 0
    keyword_n = 0
    toolish_ok = 0
    toolish_n = 0
    details = []
    try:
        for row in rows:
            request = ChatRequest(message=row["query"], user_id=row.get("user_id") or "u_demo")
            started = time.perf_counter()
            response = await container.chat_service.chat(request, trace_id=f"eval_{row['id']}")
            total_ms = (time.perf_counter() - started) * 1000
            latencies.append(response.latency.total_ms or total_ms)
            intent_ms.append(response.latency.intent_ms)
            retrieval_ms.append(response.latency.retrieval_ms)
            llm_ms.append(response.latency.llm_ms)
            tool_ms.append(response.latency.tool_ms)
            if row.get("expect_refused"):
                refuse_n += 1
                if response.refused:
                    refuse_ok += 1
            keywords = row.get("expect_keywords") or []
            if keywords:
                keyword_n += 1
                if _keyword_hit(response.answer, keywords):
                    keyword_ok += 1
            if any(token in row["query"] for token in ("ORD", "库存", "取消")):
                toolish_n += 1
                if "ORD" in response.answer or response.pending_action or "库存" in response.answer:
                    toolish_ok += 1
            details.append(
                {
                    "id": row["id"],
                    "refused": response.refused,
                    "strategy": response.strategy.value if response.strategy else None,
                    "intent": response.intent.value if response.intent else None,
                    "latency_ms": round(latencies[-1], 1),
                }
            )
    finally:
        await container.aclose()
    return {
        "n": len(rows),
        "refuse_accuracy": round(refuse_ok / refuse_n, 4) if refuse_n else None,
        "keyword_hit": round(keyword_ok / keyword_n, 4) if keyword_n else None,
        "tool_path_hit": round(toolish_ok / toolish_n, 4) if toolish_n else None,
        "latency_ms": {
            "p50": round(percentile(latencies, 50), 1),
            "p95": round(percentile(latencies, 95), 1),
            "mean": round(sum(latencies) / len(latencies), 1) if latencies else 0,
            "intent_p50": round(percentile(intent_ms, 50), 1),
            "retrieval_p50": round(percentile(retrieval_ms, 50), 1),
            "llm_p50": round(percentile(llm_ms, 50), 1),
            "tool_p50": round(percentile(tool_ms, 50), 1),
        },
        "cases": details,
    }


async def eval_multiturn(*, use_llm: bool, answers: bool = False, llm: Any | None = None) -> dict:
    """Follow-up rewriting and what it buys on retrieval.

    The retrieval arm is BM25-only on purpose: deterministic, needs no model weights,
    and isolates the rewrite effect from dense-model variance. Hybrid numbers live in
    ``--task retrieval``; answer quality with the full graph lives in ``--task e2e``.

    Args:
        use_llm: Resolve follow-ups with the configured provider instead of rules.
        answers: Also generate one answer per case (extra calls; requires an LLM).
        llm: Optional provider injection so tests can cover the answer arm offline.
    """
    from app.agents.nodes.context import resolve_query
    from app.config.settings import get_settings
    from app.core.constants import PROMPT_VERSION_FOLLOWUP
    from app.providers.embeddings import HashEmbeddingProvider
    from app.providers.fake import OfflineLLMProvider
    from app.retrieval.ingest import ingest_to_memory
    from app.schemas.chat import ChatMessage

    provider: Any = llm
    resolver_label = f"injected/{type(provider).__name__}" if provider is not None else "rules"
    if provider is None:
        provider = OfflineLLMProvider()
        if use_llm:
            from app.providers.deepseek import DeepSeekProvider

            settings = get_settings()
            if not settings.llm_configured:
                raise SystemExit("DEEPSEEK_API_KEY missing")
            provider = DeepSeekProvider(settings)
            resolver_label = f"deepseek/{PROMPT_VERSION_FOLLOWUP}"

    bm25, _, _ = await ingest_to_memory("knowledge", HashEmbeddingProvider())
    rows = load_jsonl(MULTITURN_PATH)

    resolved_ok = 0
    slot_ok = 0
    slot_n = 0
    no_rewrite_ok = 0
    no_rewrite_n = 0
    control_n = 0
    leaks: list[dict] = []
    answer_ok = 0
    answer_n = 0
    raw_prec1: list[float] = []
    raw_recall3: list[float] = []
    raw_mrrs: list[float] = []
    resolved_prec1: list[float] = []
    resolved_recall3: list[float] = []
    resolved_mrrs: list[float] = []
    misses: list[dict] = []
    details: list[dict] = []

    for row in rows:
        history = [ChatMessage(**turn) for turn in row.get("history") or []]
        resolved = await resolve_query(
            row["query"],
            history=history,
            summary=row.get("summary") or "",
            llm=provider,
        )
        missing = [
            term for term in row.get("expect_resolved_contains") or [] if term not in resolved.query
        ]
        if missing:
            misses.append(
                {
                    "id": row["id"],
                    "query": row["query"],
                    "resolved": resolved.query,
                    "missing": missing,
                }
            )
        else:
            resolved_ok += 1

        slot = row.get("expect_slot")
        if slot:
            slot_n += 1
            if slot in resolved.query:
                slot_ok += 1
        if row.get("expect_no_rewrite"):
            no_rewrite_n += 1
            if resolved.query.strip() == row["query"].strip():
                no_rewrite_ok += 1
            control_n += 1
            injected = [
                term for term in row.get("forbid_resolved_contains") or [] if term in resolved.query
            ]
            if injected:
                leaks.append({"id": row["id"], "resolved": resolved.query, "injected": injected})

        relevant = set(row["relevant_doc_ids"])
        raw_stems = _unique_stems(bm25.search(row["query"], top_k=5))
        resolved_stems = _unique_stems(bm25.search(resolved.query, top_k=5))
        raw_prec1.append(precision_at_k(relevant, raw_stems, 1))
        raw_recall3.append(recall_at_k(relevant, raw_stems, 3))
        raw_mrrs.append(mrr(relevant, raw_stems))
        resolved_prec1.append(precision_at_k(relevant, resolved_stems, 1))
        resolved_recall3.append(recall_at_k(relevant, resolved_stems, 3))
        resolved_mrrs.append(mrr(relevant, resolved_stems))

        answer_text = ""
        if answers and use_llm:
            chunks = bm25.search(resolved.query, top_k=5)
            evidence = (
                "\n\n".join(f"[{c.chunk_id}] {c.content}" for c in chunks[:6]) or "(无检索结果)"
            )
            answer_text = await _generate_answer(provider, resolved.query, evidence)
            answer_n += 1
            if _keyword_hit(answer_text, row.get("gold_answer_contains") or []):
                answer_ok += 1

        details.append(
            {
                "id": row["id"],
                "query": row["query"],
                "resolved": resolved.query,
                "source": resolved.source,
                "raw_stems": raw_stems[:3],
                "resolved_stems": resolved_stems[:3],
                "answer_keyword_hit": (
                    _keyword_hit(answer_text, row.get("gold_answer_contains") or [])
                    if answer_text
                    else None
                ),
            }
        )

    n = len(rows)

    def _arm(prec1: list[float], recall3: list[float], mrrs: list[float]) -> dict:
        return {
            "precision_at_1": round(sum(prec1) / n, 4) if n else 0.0,
            "recall_at_3": round(sum(recall3) / n, 4) if n else 0.0,
            "mrr": round(sum(mrrs) / n, 4) if n else 0.0,
        }

    return {
        "n": n,
        "resolver": resolver_label,
        "resolve_accuracy": round(resolved_ok / n, 4) if n else 0.0,
        "slot_carry": {
            "n": slot_n,
            "rate": round(slot_ok / slot_n, 4) if slot_n else None,
        },
        "no_rewrite": {
            "n": no_rewrite_n,
            "rate": round(no_rewrite_ok / no_rewrite_n, 4) if no_rewrite_n else None,
        },
        "history_leak": {
            "n": control_n,
            "rate": round(len(leaks) / control_n, 4) if control_n else None,
            "leaks": leaks,
            "note": (
                "Control rows are self-contained questions; injecting an earlier order id or "
                "topic into them is the harmful case. Verbatim equality is reported separately."
            ),
        },
        "retrieval_bm25": {
            "raw_followup": _arm(raw_prec1, raw_recall3, raw_mrrs),
            "resolved_query": _arm(resolved_prec1, resolved_recall3, resolved_mrrs),
        },
        "answer_keyword_hit": round(answer_ok / answer_n, 4) if answer_n else None,
        "misses": misses,
        "cases": details,
        "note": (
            "Retrieval arm is BM25-only (no model weights). "
            "Hybrid numbers are in --task retrieval; graph answer quality is in --task e2e."
        ),
    }


def _resume_bullets(report: dict) -> list[str]:
    bullets = []
    intent_rules = report.get("intent_rules")
    intent_v1 = report.get("intent_v1")
    intent_v2 = report.get("intent_v2")
    intent_v3 = report.get("intent_v3")
    overlay = report.get("intent_overlay")
    if intent_v3:
        parts = [f"intent_v3+overlay {intent_v3['accuracy']:.1%}"]
        if intent_v2:
            parts.insert(0, f"intent_v2 {intent_v2['accuracy']:.1%}")
        if intent_v1:
            parts.insert(0, f"intent_v1 {intent_v1['accuracy']:.1%}")
        extra = f"；规则 {intent_rules['accuracy']:.1%}" if intent_rules else ""
        if overlay:
            extra += f"；仅覆盖层 {overlay['accuracy']:.1%}"
        bullets.append(f"意图识别：{intent_v3['n']} 条 " + " → ".join(parts) + extra)
    retrieval = report.get("retrieval")
    if retrieval:
        bm25 = next((s for s in retrieval["systems"] if s["backend"] == "bm25"), None)
        dense = next(
            (
                s
                for s in retrieval["systems"]
                if s["backend"] not in {"bm25", "hybrid+cross-encoder", "hybrid+lexical-fallback"}
                and not str(s["backend"]).endswith("+rrf")
            ),
            None,
        )
        hybrid_rrf = next((s for s in retrieval["systems"] if s["backend"].endswith("+rrf")), None)
        cross_encoder = next(
            (s for s in retrieval["systems"] if s["backend"] == "hybrid+cross-encoder"), None
        )
        if bm25:
            bullets.append(
                f"BM25 Precision@1 {bm25['precision_at_1']['mean']:.1%}，Recall@3 {bm25['recall_at_3']['mean']:.1%}，MRR {bm25['mrr']['mean']:.2f}"
            )
        if dense:
            bullets.append(
                f"{dense['backend']} Precision@1 {dense['precision_at_1']['mean']:.1%}，"
                f"Recall@3 {dense['recall_at_3']['mean']:.1%}"
            )
        if hybrid_rrf:
            bullets.append(
                f"{hybrid_rrf['backend']} Precision@1 {hybrid_rrf['precision_at_1']['mean']:.1%}，"
                f"Recall@3 {hybrid_rrf['recall_at_3']['mean']:.1%}"
            )
        if cross_encoder:
            bullets.append(
                f"Hybrid+Cross-Encoder Precision@1 {cross_encoder['precision_at_1']['mean']:.1%}，"
                f"Recall@3 {cross_encoder['recall_at_3']['mean']:.1%}"
            )
    gen = report.get("generation")
    if gen:
        naive = gen["naive_no_context"]
        rag = gen["rag"]
        checked = gen["rag_plus_self_check"]
        bullets.append(
            f"Faithfulness：无检索 {naive['faithfulness']:.0%} → RAG {rag['faithfulness']:.0%} → RAG+自纠错 {checked['faithfulness']:.0%}"
        )
        bullets.append(
            f"幻觉率：无检索 {naive['hallucination_rate']:.0%} → RAG+自纠错 {checked['hallucination_rate']:.0%}，相对下降 {gen['hallucination_relative_drop']:.0%}"
        )
    e2e = report.get("e2e")
    if e2e:
        lat = e2e["latency_ms"]
        bullets.append(
            f"E2E {e2e['n']} 条：P50 {lat['p50']}ms / P95 {lat['p95']}ms；拒答准确率 {e2e['refuse_accuracy']}；关键词命中 {e2e['keyword_hit']}"
        )
    multiturn = report.get("multiturn")
    if multiturn:
        arms = multiturn["retrieval_bm25"]
        bullets.append(
            f"多轮追问 {multiturn['n']} 组：改写命中 {multiturn['resolve_accuracy']:.0%}，"
            f"BM25 P@1 不改写 {arms['raw_followup']['precision_at_1']:.1%} → "
            f"改写后 {arms['resolved_query']['precision_at_1']:.1%}"
        )
    return bullets


async def run_all(*, llm: bool) -> dict:
    report: dict = {
        "generated_at": datetime.now(UTC).isoformat(),
        "intent_rules": eval_intent_rules(),
        "intent_overlay": eval_intent_overlay(),
        "retrieval": await eval_retrieval(),
        "multiturn": await eval_multiturn(use_llm=llm),
    }
    if llm:
        report["intent_v1"] = await eval_intent_llm(version="intent_v1", overlay=False)
        report["intent_v2"] = await eval_intent_llm(version="intent_v2", overlay=False)
        report["intent_v3"] = await eval_intent_llm(version="intent_v3", overlay=True)
        report["generation"] = await eval_generation()
        report["e2e"] = await eval_e2e()
    report["resume_bullets"] = _resume_bullets(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=["all", "intent", "retrieval", "generation", "e2e", "multiturn"],
        default="all",
    )
    parser.add_argument("--llm", action="store_true")
    parser.add_argument(
        "--answers",
        action="store_true",
        help="multiturn: also generate an answer per case (extra LLM calls)",
    )
    args = parser.parse_args()

    async def _go() -> dict:
        if args.task == "all":
            return await run_all(llm=args.llm)
        if args.task == "intent":
            result: dict[str, Any] = {
                "intent_rules": eval_intent_rules(),
                "intent_overlay": eval_intent_overlay(),
            }
            if args.llm:
                result["intent_v1"] = await eval_intent_llm(version="intent_v1", overlay=False)
                result["intent_v2"] = await eval_intent_llm(version="intent_v2", overlay=False)
                result["intent_v3"] = await eval_intent_llm(version="intent_v3", overlay=True)
            result["resume_bullets"] = _resume_bullets(result)
            return result
        if args.task == "retrieval":
            return {"retrieval": await eval_retrieval()}
        if args.task == "multiturn":
            return {"multiturn": await eval_multiturn(use_llm=args.llm, answers=args.answers)}
        if args.task == "generation":
            return {"generation": await eval_generation()}
        return {"e2e": await eval_e2e()}

    report = asyncio.run(_go())
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = REPORT_DIR / f"{args.task}_{stamp}.json"
    latest = REPORT_DIR / "latest.json"
    text = json.dumps(report, ensure_ascii=False, indent=2)
    path.write_text(text, encoding="utf-8")
    latest.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
