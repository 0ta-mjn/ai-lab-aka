import csv
import json
import os
import re
import unicodedata
import uuid
from logging import getLogger
from typing import Literal, Optional
from urllib.parse import urlparse, urlunparse

from src.company_detail.agent import run_company_detail_agent
from src.company_detail.schema import CompanyDetailOutput
from src.company_detail.tracing.stats import generate_session_stats_json
from src.company_detail.workflow import run_company_detail_workflow
from src.infra.langfuse.with_span import WithSpanContext

logger = getLogger(__name__)

CompanyDetailVariant = Literal["agent", "workflow-gpt", "workflow-cost-optimized"]


def _run_company_detail_variant(
    company_name: str,
    company_url: str,
    *,
    variant: CompanyDetailVariant,
    span_context: WithSpanContext,
) -> CompanyDetailOutput:
    if variant == "agent":
        return run_company_detail_agent(
            company_name,
            company_url,
            span_context=span_context,
        )

    if variant == "workflow-gpt":
        return run_company_detail_workflow(
            company_name,
            company_url,
            hub_selection_model="openai/gpt-5.4-mini",
            candidate_selection_model="openai/gpt-5.4-mini",
            extraction_model="openai/gpt-5.4-mini",
            merge_model="openai/gpt-5.4-mini",
            span_context=span_context,
        )

    return run_company_detail_workflow(
        company_name,
        company_url,
        span_context=span_context,
    )


def run_company_detail_workflow_csv(
    csv_path: str,
    output_path: Optional[str] = None,
    session_id: Optional[str] = None,
    variant: CompanyDetailVariant = "workflow-cost-optimized",
) -> list[CompanyDetailOutput]:
    """
    CSVファイルから企業名・URLをバッチ実行し、結果を出力する
    Args:
        csv_path (str): 入力CSV (company_name, company_url)
        output_path (str, optional): 出力ファイルパス (JSON Lines形式)
        session_id (str, optional): LangfuseセッションID
    """

    results: list[CompanyDetailOutput] = []
    if session_id is None:
        session_id = f"company-detail-{variant}-{uuid.uuid4()}"
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            company_name = row.get("company_name")
            company_url = row.get("company_url")
            if not company_name or not company_url:
                logger.warning(
                    "Skipping row with missing company_name or company_url: %s", row
                )
                continue

            logger.info("Processing company: %s, URL: %s", company_name, company_url)
            span_context: WithSpanContext = {
                "trace_init": {
                    "name": "company_detail_csv_batch",
                    "session_id": session_id,
                    "tags": [variant],
                    "metadata": {
                        "company_name": company_name,
                        "company_url": company_url,
                        "variant": variant,
                    },
                },
            }

            result = _run_company_detail_variant(
                company_name,
                company_url,
                variant=variant,
                span_context=span_context,
            )
            logger.info("Finished processing company: %s", company_name)
            results.append(result)
            print(json.dumps(result.model_dump(), ensure_ascii=False))
    if output_path:
        with open(output_path, "w", encoding="utf-8") as out:
            for result in results:
                out.write(json.dumps(result.model_dump(), ensure_ascii=False) + "\n")
        logger.info("Finished writing results to CSV: %s", output_path)

    return results


def run_company_detail_eval_csv(
    csv_path: str,
    output_dir: str,
    *,
    variants: list[CompanyDetailVariant] | None = None,
    session_id_prefix: Optional[str] = None,
) -> None:
    if variants is None:
        variants = ["agent", "workflow-gpt", "workflow-cost-optimized"]
    if session_id_prefix is None:
        session_id_prefix = f"company-detail-eval-{uuid.uuid4()}"

    os.makedirs(output_dir, exist_ok=True)

    for variant in variants:
        output_path = os.path.join(output_dir, f"company-detail-output-{variant}.jsonl")
        eval_path = os.path.join(output_dir, f"company-detail-eval-{variant}.json")
        stats_path = os.path.join(output_dir, f"company-detail-stats-{variant}.json")
        session_id = f"{session_id_prefix}-{variant}"

        logger.info("Running company detail eval: variant=%s", variant)
        results = run_company_detail_workflow_csv(
            csv_path,
            output_path=output_path,
            session_id=session_id,
            variant=variant,
        )
        eval_result = evaluate_company_detail_results(csv_path, results, variant)
        eval_result["session_id"] = session_id
        eval_result["output_path"] = output_path

        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_result, f, ensure_ascii=False, indent=2)
        logger.info("Finished writing eval result: %s", eval_path)

        generate_session_stats_json(session_id, stats_path)
        logger.info("Finished writing stats result: %s", stats_path)


def evaluate_company_detail_results(
    csv_path: str,
    results: list[CompanyDetailOutput],
    variant: CompanyDetailVariant,
) -> dict:
    with open(csv_path, newline="", encoding="utf-8") as f:
        expected_rows = list(csv.DictReader(f))

    result_by_url = {result.company_url: result for result in results}
    rows = []

    for expected in expected_rows:
        company_url = expected["company_url"]
        result = result_by_url.get(company_url)
        if result is None:
            expected_keywords = _split_keywords(
                expected.get("expected_business_keyword", "")
            )
            rows.append(
                {
                    "company_name": expected["company_name"],
                    "company_url": company_url,
                    "address_match": False,
                    "business_keyword_recall": 0.0,
                    "matched_business_keywords": [],
                    "missing_business_keywords": expected_keywords,
                    "all_source_urls_viewed": False,
                    "unviewed_source_urls": [],
                    "outside_initial_host_source_urls": [],
                    "citation_mapping_complete": False,
                    "citation_missing_source_keys": [],
                    "citation_unused_source_keys": [],
                    "citation_invalid_source_keys": [],
                }
            )
            continue

        expected_address = expected.get("expected_address", "")
        actual_addresses = [item.address for item in result.address]
        address_match = _matches_any_normalized(expected_address, actual_addresses)

        expected_keywords = _split_keywords(
            expected.get("expected_business_keyword", "")
        )
        business_text = result.business_summary.detail
        matched_keywords = [
            keyword
            for keyword in expected_keywords
            if _contains_normalized(business_text, keyword)
        ]
        missing_keywords = [
            keyword for keyword in expected_keywords if keyword not in matched_keywords
        ]
        keyword_recall = (
            len(matched_keywords) / len(expected_keywords) if expected_keywords else 0.0
        )
        source_url_eval = _evaluate_source_urls(company_url, result)
        citation_eval = _evaluate_citations(result)

        rows.append(
            {
                "company_name": expected["company_name"],
                "company_url": company_url,
                "address_match": address_match,
                "business_keyword_recall": keyword_recall,
                "matched_business_keywords": matched_keywords,
                "missing_business_keywords": missing_keywords,
                **source_url_eval,
                **citation_eval,
            }
        )

    total = len(rows)
    address_matches = sum(1 for row in rows if row["address_match"])
    all_source_urls_viewed = sum(1 for row in rows if row["all_source_urls_viewed"])
    citation_mapping_complete = sum(
        1 for row in rows if row["citation_mapping_complete"]
    )
    avg_keyword_recall = (
        sum(row["business_keyword_recall"] for row in rows) / total if total else 0.0
    )
    avg_unviewed_source_urls = (
        sum(len(row["unviewed_source_urls"]) for row in rows) / total
        if total
        else 0.0
    )
    avg_outside_initial_host_source_urls = (
        sum(len(row["outside_initial_host_source_urls"]) for row in rows) / total
        if total
        else 0.0
    )

    return {
        "variant": variant,
        "summary": {
            "total": total,
            "address_accuracy": address_matches / total if total else 0.0,
            "average_business_keyword_recall": avg_keyword_recall,
            "source_url_viewed_accuracy": (
                all_source_urls_viewed / total if total else 0.0
            ),
            "citation_mapping_accuracy": (
                citation_mapping_complete / total if total else 0.0
            ),
            "average_unviewed_source_urls": avg_unviewed_source_urls,
            "average_outside_initial_host_source_urls": (
                avg_outside_initial_host_source_urls
            ),
        },
        "rows": rows,
    }


def _split_keywords(value: str) -> list[str]:
    return [item.strip() for item in value.split("|") if item.strip()]


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower()
    normalized = re.sub(r"[‐‑‒–—―ー−]", "-", normalized)
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def _contains_normalized(text: str, keyword: str) -> bool:
    return _normalize_text(keyword) in _normalize_text(text)


def _matches_any_normalized(expected: str, actual_values: list[str]) -> bool:
    expected_normalized = _normalize_text(expected)
    if not expected_normalized:
        return False
    for actual in actual_values:
        actual_normalized = _normalize_text(actual)
        if expected_normalized in actual_normalized:
            return True
        if actual_normalized and actual_normalized in expected_normalized:
            return True
    return False


def _evaluate_source_urls(company_url: str, result: CompanyDetailOutput) -> dict:
    evidence_urls = _collect_evidence_urls(result)
    viewed_urls = {
        _normalize_url_for_compare(url) for url in result.viewed_source_urls if url
    }
    unviewed_source_urls = [
        url
        for url in evidence_urls
        if _normalize_url_for_compare(url) not in viewed_urls
    ]
    outside_initial_host_source_urls = [
        url for url in evidence_urls if not _is_same_or_sub_host(company_url, url)
    ]

    return {
        "all_source_urls_viewed": len(unviewed_source_urls) == 0,
        "unviewed_source_urls": unviewed_source_urls,
        "outside_initial_host_source_urls": outside_initial_host_source_urls,
    }


def _evaluate_citations(result: CompanyDetailOutput) -> dict:
    detail_citation_keys = set(
        re.findall(r"\[(\d+)\]", result.business_summary.detail)
    )
    source_url_keys = {
        normalized_key
        for key in result.business_summary.sourceUrls.keys()
        if (normalized_key := _normalize_citation_key(key)) is not None
    }
    invalid_source_keys = [
        key
        for key in result.business_summary.sourceUrls.keys()
        if _normalize_citation_key(key) is None
    ]
    missing_source_keys = _sort_citation_keys(detail_citation_keys - source_url_keys)
    unused_source_keys = _sort_citation_keys(source_url_keys - detail_citation_keys)

    return {
        "citation_mapping_complete": not missing_source_keys
        and not unused_source_keys
        and not invalid_source_keys,
        "citation_missing_source_keys": missing_source_keys,
        "citation_unused_source_keys": unused_source_keys,
        "citation_invalid_source_keys": invalid_source_keys,
    }


def _collect_evidence_urls(result: CompanyDetailOutput) -> list[str]:
    urls = []
    seen = set()
    for address in result.address:
        if address.sourceUrl:
            normalized = _normalize_url_for_compare(address.sourceUrl)
            if normalized not in seen:
                seen.add(normalized)
                urls.append(address.sourceUrl)

    for url in result.business_summary.sourceUrls.values():
        if url:
            normalized = _normalize_url_for_compare(url)
            if normalized not in seen:
                seen.add(normalized)
                urls.append(url)
    return urls


def _normalize_url_for_compare(url: str) -> str:
    parsed = urlparse(url.strip())
    path = parsed.path.rstrip("/") or "/"
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path,
            "",
            parsed.query,
            "",
        )
    )


def _is_same_or_sub_host(base_url: str, candidate_url: str) -> bool:
    base_host = urlparse(base_url).hostname
    candidate_host = urlparse(candidate_url).hostname
    if not base_host or not candidate_host:
        return False

    base_host = base_host.lower()
    candidate_host = candidate_host.lower()
    return candidate_host == base_host or candidate_host.endswith(f".{base_host}")


def _normalize_citation_key(key: str) -> str | None:
    stripped = key.strip()
    match = re.fullmatch(r"\[?(\d+)\]?", stripped)
    if not match:
        return None
    return match.group(1)


def _sort_citation_keys(keys: set[str]) -> list[str]:
    return sorted(keys, key=int)
