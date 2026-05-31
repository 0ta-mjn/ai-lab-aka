import json
import logging
import os

from langfuse import Langfuse
from pydantic import BaseModel, ValidationError

from src.company_detail.schema import AddressOutput, BusinessSummaryOutput


class StatsTraceOutput(BaseModel):
    company_url: str
    address: list[AddressOutput]
    business_summary: BusinessSummaryOutput


logger = logging.getLogger(__name__)


def generate_session_stats_json(session_id: str, output_path: str) -> None:
    """
    session_idに紐づくTraceの情報をLangfuseから取得し、
    LLM呼び出し回数、fetch回数、アドレス数、citationSlot数、コスト、レイテンシーの
    統計情報をJSONとして出力する。
    """
    lf = Langfuse()
    logger.info("Fetching traces for session_id: %s", session_id)

    page = 1
    traces = []
    while True:
        res = lf.api.trace.list(session_id=session_id, page=page, limit=100)
        traces.extend(res.data)
        if len(res.data) < 100:
            break
        page += 1

    logger.info("Found %d traces.", len(traces))

    trace_stats_list = []

    for t in traces:
        generations = 0
        fetches = 0

        full_trace = lf.api.trace.get(trace_id=t.id)
        observations = getattr(full_trace, "observations", [])

        FETCH_NAMES = {
            "fetch_jina_reader_page",
            "fetch_initial_page",
            "fetch_hub_page",
            "fetch_page_detail",
        }

        total_input_tokens = 0
        total_output_tokens = 0
        total_observation_cost = 0.0
        llm_cost_excluding_fetches = 0.0
        fetch_cost = 0.0

        for o in observations:
            observation_cost = _get_observation_cost(o)
            total_observation_cost += observation_cost
            if o.type == "GENERATION":
                if o.name in FETCH_NAMES:
                    fetches += 1
                    fetch_cost += observation_cost
                else:
                    generations += 1
                    llm_cost_excluding_fetches += observation_cost

            usage = getattr(o, "usage", None)
            if usage:
                if hasattr(usage, "input"):
                    total_input_tokens += getattr(usage, "input", 0)

                if hasattr(usage, "output"):
                    total_output_tokens += getattr(usage, "output", 0)

        output = t.output
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except json.JSONDecodeError:
                pass

        if isinstance(output, dict):
            if "final_output" in output and isinstance(output["final_output"], dict):
                output = output["final_output"]

        num_addresses = 0
        num_citation_slots = 0
        unique_used_urls = 0
        parsed_output = None

        try:
            parsed_output = StatsTraceOutput.model_validate(output)
            num_addresses = len(parsed_output.address)
            num_citation_slots = len(parsed_output.business_summary.sourceUrls)

            used_urls = set()
            for addr in parsed_output.address:
                if addr.sourceUrl:
                    used_urls.add(addr.sourceUrl)

            for url in parsed_output.business_summary.sourceUrls.values():
                if url:
                    used_urls.add(url)

            unique_used_urls = len(used_urls)
        except ValidationError:
            pass

        total_trace_cost = float(t.total_cost) if t.total_cost else 0.0
        if total_observation_cost == 0.0:
            llm_cost_excluding_fetches = total_trace_cost
            fetch_cost = 0.0

        trace_stat = {
            "trace_id": t.id,
            "url": parsed_output.company_url if parsed_output else "",
            "llm_calls": generations,
            "fetches": fetches,
            "addresses": num_addresses,
            "citation_slots": num_citation_slots,
            "unique_used_urls": unique_used_urls,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "cost": llm_cost_excluding_fetches,
            "llm_cost_excluding_fetches": llm_cost_excluding_fetches,
            "fetch_cost": fetch_cost,
            "total_cost_including_fetches": total_trace_cost,
            "latency": float(t.latency) if t.latency else 0.0,
        }
        trace_stats_list.append(trace_stat)

    num_traces = len(trace_stats_list)
    averages = {
        "llm_calls": 0.0,
        "fetches": 0.0,
        "addresses": 0.0,
        "citation_slots": 0.0,
        "unique_used_urls": 0.0,
        "input_tokens": 0.0,
        "output_tokens": 0.0,
        "cost": 0.0,
        "llm_cost_excluding_fetches": 0.0,
        "fetch_cost": 0.0,
        "total_cost_including_fetches": 0.0,
        "latency": 0.0,
    }

    if num_traces > 0:
        for stat in trace_stats_list:
            averages["llm_calls"] += stat["llm_calls"]
            averages["fetches"] += stat["fetches"]
            averages["addresses"] += stat["addresses"]
            averages["citation_slots"] += stat["citation_slots"]
            averages["unique_used_urls"] += stat["unique_used_urls"]
            averages["input_tokens"] += stat["input_tokens"]
            averages["output_tokens"] += stat["output_tokens"]
            averages["cost"] += stat["cost"]
            averages["llm_cost_excluding_fetches"] += stat["llm_cost_excluding_fetches"]
            averages["fetch_cost"] += stat["fetch_cost"]
            averages["total_cost_including_fetches"] += stat[
                "total_cost_including_fetches"
            ]
            averages["latency"] += stat["latency"]

        for key in averages:
            averages[key] /= num_traces

    result = {
        "summary": {
            "total_traces": num_traces,
            "averages": averages,
        },
        "traces": trace_stats_list,
    }

    # 出力先ディレクトリがない場合は作成
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("Successfully saved stats to %s", output_path)


def _get_observation_cost(observation) -> float:
    total_cost = getattr(observation, "total_cost", None)
    if total_cost is not None:
        return float(total_cost)

    cost_details = getattr(observation, "cost_details", None)
    if isinstance(cost_details, dict):
        return float(cost_details.get("total", 0.0) or 0.0)

    return 0.0
