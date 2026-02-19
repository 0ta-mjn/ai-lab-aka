import csv
import json
import uuid
from logging import getLogger
from typing import Optional

from src.company_detail.workflow import run_company_detail_workflow

logger = getLogger(__name__)


def run_company_detail_workflow_csv(
    csv_path: str,
    output_path: Optional[str] = None,
    session_id: Optional[str] = None,
) -> None:
    """
    CSVファイルから企業名・URLをバッチ実行し、結果を出力する
    Args:
        csv_path (str): 入力CSV (company_name, company_url)
        output_path (str, optional): 出力ファイルパス (JSON Lines形式)
        session_id (str, optional): LangfuseセッションID
    """

    results = []
    if session_id is None:
        session_id = f"company-detail-{uuid.uuid4()}"
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
            result = run_company_detail_workflow(
                company_name,
                company_url,
                span_context={
                    "trace_init": {
                        "name": "company_detail_csv_batch",
                        "session_id": session_id,
                        "metadata": {
                            "company_name": company_name,
                            "company_url": company_url,
                        },
                    },
                },
            )
            logger.info("Finished processing company: %s", company_name)
            results.append(result)
            print(json.dumps(result.model_dump(), ensure_ascii=False))
    if output_path:
        with open(output_path, "w", encoding="utf-8") as out:
            for result in results:
                out.write(json.dumps(result.model_dump(), ensure_ascii=False) + "\n")
        logger.info("Finished writing results to CSV: %s", output_path)
