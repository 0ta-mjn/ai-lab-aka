import argparse

from dotenv import load_dotenv

from src.company_detail.run_csv_batch import (
    run_company_detail_eval_csv,
    run_company_detail_workflow_csv,
)
from src.company_detail.tracing.stats import generate_session_stats_json
from src.company_detail.workflow import run_company_detail_workflow
from src.infra.jina_ai import fetch_jina_reader_page


def register_company_detail_workflow_csv(parser: argparse.ArgumentParser) -> None:
    """
    Company Detail WorkflowをCSVバッチ実行するCLIコマンド
    """
    parser.add_argument(
        "csv_path", type=str, help="Input CSV file path (company_name, company_url)"
    )
    parser.add_argument(
        "--output_path", type=str, default=None, help="Output file path (JSON Lines)"
    )
    parser.add_argument(
        "--session_id",
        type=str,
        default=None,
        help="Langfuse Session ID for trace correlation",
    )

    parser.add_argument(
        "--variant",
        type=str,
        choices=["agent", "workflow-gpt", "workflow-cost-optimized"],
        default="workflow-cost-optimized",
        help="Variant to run",
    )

    def func(args: argparse.Namespace) -> None:
        run_company_detail_workflow_csv(
            args.csv_path,
            output_path=args.output_path,
            session_id=args.session_id,
            variant=args.variant,
        )

    parser.set_defaults(func=func)


def register_company_detail_eval_csv(parser: argparse.ArgumentParser) -> None:
    """
    Baseline / VariantA / VariantBをCSV入力で実行し、簡易評価を出力するCLIコマンド
    """
    parser.add_argument(
        "csv_path",
        type=str,
        help=(
            "Input CSV file path "
            "(company_name, company_url, expected_address, expected_business_keyword)"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for JSONL outputs and eval JSON files",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["agent", "workflow-gpt", "workflow-cost-optimized"],
        default=["agent", "workflow-gpt", "workflow-cost-optimized"],
        help="Variants to run",
    )
    parser.add_argument(
        "--session_id_prefix",
        type=str,
        default=None,
        help="Langfuse Session ID prefix for trace correlation",
    )

    def func(args: argparse.Namespace) -> None:
        run_company_detail_eval_csv(
            args.csv_path,
            args.output_dir,
            variants=args.variants,
            session_id_prefix=args.session_id_prefix,
        )

    parser.set_defaults(func=func)


def register_fetch_jina(parser: argparse.ArgumentParser) -> None:
    """
    Jina AI Readerを使用してページを取得するCLIコマンドの実装例
    """
    parser.add_argument("url", type=str, help="URL of the page to fetch")

    def func(args: argparse.Namespace) -> None:
        url = args.url
        result = fetch_jina_reader_page(url)
        if result:
            print(result.model_dump_json(indent=2))
        else:
            print(f"Failed to fetch page via Jina for URL: {url}")

    parser.set_defaults(func=func)


def register_company_detail_workflow(parser: argparse.ArgumentParser) -> None:
    """
    Company Detail Workflowを実行するCLIコマンド
    """
    parser.add_argument("--company_name", type=str, help="Company Name")
    parser.add_argument("--company_url", type=str, help="Company URL")
    parser.add_argument(
        "--session_id",
        type=str,
        default=None,
        help="Langfuse Session ID for trace correlation",
    )

    def func(args: argparse.Namespace) -> None:
        result = run_company_detail_workflow(
            args.company_name,
            args.company_url,
            span_context={
                "trace_init": {
                    "name": "company_detail_cli",
                    "session_id": args.session_id,
                    "metadata": {
                        "company_name": args.company_name,
                        "company_url": args.company_url,
                    },
                },
            },
        )
        print(result.model_dump_json(indent=2))

    parser.set_defaults(func=func)


def register_company_detail_stats(parser: argparse.ArgumentParser) -> None:
    """
    session_idからLangfuseの統計情報を取得するCLIコマンド
    """
    parser.add_argument("session_id", type=str, help="Langfuse Session ID")
    parser.add_argument(
        "--output_path", type=str, required=True, help="Output JSON file path"
    )

    def func(args: argparse.Namespace) -> None:
        generate_session_stats_json(args.session_id, args.output_path)

    parser.set_defaults(func=func)


def build_parser() -> argparse.ArgumentParser:
    """
    CLIコマンドを定義する
    """
    parser = argparse.ArgumentParser(description="CLIコマンドを実行する")
    subparsers = parser.add_subparsers(dest="command", required=True)

    register_fetch_jina(
        subparsers.add_parser("fetch-jina", help="Fetch a page using Jina AI Reader")
    )

    register_company_detail_workflow(
        subparsers.add_parser("company-detail", help="Run company detail workflow")
    )

    register_company_detail_workflow_csv(
        subparsers.add_parser(
            "company-detail-csv", help="Run company detail workflow in batch from CSV"
        )
    )

    register_company_detail_eval_csv(
        subparsers.add_parser(
            "company-detail-eval-csv",
            help="Run Baseline / VariantA / VariantB and evaluate from CSV",
        )
    )

    register_company_detail_stats(
        subparsers.add_parser(
            "company-detail-stats", help="Get Langfuse stats for a session"
        )
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
