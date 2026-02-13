import argparse

from dotenv import load_dotenv

from src.infra.jina_ai_reader import fetch_jina_reader_page


def register_fetch_jina(parser: argparse.ArgumentParser) -> None:
    """
    Jina AI Readerを使用してページを取得するCLIコマンドの実装例
    """
    parser.add_argument("url", type=str, help="URL of the page to fetch")

    def func(args: argparse.Namespace) -> None:
        url = args.url
        result = fetch_jina_reader_page(url)
        if result:
            print("Content:", result["content"])
            print("Title:", result.get("title"))
            print("Description:", result.get("description"))
            print("URL:", result["url"])
            print("Links:", result["links"])
        else:
            print(f"Failed to fetch page via Jina for URL: {url}")

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

    return parser


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
