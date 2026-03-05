"""Health check — mirrors scripts/health_check.py from reference repo.

Checks: Groq API · DuckDuckGo · FastAPI server

Usage:
    python scripts/health_check.py [--api-url http://localhost:8000]
"""

from __future__ import annotations
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import httpx
from rich.console import Console
from rich.table import Table

console = Console()


def check_groq(api_key: str, model: str) -> tuple[bool, str]:
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=model, api_key=api_key, temperature=0)
        llm.invoke([{"role": "user", "content": "Reply with the single word: ok"}])
        return True, "Connected"
    except Exception as e:
        return False, str(e)[:80]


def check_ddg() -> tuple[bool, str]:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text("test", max_results=1))
        return bool(results), "Reachable" if results else "No results"
    except Exception as e:
        return False, str(e)[:80]


def check_api(url: str) -> tuple[bool, str]:
    try:
        r = httpx.get(f"{url}/health", timeout=5)
        r.raise_for_status()
        return True, r.json().get("status", "unknown")
    except Exception as e:
        return False, str(e)[:80]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000")
    args = parser.parse_args()

    from config.settings import get_settings
    s = get_settings()

    table = Table(title="Market Agent — Health Check", show_header=True)
    table.add_column("Component", style="bold cyan")
    table.add_column("Status")
    table.add_column("Detail")

    checks = [
        ("Groq API",   *check_groq(s.groq_api_key, s.groq_model)),
        ("DuckDuckGo", *check_ddg()),
        ("FastAPI",    *check_api(args.api_url)),
    ]

    all_ok = True
    for name, ok, detail in checks:
        table.add_row(name, "[green]✓ OK[/]" if ok else "[red]✗ FAIL[/]", detail)
        if not ok:
            all_ok = False

    console.print(table)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
