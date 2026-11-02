"""Command line: `python -m deskcast run | pm | generate`."""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="deskcast", description="Service desk demand forecast, SLA risk and staffing.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="generate data if missing, run the full analysis, write charts and reports")
    p_run.add_argument("--root", default=".", help="repository root (default: current directory)")
    p_run.add_argument("--regenerate", action="store_true", help="regenerate the synthetic tickets even if present")

    p_gen = sub.add_parser("generate", help="(re)generate the synthetic ticket data only")
    p_gen.add_argument("--root", default=".")
    p_gen.add_argument("--seed", type=int, default=42)

    p_pm = sub.add_parser("pm", help="run the AI project manager and write project/STATUS.md")
    p_pm.add_argument("--root", default=".")
    p_pm.add_argument("--date", default=None, help="as-of date YYYY-MM-DD (default: today)")
    p_pm.add_argument("--no-llm", action="store_true", help="skip the optional LLM executive summary")

    args = parser.parse_args(argv)
    root = Path(args.root)

    if args.command == "run":
        from .pipeline import run

        run(root, regenerate=args.regenerate)
    elif args.command == "generate":
        from .generator import GeneratorConfig
        from .pipeline import ensure_data

        path = ensure_data(root, regenerate=True, cfg=GeneratorConfig(seed=args.seed))
        print(f"[deskcast] wrote {path}")
    elif args.command == "pm":
        from .ai_pm import run_pm

        facts = run_pm(root, as_of=args.date, use_llm=not args.no_llm)
        print(f"[ai-pm] health {facts['health']['rag']}: {facts['health']['reason']}")
        print(f"[ai-pm] wrote project/STATUS.md and project/status.json (summary: {facts['executive_summary_source']})")
    return 0
