"""CLI entrypoint for PyAI-Foundry."""

from __future__ import annotations

import argparse
import json
from . import list_losses, list_models


def main() -> None:
    parser = argparse.ArgumentParser(prog="pyai-foundry", description="All-in-One Python DS/AI/ML/DL/LLM Toolkit")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("catalog", help="List all registered architectures and losses")
    args = parser.parse_args()
    if args.cmd == "catalog":
        print(json.dumps({"models": list_models(), "losses": list_losses()}, indent=2))


if __name__ == "__main__":
    main()
