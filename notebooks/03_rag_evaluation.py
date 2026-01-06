"""Run qualitative RAG evaluation.

This is a thin wrapper around `src.evaluate` so you can run evaluation from the
`notebooks/` area as originally planned.

Usage:
    python notebooks/03_rag_evaluation.py
"""

import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    from src.evaluate import run_evaluation

    run_evaluation()


if __name__ == "__main__":
    main()
