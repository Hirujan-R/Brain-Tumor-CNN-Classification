"""Backwards-compatible alias for the unified CV pipeline.

The canonical implementation now lives in ``src.pipelines.train``. This module
is kept so that existing commands / notebooks that call
``python -m src.pipelines.train_cv`` continue to work.
"""

from src.pipelines.train import main

if __name__ == "__main__":
    main()
