"""Dedicated trainer entry point for the M1 low-light single-view baseline.

This script reuses the dataset, loss, validation, checkpoint, and resume logic
from :mod:`train_LCR`, while forcing ``--model_variant m1``.  It is intended
to be run directly from PyCharm without having to add the model-variant
argument manually.

Expected paired-data layout::

    data_root/
        train/
            low/
            high/
        val/
            low/
            high/

The defaults below can be edited for an IDE run, or overridden with normal
command-line arguments such as ``--data_root`` and ``--output_dir``.
"""

from __future__ import annotations

import sys
from typing import List, Optional

import train_LCR


# Direct-run defaults. Keep M1 checkpoints separate from the full model.
DEFAULT_DATA_ROOT = "data/v1"
DEFAULT_OUTPUT_DIR = "checkpoints_lcr_m1/v1"


def _option_value(arguments: List[str], option: str) -> Optional[str]:
    """Return an option value for both ``--key value`` and ``--key=value``."""
    prefix = option + "="
    for index, argument in enumerate(arguments):
        if argument.startswith(prefix):
            return argument[len(prefix) :]
        if argument == option:
            if index + 1 >= len(arguments):
                raise ValueError(f"{option} requires a value")
            return arguments[index + 1]
    return None


def _build_m1_arguments(arguments: List[str]) -> List[str]:
    """Add safe M1 defaults while preserving explicit user overrides."""
    result = list(arguments)

    selected_variant = _option_value(result, "--model_variant")
    if selected_variant is None:
        result.extend(("--model_variant", "m1"))
    elif selected_variant != "m1":
        raise ValueError(
            "train_LCR_m1.py is dedicated to the M1 baseline; "
            f"received --model_variant {selected_variant!r}"
        )

    if _option_value(result, "--data_root") is None:
        result.extend(("--data_root", DEFAULT_DATA_ROOT))
    if _option_value(result, "--output_dir") is None:
        result.extend(("--output_dir", DEFAULT_OUTPUT_DIR))

    return result


def main() -> None:
    sys.argv[1:] = _build_m1_arguments(sys.argv[1:])
    print(
        "M1 dedicated trainer: low-light single-view model; "
        "Zero-DCE is disabled."
    )
    train_LCR.main()


if __name__ == "__main__":
    main()
