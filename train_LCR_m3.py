"""Dedicated trainer for M3 curve-prior dual-view fusion.

This entry point reuses the dataset, loss, validation, checkpoint, and resume
logic in ``train_LCR.py`` while forcing ``--model_variant m3``.  M3 loads and
freezes the separately trained Zero-DCE, retains curve-prior-guided cross
gating, and removes the detail-denoising frequency-refinement module.
"""

from __future__ import annotations

import sys
from typing import List, Optional

import train_LCR


# Direct-run defaults. Keep M3 checkpoints isolated from all other variants.
DEFAULT_DATA_ROOT = "data/v1"
DEFAULT_OUTPUT_DIR = "checkpoints_lcr_m3/v1"


def _option_value(arguments: List[str], option: str) -> Optional[str]:
    """Return an option value for ``--key value`` and ``--key=value``."""
    prefix = option + "="
    for index, argument in enumerate(arguments):
        if argument.startswith(prefix):
            return argument[len(prefix) :]
        if argument == option:
            if index + 1 >= len(arguments):
                raise ValueError(f"{option} requires a value")
            return arguments[index + 1]
    return None


def _build_m3_arguments(arguments: List[str]) -> List[str]:
    """Add M3 defaults while preserving explicit command-line overrides."""
    result = list(arguments)

    selected_variant = _option_value(result, "--model_variant")
    if selected_variant is None:
        result.extend(("--model_variant", "m3"))
    elif selected_variant != "m3":
        raise ValueError(
            "train_LCR_m3.py is dedicated to the M3 ablation; "
            f"received --model_variant {selected_variant!r}"
        )

    if _option_value(result, "--data_root") is None:
        result.extend(("--data_root", DEFAULT_DATA_ROOT))
    if _option_value(result, "--output_dir") is None:
        result.extend(("--output_dir", DEFAULT_OUTPUT_DIR))

    return result


def main() -> None:
    sys.argv[1:] = _build_m3_arguments(sys.argv[1:])
    print(
        "M3 dedicated trainer: curve-prior dual-view cross gating; "
        "Zero-DCE is frozen and detail-denoising refinement is disabled."
    )
    train_LCR.main()


if __name__ == "__main__":
    main()
