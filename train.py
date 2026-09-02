from __future__ import annotations

import argparse
from pathlib import Path

from config import load_config
from run_DLPFC import run_dlpfc
from run_hbrc import run_hbrc
from run_mba import run_mba
from run_mouse import run_mouse


RUNNERS = {
    "dlpfc": run_dlpfc,
    "mouse": run_mouse,
    "mba": run_mba,
    "hbrc": run_hbrc,
}


def run_from_config(config_path: str | Path) -> float:
    config = load_config(config_path)
    dataset_name = str(config.pop("name")).lower()
    try:
        runner = RUNNERS[dataset_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset.name '{dataset_name}'. Choose from {sorted(RUNNERS)}.") from exc
    return float(runner(config))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MSFG from a reproducible YAML configuration.")
    parser.add_argument("--config", required=True, type=Path, help="Path to a YAML experiment configuration.")
    args = parser.parse_args()
    ari = run_from_config(args.config)
    print(f"Best ARI: {ari:.6f}")


if __name__ == "__main__":
    main()
