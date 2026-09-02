from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from sklearn.cluster import KMeans
from torch.optim import NAdam

from config import load_config
from models import FuseMGCN_Net
from run_DLPFC import DEFAULT_CONFIG as DLPFC_DEFAULTS, prepare_dlpfc_data
from run_hbrc import DEFAULT_CONFIG as HBRC_DEFAULTS, prepare_hbrc_data
from run_mba import DEFAULT_CONFIG as MBA_DEFAULTS, prepare_mba_data
from run_mouse import DEFAULT_CONFIG as MOUSE_DEFAULTS, prepare_mouse_data
from utils import (
    ZINB,
    build_model_inputs,
    consistency_loss,
    move_to_device,
    normalize_config_types,
    regularization_loss,
    save_run_outputs,
    set_seed,
)


# ---------------------------------------------------------------------------
# Per-dataset metadata (mirrors the save_run_outputs calls in each run_*.py).
# This script reuses the existing prepare/build/save helpers and only adds
# (1) multi-round training and (2) persistent model-weight checkpoints.
# ---------------------------------------------------------------------------
DATASETS = {
    "dlpfc": {
        "defaults": DLPFC_DEFAULTS,
        "prepare": prepare_dlpfc_data,
        "result_dir": lambda cfg: Path("result/DLPFC") / str(cfg["dataset"]),
        "dataset_label": lambda cfg: str(cfg["dataset"]),
        "save_graphs": True,
        "gt_title": "Ground Truth Annotation",
    },
    "mouse": {
        "defaults": MOUSE_DEFAULTS,
        "prepare": prepare_mouse_data,
        "result_dir": lambda cfg: Path("result_mouse") / "mouse_E1S1",
        "dataset_label": lambda cfg: "mouse_E1S1",
        "save_graphs": False,
        "gt_title": "Ground Truth (annotation)",
    },
    "mba": {
        "defaults": MBA_DEFAULTS,
        "prepare": prepare_mba_data,
        "result_dir": lambda cfg: Path("result_mba") / str(cfg["dataset"]),
        "dataset_label": lambda cfg: str(cfg["dataset"]),
        "save_graphs": False,
        "gt_title": "Ground Truth",
    },
    "hbrc": {
        "defaults": HBRC_DEFAULTS,
        "prepare": prepare_hbrc_data,
        "result_dir": lambda cfg: Path("result_hbrc") / "HBRC",
        "dataset_label": lambda cfg: "HBRC",
        "save_graphs": False,
        "gt_title": "Ground Truth",
    },
}


def resolve_device(preference: str) -> torch.device:
    if preference == "cuda":
        return torch.device("cuda")
    if preference == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_round(adata, model_inputs: dict, config: dict, device: torch.device) -> dict:
    """Run a single training pass and return the best result plus the model weights.

    This mirrors `utils.train_model` exactly (losses, optimizer, eval schedule,
    best-by-ARI selection) but additionally snapshots `model.state_dict()` at the
    best-ARI point so the trained weights can be persisted.
    """
    seed = int(config["seed"])
    set_seed(seed)

    features, fadj, sadj, fuse_adj, graph_nei, graph_neg = move_to_device(
        device,
        model_inputs["features"],
        model_inputs["fadj"],
        model_inputs["sadj"],
        model_inputs["fuse_adj"],
        model_inputs["graph_nei"],
        model_inputs["graph_neg"],
    )

    model = FuseMGCN_Net(
        nfeat_gene=features.shape[1],
        nhid1=int(config["nhid1"]),
        nhid2=int(config["nhid2"]),
        dropout=float(config["dropout"]),
        att_hidden=int(config["att_hidden"]),
    ).to(device)

    optimizer = NAdam(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    labels = model_inputs["labels"]
    n_clusters = int(model_inputs["n_clusters"])

    ari_max = -1.0
    best = None

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        optimizer.zero_grad()
        com_u, com_s, emb, pi, disp, mean, att_weights = model(features, fadj, sadj, fuse_adj)

        zinb_loss = ZINB(pi, theta=disp, ridge_lambda=0).loss(features, mean, mean=True)
        reg_loss = regularization_loss(emb, graph_nei, graph_neg)
        con_loss = consistency_loss(com_u, com_s)
        aw = att_weights.squeeze(-1)
        attention_entropy = -torch.sum(aw * torch.log(aw + 1e-8), dim=1).mean()

        total_loss = (
            float(config["alpha"]) * zinb_loss
            + float(config["beta"]) * con_loss
            + float(config["gamma"]) * reg_loss
            - float(config["lambda_entropy"]) * attention_entropy
        )
        total_loss.backward()
        optimizer.step()

        if (epoch % int(config["eval_interval"]) == 0) or epoch == 1 or epoch == int(config["epochs"]):
            with torch.no_grad():
                model.eval()
                _, _, emb_eval, _, _, mean_eval, att_w_eval = model(features, fadj, sadj, fuse_adj)
                emb_np = pd.DataFrame(emb_eval.cpu().numpy()).fillna(0).values
                kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=seed).fit(emb_np)
                ari_res = metrics.adjusted_rand_score(labels, kmeans.labels_)

                if float(ari_res) > ari_max:
                    ari_max = float(ari_res)
                    best = {
                        "ari": ari_max,
                        "best_emb": emb_np,
                        "best_clusters": kmeans.labels_,
                        "best_mean_recon": mean_eval.cpu().numpy(),
                        "best_epoch": epoch,
                        "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    }

    if best is None or best.get("model_state_dict") is None:
        raise RuntimeError("Training finished without a valid best result.")

    return best


def save_checkpoint(run_dir: Path, config: dict, result: dict, dataset_name: str, feature_dim: int) -> None:
    """Persist the best-round model weights and a self-describing metadata file."""
    state_path = run_dir / "model_state.pth"
    torch.save(result["model_state_dict"], state_path)

    meta = {
        "dataset": dataset_name,
        "seed": config["seed"],
        "ari": result["ari"],
        "best_epoch": result["best_epoch"],
        "round": result.get("round"),
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_arch": {
            "class": "FuseMGCN_Net",
            "nfeat_gene": feature_dim,
            "nhid1": config["nhid1"],
            "nhid2": config["nhid2"],
            "dropout": config["dropout"],
            "att_hidden": config["att_hidden"],
        },
        "weight_file": state_path.name,
        "config": {k: (float(v) if isinstance(v, np.floating) else v) for k, v in config.items()},
    }
    with open(run_dir / "checkpoint_meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False, default=str)

    print(f"Saved model weights: {state_path}")


def run_config(config_path: Path, rounds: int, device: torch.device) -> float:
    config = load_config(config_path)
    dataset_name = str(config.pop("name")).lower()
    meta = DATASETS[dataset_name]

    cfg = dict(meta["defaults"])
    cfg.update(config)
    cfg = normalize_config_types(cfg)

    print(f"\n{'=' * 70}\nConfig: {config_path}")
    print(f"Dataset: {dataset_name} | Rounds: {rounds} | Device: {device}\n{'=' * 70}")

    adata = meta["prepare"](cfg)
    model_inputs = build_model_inputs(adata, cfg)
    feature_dim = int(model_inputs["features"].shape[1])

    best_result = None
    best_ari = -1.0
    round_log = []

    for r in range(1, rounds + 1):
        print(f"\n--- Round {r}/{rounds} (seed={cfg['seed']}) ---")
        result = train_one_round(adata, model_inputs, cfg, device)
        result["round"] = r
        round_log.append({"round": r, "seed": cfg["seed"], "ari": float(result["ari"])})
        print(f">>> Round {r} best ARI = {result['ari']:.4f}")
        if result["ari"] > best_ari:
            best_ari = result["ari"]
            best_result = result

    assert best_result is not None

    run_dir = save_run_outputs(
        adata=adata,
        dataset_result_dir=meta["result_dir"](cfg),
        dataset_label=meta["dataset_label"](cfg),
        config=cfg,
        train_result=best_result,
        features_gft_np=model_inputs["features_gft_np"],
        save_graphs=meta["save_graphs"],
        fadj=model_inputs["fadj"],
        fuse_adj=model_inputs["fuse_adj"],
        gt_title=meta["gt_title"],
    )

    save_checkpoint(run_dir, cfg, best_result, dataset_name, feature_dim)

    with open(run_dir / "rounds_log.json", "w", encoding="utf-8") as handle:
        json.dump(round_log, handle, indent=2)

    print(f"\nFinal best ARI over {rounds} round(s): {best_ari:.4f} (round {best_result['round']})")
    print(f"Run directory: {run_dir}\n")
    return best_ari


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MSFG over multiple rounds and save the best model weights "
                    "(new, additive utility; does not modify the existing training path)."
    )
    parser.add_argument("--config", type=Path, default=None, help="Single YAML config file.")
    parser.add_argument("--all", action="store_true", help="Run every YAML in configs/paper/.")
    parser.add_argument("--rounds", type=int, default=5, help="Number of training rounds per config.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if not args.config and not args.all:
        parser.error("Provide --config PATH or --all.")

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    if args.all:
        config_files = sorted(Path("configs/paper").glob("*.yaml"))
    else:
        config_files = [args.config]

    summary = {}
    for path in config_files:
        summary[str(path)] = run_config(path, args.rounds, device)

    print("\n" + "=" * 70)
    print("Summary of best ARI per config:")
    for path, ari in summary.items():
        print(f"  {path}: {ari:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()