import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
import scanpy as sc

from utils import (
    attach_ground_truth_from_metadata,
    build_model_inputs,
    make_config,
    normalize_config_types,
    preprocess_adata,
    save_run_outputs,
    train_model,
)


MBA_ROOT = Path("./data/MBA")
RESULT_ROOT = Path("result_mba")


DEFAULT_CONFIG = make_config(
    dataset="Mouse_Brain_Anterior",
    alpha=1.0,
    beta=1.0,
    gamma=1.0,
    epochs=800,
    gft_fraction_feature=0.1,
    n_clusters=-1,
    radius=150.0,
    seed=0,
    k=15,
    fusion_k=20,
    fusion_t=10,
    spatial_knn_k=20,
)


def prepare_mba_data(config: dict):
    dataset = str(config["dataset"])
    visium_dir = MBA_ROOT / dataset
    if not visium_dir.exists():
        raise FileNotFoundError(f"MBA dataset dir not found: {visium_dir.resolve()}")

    print(f"--- Loading MBA dataset: {dataset} ---")
    adata = sc.read_visium(str(visium_dir), count_file="filtered_feature_bc_matrix.h5", load_images=True)
    adata.var_names_make_unique()

    metadata = pd.read_table(visium_dir / "metadata.tsv", sep="\t")
    adata = attach_ground_truth_from_metadata(
        adata,
        metadata,
        label_candidates=["ground_truth"],
        index_candidates=["Unnamed: 0", "barcode", "ID"],
        allow_order_fallback=False,
    )

    gt_k = int(adata.obs["ground_truth"].nunique())
    if int(config["n_clusters"]) <= 0:
        config["n_clusters"] = gt_k
    print(f"n_clusters used: {config['n_clusters']} | GT classes={gt_k}")
    adata = preprocess_adata(adata, config)
    return adata


def run_mba(config: dict | None = None) -> float:
    cfg = dict(DEFAULT_CONFIG if config is None else {**DEFAULT_CONFIG, **config})
    cfg = normalize_config_types(cfg)

    adata = prepare_mba_data(cfg)
    model_inputs = build_model_inputs(adata, cfg)
    result = train_model(adata, model_inputs, cfg)
    save_run_outputs(
        adata=adata,
        dataset_result_dir=RESULT_ROOT / str(cfg["dataset"]),
        dataset_label=str(cfg["dataset"]),
        config=cfg,
        train_result=result,
        features_gft_np=model_inputs["features_gft_np"],
        gt_title="Ground Truth",
    )
    return result["ari"]


if __name__ == "__main__":
    run_mba()
