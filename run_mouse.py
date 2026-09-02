import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import scanpy as sc

from utils import (
    build_model_inputs,
    ensure_spatial,
    make_config,
    normalize_config_types,
    preprocess_adata,
    save_run_outputs,
    train_model,
)


MOUSE_H5AD_PATH = Path("./data/mouse/E1S1.h5ad")
RESULT_ROOT = Path("result_mouse")
DATASET_NAME = "mouse_E1S1"


DEFAULT_CONFIG = make_config(
    dataset=DATASET_NAME,
    alpha=1.0,
    beta=1.0,
    gamma=1.0,
    epochs=800,
    gft_fraction_feature=0.1,
    n_clusters=-1,
    radius=1.1,
    seed=0,
    k=20,
    fusion_k=20,
    fusion_t=10,
    spatial_knn_k=20,
)


def prepare_mouse_data(config: dict):
    if not MOUSE_H5AD_PATH.exists():
        raise FileNotFoundError(f"Mouse h5ad not found: {MOUSE_H5AD_PATH.resolve()}")

    print(f"--- Loading mouse h5ad: {MOUSE_H5AD_PATH} ---")
    adata = sc.read_h5ad(str(MOUSE_H5AD_PATH))
    adata.var_names_make_unique()
    ensure_spatial(adata)

    if "annotation" not in adata.obs.columns:
        raise ValueError("Mouse h5ad must contain obs['annotation'] for ground truth labels.")
    adata.obs["ground_truth"] = adata.obs["annotation"].astype("category")
    gt_k = int(adata.obs["ground_truth"].nunique())
    config["n_clusters"] = gt_k
    print(f"Fixed n_clusters = {gt_k} (from obs['annotation'] classes)")
    adata = preprocess_adata(adata, config)
    return adata


def run_mouse(config: dict | None = None) -> float:
    cfg = dict(DEFAULT_CONFIG if config is None else {**DEFAULT_CONFIG, **config})
    cfg = normalize_config_types(cfg)

    adata = prepare_mouse_data(cfg)
    model_inputs = build_model_inputs(adata, cfg)
    result = train_model(adata, model_inputs, cfg)
    save_run_outputs(
        adata=adata,
        dataset_result_dir=RESULT_ROOT / DATASET_NAME,
        dataset_label=DATASET_NAME,
        config=cfg,
        train_result=result,
        features_gft_np=model_inputs["features_gft_np"],
        gt_title="Ground Truth (annotation)",
    )
    return result["ari"]


if __name__ == "__main__":
    run_mouse()
