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


HBRC_DIR = Path("./data/Human_Breast_Cancer")
RESULT_ROOT = Path("result_hbrc")
DATASET_NAME = "HBRC"


DEFAULT_CONFIG = make_config(
    dataset=DATASET_NAME,
    alpha=1.0,
    beta=1.0,
    gamma=1.0,
    epochs=800,
    gft_fraction_feature=0.1,
    n_clusters=20,
    radius=450.0,
    seed=0,
    k=15,
    fusion_k=20,
    fusion_t=10,
    spatial_knn_k=20,
)


def prepare_hbrc_data(config: dict):
    if not HBRC_DIR.exists():
        raise FileNotFoundError(f"HBRC dir not found: {HBRC_DIR.resolve()}")

    print(f"--- Loading HBRC dataset: {HBRC_DIR} ---")
    adata = sc.read_visium(str(HBRC_DIR), count_file="filtered_feature_bc_matrix.h5", load_images=True)
    adata.var_names_make_unique()

    metadata = pd.read_table(HBRC_DIR / "metadata.tsv", sep="\t")
    adata = attach_ground_truth_from_metadata(
        adata,
        metadata,
        label_candidates=["ground_truth"],
        index_candidates=["ID", "barcode", "Unnamed: 0"],
        allow_order_fallback=False,
    )
    print(f"Using K(n_clusters) from config: {config['n_clusters']} | GT classes={adata.obs['ground_truth'].nunique()}")
    adata = preprocess_adata(adata, config)
    return adata


def run_hbrc(config: dict | None = None) -> float:
    cfg = dict(DEFAULT_CONFIG if config is None else {**DEFAULT_CONFIG, **config})
    cfg = normalize_config_types(cfg)

    adata = prepare_hbrc_data(cfg)
    model_inputs = build_model_inputs(adata, cfg)
    result = train_model(adata, model_inputs, cfg)
    save_run_outputs(
        adata=adata,
        dataset_result_dir=RESULT_ROOT / DATASET_NAME,
        dataset_label=DATASET_NAME,
        config=cfg,
        train_result=result,
        features_gft_np=model_inputs["features_gft_np"],
        gt_title="Ground Truth",
    )
    return result["ari"]


if __name__ == "__main__":
    run_hbrc()
