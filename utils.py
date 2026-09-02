import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.linalg as linalg
import scipy.sparse as sp
import sklearn
import torch
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from torch.optim import NAdam

from models import FuseMGCN_Net


DEFAULT_FIXED_CONFIG: Dict[str, float] = {
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "highly_genes": 3000,
    "min_cells": 100,
    "target_sum": 1e4,
    "scale_max_value": 10.0,
    "gft_on_feature": 1,
    "nhid1": 128,
    "nhid2": 64,
    "dropout": 0.1,
    "att_hidden": 16,
    "lambda_entropy": 0.1,
    "eval_interval": 25,
}

EXPOSED_HPARAM_KEYS: List[str] = [
    "dataset",
    "alpha",
    "beta",
    "gamma",
    "epochs",
    "gft_fraction_feature",
    "n_clusters",
    "radius",
    "seed",
    "k",
    "fusion_k",
    "fusion_t",
    "spatial_knn_k",
]


def make_config(**kwargs) -> dict:
    config = dict(DEFAULT_FIXED_CONFIG)
    config.update(kwargs)
    return config


def set_seed(seed: int = 0) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def move_to_device(device, *tensors):
    return [t.to(device) for t in tensors]


def to_int(x):
    try:
        return int(x)
    except Exception:
        return int(float(x))


def to_float(x):
    try:
        return float(x)
    except Exception:
        return float(str(x))


def make_run_dir_by_ari(dataset_dir: Path, ari: float) -> Path:
    ari_str = f"{float(ari):.4f}"
    run_dir = dataset_dir / ari_str
    if run_dir.exists():
        suffix = 1
        while True:
            alt = dataset_dir / f"{ari_str}_run{suffix}"
            if not alt.exists():
                run_dir = alt
                break
            suffix += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def safe_category(series: pd.Series) -> pd.Categorical:
    return pd.Categorical(series.astype(str))


def ensure_spatial(adata: sc.AnnData) -> None:
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"])
    elif "X_spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["X_spatial"])
        adata.obsm["spatial"] = coords
    else:
        raise ValueError("AnnData must contain adata.obsm['spatial'] or adata.obsm['X_spatial'].")

    coords = coords.astype(np.float32)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("adata.obsm['spatial'] must have shape (N, 2) or (N, >=2).")
    adata.obsm["spatial"] = coords[:, :2]


def has_visium_image(adata: sc.AnnData) -> bool:
    if "spatial" not in adata.uns:
        return False
    if not isinstance(adata.uns["spatial"], dict) or len(adata.uns["spatial"]) == 0:
        return False
    lib_id = next(iter(adata.uns["spatial"].keys()))
    entry = adata.uns["spatial"].get(lib_id, {})
    imgs = entry.get("images", {})
    return isinstance(imgs, dict) and (("hires" in imgs) or ("lowres" in imgs))


def get_library_id(adata: sc.AnnData) -> Optional[str]:
    if "spatial" not in adata.uns or not isinstance(adata.uns["spatial"], dict) or len(adata.uns["spatial"]) == 0:
        return None
    return next(iter(adata.uns["spatial"].keys()))


def plot_spatial(adata: sc.AnnData, color: str, out_png: Path, title: str, legend_title: Optional[str] = None) -> None:
    if color in adata.obs.columns and not pd.api.types.is_categorical_dtype(adata.obs[color]):
        adata.obs[color] = pd.Categorical(adata.obs[color])

    if has_visium_image(adata):
        lib_id = get_library_id(adata)
        entry = adata.uns["spatial"][lib_id]
        imgs = entry.get("images", {})
        if "hires" in imgs:
            img_key = "hires"
        elif "lowres" in imgs:
            img_key = "lowres"
        else:
            img_key = None
        plt.rcParams["figure.figsize"] = (6.8, 5.2)
        sc.pl.spatial(
            adata,
            color=color,
            img_key=img_key,
            library_id=lib_id,
            size=1.2,
            alpha_img=0.9,
            title=title,
            legend_loc="right margin",
            legend_fontsize=9,
            show=False,
        )
        plt.savefig(out_png, bbox_inches="tight", dpi=600)
        plt.close()
        return

    coords = np.asarray(adata.obsm["spatial"]).astype(np.float32)
    if pd.api.types.is_categorical_dtype(adata.obs[color]):
        y = adata.obs[color].cat.codes.values.astype(int)
        cats = list(adata.obs[color].cat.categories)
    else:
        y = np.asarray(adata.obs[color]).astype(int)
        cats = [str(i) for i in np.unique(y)]

    uniq = np.unique(y)
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    cmap = plt.cm.get_cmap("tab20", max(len(uniq), 1))
    handles, labels = [], []
    for i, lab in enumerate(uniq):
        mask = y == lab
        handle = ax.scatter(coords[mask, 0], coords[mask, 1], s=10, c=[cmap(i)], linewidths=0)
        handles.append(handle)
        labels.append(str(cats[lab]) if lab < len(cats) else str(lab))

    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("spatial1")
    ax.set_ylabel("spatial2")
    ax.legend(
        handles=handles,
        labels=labels,
        title=legend_title if legend_title else color,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        borderaxespad=0.0,
        fontsize=9,
        title_fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=600)
    plt.close(fig)


# ---------------------------
# Graph utilities
# ---------------------------
def gft_filter(features, adj, fraction=0.1):
    print(f"--- Applying GFT low-pass filter, keeping {fraction * 100:.1f}% low frequencies ---")
    deg = np.array(adj.sum(1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0
    d_inv_sqrt = sp.diags(deg_inv_sqrt)
    l_norm = sp.eye(adj.shape[0]) - d_inv_sqrt @ adj @ d_inv_sqrt
    _, eigenvectors = linalg.eigh(l_norm.toarray())
    transformed = eigenvectors.T @ features
    cutoff = int(eigenvectors.shape[1] * fraction)
    transformed[cutoff:, :] = 0
    return eigenvectors @ transformed


def _safe_row_normalize(mx: sp.spmatrix) -> sp.csr_matrix:
    mx = mx.tocsr()
    rowsum = np.array(mx.sum(1)).flatten()
    rowsum[rowsum == 0] = 1.0
    inv = 1.0 / rowsum
    return (sp.diags(inv) @ mx).tocsr()


def symmetrize(mx: sp.spmatrix, method="avg") -> sp.csr_matrix:
    mx = mx.tocsr()
    if method == "avg":
        return ((mx + mx.T) * 0.5).tocsr()
    if method == "max":
        return mx.maximum(mx.T).tocsr()
    raise ValueError("method must be 'avg' or 'max'")


def keep_topk_per_row(mx: sp.spmatrix, k: int) -> sp.csr_matrix:
    mx = mx.tocsr()
    n = mx.shape[0]
    rows, cols, data = [], [], []
    indptr, indices, values = mx.indptr, mx.indices, mx.data

    for i in range(n):
        start, end = indptr[i], indptr[i + 1]
        if start == end:
            continue
        row_idx = indices[start:end]
        row_val = values[start:end]
        if row_val.size > k:
            topk = np.argpartition(row_val, -k)[-k:]
            sel_cols = row_idx[topk]
            sel_val = row_val[topk]
        else:
            sel_cols = row_idx
            sel_val = row_val
        rows.append(np.full_like(sel_cols, i))
        cols.append(sel_cols)
        data.append(sel_val)

    if not data:
        return sp.csr_matrix(mx.shape)

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)
    out = sp.coo_matrix((data, (rows, cols)), shape=mx.shape).tocsr()
    out.eliminate_zeros()
    return out


def normalize_adj_symmetric(adj: sp.spmatrix, add_self_loops: bool = True) -> sp.coo_matrix:
    adj = adj.tocsr()
    if add_self_loops:
        adj = adj + sp.eye(adj.shape[0], format="csr")
    adj = symmetrize(adj, method="max")
    deg = np.array(adj.sum(1)).flatten()
    deg = np.clip(deg, 1e-12, None)
    deg_inv_sqrt = 1.0 / np.sqrt(deg)
    d_inv_sqrt = sp.diags(deg_inv_sqrt)
    return (d_inv_sqrt @ adj @ d_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx: sp.spmatrix) -> torch.Tensor:
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def torch_adj_to_scipy(adj_torch):
    adj_torch = adj_torch.detach().cpu()
    if adj_torch.is_sparse:
        adj_torch = adj_torch.coalesce()
        indices = adj_torch.indices().numpy()
        values = adj_torch.values().numpy()
        return sp.coo_matrix((values, (indices[0], indices[1])), shape=adj_torch.shape).tocsr()
    return sp.csr_matrix(adj_torch.numpy())


def build_knn_affinity(X: np.ndarray, k: int, metric: str = "cosine") -> sp.csr_matrix:
    n = X.shape[0]
    k = int(min(max(1, k), n - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(X)
    dist, idx = nn.kneighbors(X, return_distance=True)

    dist = dist[:, 1:]
    idx = idx[:, 1:]
    sigma = np.clip(dist[:, -1].copy(), 1e-6, None)

    rows = np.repeat(np.arange(n), k)
    cols = idx.reshape(-1)
    d = dist.reshape(-1)
    sigma_i = np.repeat(sigma, k)
    sigma_j = sigma[cols]
    w = np.exp(-(d ** 2) / (sigma_i * sigma_j + 1e-12))

    W = sp.coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    W = symmetrize(W, method="avg")
    W.setdiag(0.0)
    W.eliminate_zeros()
    return W


def snf_fuse_sparse(W_list, k: int = 20, t: int = 10) -> sp.csr_matrix:
    if len(W_list) < 2:
        raise ValueError("Need at least 2 networks to fuse.")

    P_list, S_list = [], []
    for W in W_list:
        W = W.tocsr()
        W = keep_topk_per_row(W, k)
        W = symmetrize(W, method="avg")
        W.setdiag(0.0)
        W.eliminate_zeros()
        P = _safe_row_normalize(W)
        S = _safe_row_normalize(W)
        P_list.append(P)
        S_list.append(S)

    m = len(P_list)
    for _ in range(int(t)):
        new_P = []
        P_mean = sum(P_list) * (1.0 / m)
        for v in range(m):
            P_bar = (P_mean * m - P_list[v]) * (1.0 / (m - 1))
            Pv = S_list[v].dot(P_bar).dot(S_list[v].T)
            Pv = keep_topk_per_row(Pv, k)
            Pv = _safe_row_normalize(Pv)
            new_P.append(Pv)
        P_list = new_P

    W_fuse = sum(P_list) * (1.0 / m)
    W_fuse = symmetrize(W_fuse, method="avg")
    W_fuse.setdiag(0.0)
    W_fuse.eliminate_zeros()
    return W_fuse.tocsr()


def spatial_construct_graph(adata, radius=150):
    coords = pd.DataFrame(adata.obsm["spatial"], index=adata.obs.index, columns=["imagerow", "imagecol"])
    nbrs = sklearn.neighbors.NearestNeighbors(radius=radius).fit(coords)
    _, indices = nbrs.radius_neighbors(coords, return_distance=True)

    A = np.zeros((coords.shape[0], coords.shape[0]), dtype=np.float32)
    for i, ind in enumerate(indices):
        A[i, ind] = 1.0
    np.fill_diagonal(A, 0.0)

    print(f"The graph contains {A.sum()} edges, {adata.n_obs} cells.")
    print(f"{A.sum() / adata.n_obs:.4f} neighbors per cell on average.")

    graph_nei = torch.from_numpy(A)
    graph_neg = torch.ones(coords.shape[0], coords.shape[0]) - graph_nei
    sadj = sp.coo_matrix(A)
    return sadj, graph_nei, graph_neg


def features_construct_graph(features, k=15, metric="cosine"):
    print("Constructing features graph...")
    A = kneighbors_graph(features, k, mode="connectivity", metric=metric, include_self=False)
    return A.tocoo()


# ---------------------------
# Label and data preparation helpers
# ---------------------------
def attach_ground_truth_from_metadata(
    adata: sc.AnnData,
    metadata: pd.DataFrame,
    label_candidates: Sequence[str],
    index_candidates: Optional[Sequence[str]] = None,
    allow_order_fallback: bool = False,
) -> sc.AnnData:
    label_col = next((c for c in label_candidates if c in metadata.columns), None)
    if label_col is None:
        raise ValueError(f"Metadata must contain one of {list(label_candidates)}.")

    meta = metadata.copy()
    if index_candidates:
        for col in index_candidates:
            if col in meta.columns:
                meta.index = meta[col].astype(str)
                break

    if meta.index.astype(str).equals(pd.Index(adata.obs_names.astype(str))):
        gt = meta[label_col].values
    else:
        common = adata.obs_names.intersection(meta.index.astype(str))
        if len(common) == adata.n_obs:
            gt = meta.reindex(adata.obs_names)[label_col].values
        elif allow_order_fallback and len(meta) == adata.n_obs:
            print("[Warn] Metadata index does not match obs_names exactly. Falling back to row order.")
            gt = meta[label_col].values
        else:
            raise ValueError(
                f"Failed to align metadata to AnnData. matched={len(common)}/{adata.n_obs}, label_col={label_col}"
            )

    adata.obs["ground_truth"] = gt
    adata = adata[~pd.isna(adata.obs["ground_truth"])].copy()
    adata.obs["ground_truth"] = pd.Categorical(adata.obs["ground_truth"])
    return adata


def preprocess_adata(adata: sc.AnnData, config: dict) -> sc.AnnData:
    sc.pp.filter_genes(adata, min_cells=int(config["min_cells"]))
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=int(config["highly_genes"]))
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.normalize_total(adata, target_sum=float(config["target_sum"]))
    sc.pp.scale(adata, zero_center=False, max_value=float(config["scale_max_value"]))
    return adata


def build_model_inputs(adata: sc.AnnData, config: dict):
    sadj_raw, graph_nei, graph_neg = spatial_construct_graph(adata, radius=float(config["radius"]))
    features_raw = adata.X.toarray() if sp.issparse(adata.X) else adata.X
    if bool(int(config["gft_on_feature"])):
        features_gft = gft_filter(features_raw, sadj_raw, fraction=float(config["gft_fraction_feature"]))
    else:
        features_gft = features_raw
    features_gft_np = np.asarray(features_gft, dtype=np.float32)

    fadj_raw = features_construct_graph(features_gft_np, k=int(config["k"]), metric="cosine")
    W_gene = build_knn_affinity(features_gft_np, k=int(config["fusion_k"]), metric="cosine")
    coords = adata.obsm["spatial"].astype(np.float32)
    W_spatial = build_knn_affinity(coords, k=int(config["spatial_knn_k"]), metric="euclidean")
    W_fuse = snf_fuse_sparse([W_gene, W_spatial], k=int(config["fusion_k"]), t=int(config["fusion_t"]))

    fadj = normalize_adj_symmetric(fadj_raw, add_self_loops=True)
    sadj = normalize_adj_symmetric(sadj_raw, add_self_loops=True)
    fuse_adj = normalize_adj_symmetric(W_fuse, add_self_loops=True)

    nfadj = sparse_mx_to_torch_sparse_tensor(fadj)
    nsadj = sparse_mx_to_torch_sparse_tensor(sadj)
    nfuse = sparse_mx_to_torch_sparse_tensor(fuse_adj)
    features = torch.FloatTensor(features_gft_np)

    labels = adata.obs["ground_truth"].cat.codes.values
    graph_nei_tensor = torch.FloatTensor(graph_nei.float())
    graph_neg_tensor = torch.FloatTensor(graph_neg.float())

    return {
        "features": features,
        "labels": labels,
        "fadj": nfadj,
        "sadj": nsadj,
        "fuse_adj": nfuse,
        "graph_nei": graph_nei_tensor,
        "graph_neg": graph_neg_tensor,
        "n_clusters": int(config["n_clusters"]),
        "features_gft_np": features_gft_np,
    }


# ---------------------------
# Losses
# ---------------------------
def _nan2zero(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)


def cosine_similarity(emb):
    mat = torch.matmul(emb, emb.T)
    norm = torch.norm(emb, p=2, dim=1, keepdim=True)
    mat = torch.div(mat, torch.matmul(norm, norm.T).clamp(min=1e-8))
    mat = _nan2zero(mat)
    mat.fill_diagonal_(0)
    return mat


def regularization_loss(emb, graph_nei, graph_neg):
    mat = torch.sigmoid(cosine_similarity(emb))
    neigh_loss = -torch.mul(graph_nei, torch.log(mat.clamp(min=1e-8))).mean()
    neg_loss = -torch.mul(graph_neg, torch.log((1 - mat).clamp(min=1e-8))).mean()
    return (neigh_loss + neg_loss) / 2


def consistency_loss(emb1, emb2):
    emb1 = emb1 - emb1.mean(dim=0, keepdim=True)
    emb2 = emb2 - emb2.mean(dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.T)
    cov2 = torch.matmul(emb2, emb2.T)
    return torch.mean((cov1 - cov2) ** 2)


def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.full_like(x, float("inf")), x)


class NB:
    def __init__(self, theta=None, scale_factor=1.0):
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        y_pred = y_pred * self.scale_factor
        theta = self.theta.clamp(max=1e6)
        t1 = torch.lgamma(theta + self.eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + self.eps)
        t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + self.eps))) + (
            y_true * (torch.log(theta + self.eps) - torch.log(y_pred + self.eps))
        )
        final = _nan2inf(t1 + t2)
        return final.mean() if mean else final


class ZINB(NB):
    def __init__(self, pi, ridge_lambda=0.0, **kwargs):
        super().__init__(**kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        eps = self.eps
        theta = self.theta.clamp(max=1e6)
        nb_case = super().loss(y_true, y_pred, mean=False) - torch.log(1.0 - self.pi + eps)
        y_pred = y_pred * self.scale_factor
        zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
        zero_case = -torch.log(self.pi + ((1.0 - self.pi) * zero_nb) + eps)
        result = torch.where(y_true < 1e-8, zero_case, nb_case)
        ridge = self.ridge_lambda * torch.square(self.pi)
        result += ridge
        result = _nan2inf(result)
        return result.mean() if mean else result


# ---------------------------
# Training / saving
# ---------------------------
def train_model(adata: sc.AnnData, model_inputs: dict, config: dict) -> dict:
    seed = int(config["seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
    best_emb, best_clusters, best_mean_recon = None, None, None

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
                avg_att = np.mean(att_w_eval.cpu().numpy().squeeze(), axis=0)
                print(
                    f"Epoch {epoch:03d} | Loss {total_loss.item():.4f} | ARI {ari_res:.4f} | "
                    f"Att(Fused/Spatial/Feat/Shared)=({float(avg_att[0]):.3f}, {float(avg_att[1]):.3f}, "
                    f"{float(avg_att[2]):.3f}, {float(avg_att[3]):.3f})"
                )
                if float(ari_res) > ari_max:
                    ari_max = float(ari_res)
                    best_emb = emb_np
                    best_clusters = kmeans.labels_
                    best_mean_recon = mean_eval.cpu().numpy()

    if best_emb is None or best_clusters is None or best_mean_recon is None:
        raise RuntimeError("Training finished without a valid best result.")

    return {
        "ari": ari_max,
        "best_emb": best_emb,
        "best_clusters": best_clusters,
        "best_mean_recon": best_mean_recon,
    }


def save_run_outputs(
    adata: sc.AnnData,
    dataset_result_dir: Path,
    dataset_label: str,
    config: dict,
    train_result: dict,
    features_gft_np: np.ndarray,
    save_graphs: bool = False,
    fadj=None,
    fuse_adj=None,
    gt_title: str = "Ground Truth",
) -> Path:
    run_dir = make_run_dir_by_ari(dataset_result_dir, train_result["ari"])
    print(f"Best ARI={train_result['ari']:.4f} -> {run_dir}")

    adata.obs["fuse_mgcn_cluster"] = pd.Categorical(train_result["best_clusters"].astype(int).astype(str))
    adata.obsm["fuse_mgcn_emb"] = train_result["best_emb"]

    plot_spatial(adata, "fuse_mgcn_cluster", run_dir / "cluster.png", f"FuseMGCN Clustering (ARI: {train_result['ari']:.4f})", legend_title="cluster_id")
    plot_spatial(adata, "ground_truth", run_dir / "ground_truth.png", gt_title, legend_title="ground_truth")

    recon_adata = sc.AnnData(
        X=np.asarray(train_result["best_mean_recon"], dtype=np.float32),
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )
    recon_adata.obsm["spatial"] = adata.obsm["spatial"].copy()
    recon_adata.obsm["fuse_mgcn_emb"] = adata.obsm["fuse_mgcn_emb"].copy()
    recon_adata.layers["denoised"] = np.asarray(features_gft_np, dtype=np.float32)
    if "spatial" in adata.uns:
        recon_adata.uns["spatial"] = adata.uns["spatial"]
    if save_graphs and fadj is not None and fuse_adj is not None:
        recon_adata.obsp["feature_graph"] = torch_adj_to_scipy(fadj)
        recon_adata.obsp["fused_graph"] = torch_adj_to_scipy(fuse_adj)
    recon_adata.write_h5ad(run_dir / "reconstructed_data.h5ad", compression="gzip")

    with open(run_dir / "run_info.txt", "w", encoding="utf-8") as f:
        f.write(f"dataset: {dataset_label}\n")
        f.write(f"seed: {config['seed']}\n")
        f.write(f"ARI: {train_result['ari']:.6f}\n")
        f.write(f"generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n[exposed_hparams]\n")
        for key in EXPOSED_HPARAM_KEYS:
            if key in config:
                f.write(f"{key}: {config[key]}\n")
        f.write("\n[fixed_hparams]\n")
        fixed_keys = sorted(set(config.keys()) - set(EXPOSED_HPARAM_KEYS))
        for key in fixed_keys:
            f.write(f"{key}: {config[key]}\n")

    return run_dir


def normalize_config_types(config: dict) -> dict:
    out = dict(config)
    int_keys = {"epochs", "n_clusters", "seed", "k", "fusion_k", "fusion_t", "spatial_knn_k", "gft_on_feature", "highly_genes", "min_cells", "nhid1", "nhid2", "att_hidden", "eval_interval"}
    float_keys = {"alpha", "beta", "gamma", "gft_fraction_feature", "radius", "lr", "weight_decay", "target_sum", "scale_max_value", "dropout", "lambda_entropy"}
    for key in list(out.keys()):
        if key in int_keys:
            out[key] = to_int(out[key])
        elif key in float_keys:
            out[key] = to_float(out[key])
    return out


def build_config_from_csv_row(row: pd.Series, defaults: dict, dataset_required: bool = True) -> dict:
    config = dict(defaults)
    if dataset_required:
        dataset = str(row["dataset"]).strip()
        if dataset.endswith(".0"):
            dataset = dataset[:-2]
        config["dataset"] = dataset

    for key in EXPOSED_HPARAM_KEYS:
        if key == "dataset":
            continue
        if key in row.index and pd.notna(row[key]):
            config[key] = row[key]

    return normalize_config_types(config)
