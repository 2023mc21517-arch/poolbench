"""
src/probe_training.py
AUROC computation, statistical tests, and linearity validation.

Public API
----------
compute_auroc_for_strategy(pos_pooled, neg_pooled, construction_method, n_folds, n_bootstrap)
    → {"auroc": float, "ci_low": float, "ci_high": float, "std": float}

compute_all_auroc(pooled_results, model_name, out_dir, construction_method)
    → saves per-concept×strategy AUROC matrices to out_dir

nemenyi_strategy_significance(auroc_matrix)
    → {"significant_pairs": list, "tier_boundaries": list, "nemenyi_pvalues": ndarray}

build_nemenyi_auroc_matrix(auroc_results_dict)
    → ndarray (n_strategies, n_models × n_concepts)

compute_layer_icc(layer_aurocs_per_concept: dict)
    → {"mean_icc": float, "N_eff": int, "icc_per_concept": dict}

check_linearity_assumption(pos_pooled, neg_pooled, concept_name)
    → {"passes": bool, "linear_auroc": float, "mlp_auroc": float, "gap": float}
"""

from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import Any

from poolbench.construction_methods import get_construction_method, DEFAULT_CONSTRUCTION

LINEARITY_GAP_THRESHOLD = 0.03   # max allowed gap (MLP - linear probe)
N_BOOTSTRAP             = 1000   # reproducibility default
N_FOLDS                 = 5      # stratified CV folds
RANDOM_SEED             = 42


# ── AUROC helpers ─────────────────────────────────────────────────────────────

def _auroc_from_scores(y_true: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, scores))


def _probe_scores(X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray) -> np.ndarray:
    """Fit L2 logistic regression and return decision function scores on X_test."""
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    clf = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs", random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)
    return clf.decision_function(X_test)


# ── D1 — Per-strategy AUROC ───────────────────────────────────────────────────

def compute_auroc_for_strategy(
    pos_pooled: np.ndarray,
    neg_pooled: np.ndarray,
    construction_method: str = DEFAULT_CONSTRUCTION,
    n_folds: int                = N_FOLDS,
    n_bootstrap: int            = N_BOOTSTRAP,
    sae_model=None,
) -> dict[str, float]:
    """
    5-fold stratified CV + bootstrap 95% CI for D1 (AUROC).

    Construction method (C1–C5) converts (pos, neg) pooled activations → a linear
    concept direction d. Probe = signed projection onto d.
    AUROC = AUC(y, X @ d) on the held-out fold, averaged across folds.

    Bootstrap CI: resample fold-level AUROC values with replacement (1 000 iterations).
    Final CI: [2.5th, 97.5th] percentile of bootstrap distribution.

    Returns
    -------
    {"auroc": float, "ci_low": float, "ci_high": float, "std": float, "n_pos": int,
     "n_neg": int, "construction_method": str}
    """
    from sklearn.model_selection import StratifiedKFold  # noqa: PLC0415

    X   = np.vstack([pos_pooled, neg_pooled]).astype(np.float32)
    y   = np.array([1] * len(pos_pooled) + [0] * len(neg_pooled), dtype=np.int32)
    X  /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9

    construct_fn = get_construction_method(construction_method)
    kf           = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    fold_aurocs: list[float] = []

    for tr_idx, te_idx in kf.split(X, y):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        n_tr_pos = int(y_tr.sum())
        n_tr_neg = int((~y_tr.astype(bool)).sum())
        if n_tr_pos == 0 or n_tr_neg == 0:
            fold_aurocs.append(0.5)
            continue

        pos_tr = X_tr[y_tr == 1]
        neg_tr = X_tr[y_tr == 0]

        # C5_sae_feature may return None → fall back to C1
        if construction_method == "C5_sae_feature":
            d = construct_fn(pos_tr, neg_tr, sae_model)
            if d is None:
                from poolbench.construction_methods import construct_difmean as _dm  # noqa
                d = _dm(pos_tr, neg_tr)
        else:
            d = construct_fn(pos_tr, neg_tr)

        scores = X_te @ d
        fold_aurocs.append(_auroc_from_scores(y_te, scores))

    fold_aurocs_arr = np.array(fold_aurocs)
    rng             = np.random.default_rng(RANDOM_SEED)
    boot_means      = [
        rng.choice(fold_aurocs_arr, size=len(fold_aurocs_arr), replace=True).mean()
        for _ in range(n_bootstrap)
    ]
    ci_low, ci_high = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

    return {
        "auroc":                float(fold_aurocs_arr.mean()),
        "ci_low":               ci_low,
        "ci_high":              ci_high,
        "std":                  float(fold_aurocs_arr.std()),
        "n_pos":                len(pos_pooled),
        "n_neg":                len(neg_pooled),
        "construction_method":  construction_method,
    }


# ── Compute AUROC for all strategies × concepts ───────────────────────────────

def compute_all_auroc(
    pooled_results: dict,
    model_name: str,
    out_dir: str | Path,
    construction_method: str = DEFAULT_CONSTRUCTION,
    sae_model=None,
) -> dict:
    """
    Iterate over pooled_results (output of pooling_strategies.compute_all_pooling_strategies)
    and compute AUROC for each (concept, strategy) pair.

    Saves result JSON to:
        {out_dir}/{model_name}_auroc_results.json

    Returns dict keyed as "{concept}_{strategy_id}" → AUROC result dict.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    total = len(pooled_results)
    for i, (key, data) in enumerate(pooled_results.items(), 1):
        pos = data["pos_pooled"]
        neg = data["neg_pooled"]
        if len(pos) == 0 or len(neg) == 0:
            print(f"  [probe] {key}: empty activations — skipping")
            continue
        res = compute_auroc_for_strategy(
            pos, neg,
            construction_method=construction_method,
            sae_model=sae_model,
        )
        results[key] = res
        if i % 20 == 0 or i == total:
            print(f"  [probe] {i}/{total}  {key}  AUROC={res['auroc']:.3f} "
                  f"CI=[{res['ci_low']:.3f},{res['ci_high']:.3f}]")

    out_path = out_dir / f"{model_name}_auroc_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [probe] Saved AUROC results → {out_path}")
    return results


# ── D1 — Nemenyi pair significance test ──────────────────────────────────────

def build_nemenyi_auroc_matrix(
    auroc_results_per_model: dict[str, dict],
    strategy_ids: list[str],
    concept_names: list[str],
    model_names: list[str],
) -> np.ndarray:
    """
    Build the (n_strategies, n_models × n_concepts) AUROC matrix needed for the
    Nemenyi Friedman test.

    auroc_results_per_model: {model_name: {f"{concept}_{strategy}": {"auroc": float}}}

    Returns float32 ndarray, NaN for missing cells.
    """
    n_strats  = len(strategy_ids)
    cols       = [f"{m}_{c}" for m in model_names for c in concept_names]
    n_cols     = len(cols)
    mat        = np.full((n_strats, n_cols), np.nan, dtype=np.float32)

    for col_i, col_key in enumerate(cols):
        model, concept = col_key.split("_", 1)
        model_results  = auroc_results_per_model.get(model, {})
        for row_i, strat in enumerate(strategy_ids):
            cell_key  = f"{concept}_{strat}"
            cell_data = model_results.get(cell_key)
            if cell_data and not np.isnan(cell_data.get("auroc", float("nan"))):
                mat[row_i, col_i] = cell_data["auroc"]
    return mat


def nemenyi_strategy_significance(auroc_matrix: np.ndarray,
                                  strategy_ids: list[str],
                                  alpha: float = 0.05,
                                  cd_tier_threshold: float = 0.30) -> dict:
    """
    Friedman test + Nemenyi post-hoc pairwise significance test over strategy rankings.

    auroc_matrix: (n_strategies, n_columns) where n_columns = n_models × n_concepts.
    Treats each column as a "problem" in the Friedman sense.

    cd_tier_threshold: fraction of total rank range above which we report tier boundaries.
    E.g. 0.30 means the critical difference must exceed 30% of (n_strategies - 1) / 2 for
    tiers to be reported (avoids meaningless tier splits when all strategies are similar).

    Returns
    -------
    {
      "friedman_p":          float,
      "cd":                  float,
      "avg_ranks":           dict {strategy_id: float},
      "significant_pairs":   list of (s1, s2, p_value),
      "tier_boundaries":     list of tier group lists (or [] if CD below threshold),
      "nemenyi_pvalues":     np.ndarray (n_strats, n_strats),
    }
    """
    from scipy.stats import friedmanchisquare  # noqa: PLC0415
    try:
        from scikit_posthocs import posthoc_nemenyi_friedman  # noqa: PLC0415
    except ImportError:
        posthoc_nemenyi_friedman = None

    import pandas as pd  # noqa: PLC0415

    n_strats, n_cols = auroc_matrix.shape
    # Drop columns with any NaN (incomplete observations can't be ranked)
    valid_cols = ~np.isnan(auroc_matrix).any(axis=0)
    mat_clean  = auroc_matrix[:, valid_cols]
    if mat_clean.shape[1] < 5:
        return {"error": "Too few complete observation columns for Friedman test."}

    # Rank each column (lower rank = higher AUROC in that column)
    ranks = np.zeros_like(mat_clean)
    for col in range(mat_clean.shape[1]):
        col_data   = mat_clean[:, col]
        sorted_idx = np.argsort(-col_data)   # descending AUROC → ascending rank
        r          = np.zeros(n_strats)
        r[sorted_idx] = np.arange(1, n_strats + 1)
        ranks[:, col] = r

    avg_ranks = ranks.mean(axis=1)         # (n_strats,)

    # Friedman test
    friedman_args = [mat_clean[i, :] for i in range(n_strats)]
    _, friedman_p = friedmanchisquare(*friedman_args)

    # Critical difference (alpha, two-tailed Wilcoxon signed rank)
    k  = n_strats
    N  = mat_clean.shape[1]
    # Nemenyi CD formula: CD = q_alpha * sqrt(k*(k+1) / (6*N))
    # q_alpha table (from Demšar 2006) for k strategies at alpha=0.05
    _q_table = {
        2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
        15: 3.394, 20: 3.561, 25: 3.689, 30: 3.795,
    }
    ks = sorted(_q_table.keys())
    q  = _q_table.get(k)
    if q is None:
        # Linear interpolation
        import bisect as _b  # noqa
        idx = _b.bisect_left(ks, k)
        if idx == 0:
            q = _q_table[ks[0]]
        elif idx >= len(ks):
            q = _q_table[ks[-1]]
        else:
            k_lo, k_hi = ks[idx - 1], ks[idx]
            q = _q_table[k_lo] + (_q_table[k_hi] - _q_table[k_lo]) * (k - k_lo) / (k_hi - k_lo)
    cd = float(q * np.sqrt(k * (k + 1) / (6 * N)))

    # Pairwise Nemenyi p-values
    nemenyi_pvalues = np.ones((n_strats, n_strats), dtype=np.float64)
    if posthoc_nemenyi_friedman is not None:
        try:
            df_mat = pd.DataFrame(mat_clean.T, columns=strategy_ids)
            ph_result = posthoc_nemenyi_friedman(df_mat)
            nemenyi_pvalues = ph_result.values.astype(np.float64)
        except Exception as exc:
            print(f"  [nemenyi] posthoc fallback: {exc}")

    significant_pairs = []
    for i in range(n_strats):
        for j in range(i + 1, n_strats):
            rank_diff = abs(avg_ranks[i] - avg_ranks[j])
            p_val     = float(nemenyi_pvalues[i, j])
            if rank_diff >= cd or p_val < alpha:
                significant_pairs.append((strategy_ids[i], strategy_ids[j], p_val))

    # Tier boundaries: only report if CD > cd_tier_threshold × max possible rank difference
    tier_boundaries: list = []
    max_rank_diff = (n_strats - 1) / 2.0
    if cd > cd_tier_threshold * max_rank_diff:
        sorted_strat_idx = np.argsort(avg_ranks)
        tiers: list[list[str]] = []
        current_tier: list[str] = [strategy_ids[sorted_strat_idx[0]]]
        for k_pos in range(1, n_strats):
            prev_rank = avg_ranks[sorted_strat_idx[k_pos - 1]]
            curr_rank = avg_ranks[sorted_strat_idx[k_pos]]
            if (curr_rank - prev_rank) >= cd:
                tiers.append(current_tier)
                current_tier = []
            current_tier.append(strategy_ids[sorted_strat_idx[k_pos]])
        tiers.append(current_tier)
        tier_boundaries = tiers

    return {
        "friedman_p":        float(friedman_p),
        "cd":                cd,
        "avg_ranks":         {strategy_ids[i]: float(avg_ranks[i]) for i in range(n_strats)},
        "significant_pairs": significant_pairs,
        "tier_boundaries":   tier_boundaries,
        "nemenyi_pvalues":   nemenyi_pvalues,
    }


# ── D3 — Linearity check (per-concept) ───────────────────────────────────────

def check_linearity_assumption(
    pos_pooled: np.ndarray,
    neg_pooled: np.ndarray,
    concept_name: str,
    construction_method: str = DEFAULT_CONSTRUCTION,
) -> dict[str, Any]:
    """
    Linearity validation (Appendix C requirement).
    Compare a linear logistic probe with a 2-layer MLP on the SAME activations.
    Threshold: gap < LINEARITY_GAP_THRESHOLD (0.03 AUROC) → concept is "linearly
    representable" in the chosen pooling space.

    Returns
    -------
    {
      "concept":       str,
      "passes":        bool,
      "linear_auroc":  float,
      "mlp_auroc":     float,
      "gap":           float,
    }
    """
    from sklearn.model_selection import StratifiedKFold  # noqa: PLC0415
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    X   = np.vstack([pos_pooled, neg_pooled]).astype(np.float32)
    y   = np.array([1] * len(pos_pooled) + [0] * len(neg_pooled), dtype=np.int32)
    X  /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9

    construct_fn = get_construction_method(construction_method)
    kf           = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    lin_aurocs: list[float] = []
    mlp_aurocs: list[float] = []

    for tr_idx, te_idx in kf.split(X, y):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        pos_tr = X_tr[y_tr == 1]
        neg_tr = X_tr[y_tr == 0]
        if len(pos_tr) == 0 or len(neg_tr) == 0:
            lin_aurocs.append(0.5)
            mlp_aurocs.append(0.5)
            continue

        # Linear: projection onto concept direction
        d          = construct_fn(pos_tr, neg_tr)
        lin_scores = X_te @ d
        lin_aurocs.append(_auroc_from_scores(y_te, lin_scores))

        # MLP: 2-layer, hidden=128, ReLU
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu",
            max_iter=300, random_state=RANDOM_SEED, early_stopping=True,
        )
        mlp.fit(X_tr_s, y_tr)
        mlp_scores = mlp.predict_proba(X_te_s)[:, 1]
        mlp_aurocs.append(_auroc_from_scores(y_te, mlp_scores))

    linear_auroc = float(np.mean(lin_aurocs))
    mlp_auroc    = float(np.mean(mlp_aurocs))
    gap          = mlp_auroc - linear_auroc

    return {
        "concept":      concept_name,
        "passes":       gap < LINEARITY_GAP_THRESHOLD,
        "linear_auroc": linear_auroc,
        "mlp_auroc":    mlp_auroc,
        "gap":          round(gap, 4),
    }


# ── Layer ICC (selection correction for D1) ──────────────────────────────────

def compute_layer_icc(layer_aurocs_per_concept: dict[str, list[float]],
                      n_base: int = 300) -> dict[str, Any]:
    """
    Intraclass Correlation Coefficient across layers, used to apply the
    effective-N correction for multi-layer testing.

        N_eff = N_BASE / (1 + (k - 1) × mean_ICC)

    where k = number of candidate layers evaluated per concept (typically 3–5),
    and N_BASE is the test set size per class (default 300).

    layer_aurocs_per_concept: {concept_name: [auroc_layer_1, auroc_layer_2, ...]}
    Each list must have the same length (n_layers_per_concept).

    ICC formula: Shrout & Fleiss (1979) ICC(2,1).
    We use the one-way random effects model:
        ICC = (MS_between - MS_within) / (MS_between + (k-1) * MS_within)

    Returns
    -------
    {
      "mean_icc": float,
      "N_eff":    int,
      "icc_per_concept": {concept_name: float},
    }
    """
    icc_scores: dict[str, float] = {}

    for concept, aucs in layer_aurocs_per_concept.items():
        if len(aucs) < 2:
            icc_scores[concept] = 1.0
            continue
        arr = np.array(aucs, dtype=np.float64)   # (k,)
        k   = len(arr)
        grand_mean = arr.mean()
        ss_between = k * ((arr - grand_mean) ** 2).sum()  # simplified one-rater, n=1
        # With n=1 sample per rater, we use total variance as MS_within proxy
        ss_within  = ((arr - grand_mean) ** 2).sum()
        if ss_within < 1e-10:
            icc_scores[concept] = 1.0
        else:
            icc  = (ss_between - ss_within) / (ss_between + (k - 1) * ss_within + 1e-9)
            icc_scores[concept] = float(np.clip(icc, 0.0, 1.0))

    mean_icc = float(np.mean(list(icc_scores.values()))) if icc_scores else 0.0
    # Assume same k for all (use first concept's layer count)
    all_ks = [len(v) for v in layer_aurocs_per_concept.values()]
    k_avg  = int(np.round(np.mean(all_ks))) if all_ks else 1
    N_eff  = int(n_base / (1 + (k_avg - 1) * mean_icc + 1e-9))

    return {
        "mean_icc":         mean_icc,
        "N_eff":            N_eff,
        "icc_per_concept":  icc_scores,
    }


# ── D3 — Keyword ablation check ──────────────────────────────────────────────

def keyword_ablation_check(
    pos_pooled_full: np.ndarray,
    neg_pooled_full: np.ndarray,
    pos_pooled_ablated: np.ndarray,
    neg_pooled_ablated: np.ndarray,
    concept_name: str,
    construction_method: str = DEFAULT_CONSTRUCTION,
    drop_threshold: float = 0.03,
) -> dict[str, Any]:
    """
    Keyword ablation validation (D3 disentanglement, Appendix D).
    Compare full-text AUROC vs. AUROC on keyword-masked texts.
    Seed words masked with [MASK] tokens in ablated versions.

    If ablated_auroc < full_auroc - drop_threshold, concept signal is partially
    carried by keyword surface forms → may overestimate true structural encoding.

    pos/neg_pooled_ablated: activations extracted from the ablated (masked) texts.

    Returns
    -------
    {
      "concept":           str,
      "full_auroc":        float,
      "ablated_auroc":     float,
      "drop":              float,
      "signal_in_surface": bool,
    }
    """
    full_res    = compute_auroc_for_strategy(pos_pooled_full,    neg_pooled_full,    construction_method)
    ablated_res = compute_auroc_for_strategy(pos_pooled_ablated, neg_pooled_ablated, construction_method)
    drop        = full_res["auroc"] - ablated_res["auroc"]

    return {
        "concept":           concept_name,
        "full_auroc":        full_res["auroc"],
        "ablated_auroc":     ablated_res["auroc"],
        "drop":              round(drop, 4),
        "signal_in_surface": drop > drop_threshold,
    }


def keyword_ablation_full(
    ablation_act_dir: str | Path,
    model_name: str,
    concept_names: list[str],
    strategy_id: str,
    out_dir: str | Path,
    construction_method: str = DEFAULT_CONSTRUCTION,
) -> None:
    """
    Run keyword_ablation_check for all concepts and save results JSON.
    ablation_act_dir: directory containing {concept}_pos_ablated.npy / _neg_ablated.npy.
    Full activations are loaded from the sibling directory (parent / 'activations').
    """
    ablation_act_dir = Path(ablation_act_dir)
    out_dir          = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_act_dir = ablation_act_dir.parent / "activations" / model_name
    results: dict = {}

    for concept in concept_names:
        pos_full_path = full_act_dir / f"{concept}_pos.npy"
        neg_full_path = full_act_dir / f"{concept}_neg.npy"
        pos_abl_path  = ablation_act_dir / f"{concept}_pos_ablated.npy"
        neg_abl_path  = ablation_act_dir / f"{concept}_neg_ablated.npy"

        if not all(p.exists() for p in [pos_full_path, neg_full_path, pos_abl_path, neg_abl_path]):
            print(f"  [ablation] {concept}: missing files — skipping")
            continue

        full_acts = np.load(pos_full_path, allow_pickle=True)
        neg_full  = np.load(neg_full_path, allow_pickle=True)
        abl_pos   = np.load(pos_abl_path,  allow_pickle=True)
        abl_neg   = np.load(neg_abl_path,  allow_pickle=True)

        # For ablation, pooling strategy is just mean (strategy-agnostic baseline)
        def _pool_all(acts):
            return np.stack([item["hidden"].mean(axis=0) for item in acts])

        results[concept] = keyword_ablation_check(
            _pool_all(full_acts), _pool_all(neg_full),
            _pool_all(abl_pos),   _pool_all(abl_neg),
            concept_name=concept,
            construction_method=construction_method,
        )

    out_path = out_dir / f"{model_name}_keyword_ablation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [ablation] results saved → {out_path}")
