"""
Visualization script: Cross-Dataset Robustness (Macro F1 across layers).

For each taxonomy it produces:

  (A) `cross_test_{tax}.png`        -> the original figure: cross-domain curves
                                       with their 95% CI bands + in-domain
                                       baselines as POINT estimates (dashed).
  (B) `cross_test_{tax}_{key}.png`  -> ONE figure per dispersion metric in
                                       DISPERSION_METRICS (ci95, sd196, sd1, sd2).
                                       ALL four series get the same band so you
                                       can see how the shading changes with the
                                       dispersion metric used around the mean.

Bands use Option 1 (test-set bootstrap, no retraining):
  * Cross-domain (Gen->Human, Human->Gen): read from the CSV produced by
    `interdataset_cross_test.py`, which now stores mean, std and percentiles.
  * In-domain (Gen->Gen, Human->Human): computed here by reloading the saved
    probe + activations, reconstructing the exact training split, and
    stratified-bootstrapping the held-out predictions. Cached to CSV.

Both store mean/std/percentiles, so any band is drawn without re-bootstrapping.
If the cross-domain CSV is the OLD format (only mean/lower/upper), SD bands for
the cross-domain series are skipped (a warning is printed); re-run
`interdataset_cross_test.py` to enable them. The y-axis is anchored at 0.
"""

import os
from collections import Counter

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils import resample

# --- CONFIGURACIÓN ---
LLM_USED = "Qwen2.5-14B-Instruct" # "Llama-2-7b-chat-hf"
BASE_DIR = "/home/jcuello/emotion_drift"
MODEL_DIM = 5120 # 4096 #  for Qwen2.5-14B
FIGURES_DIR = os.path.join(BASE_DIR, "figures", f"cross_testing_performance_{LLM_USED}")

# Cross-testing bootstrap results (from interdataset_cross_test.py)
INPUT_CSV = os.path.join(FIGURES_DIR, "cross_test_bootstrap_results.csv")

# Cached in-domain bootstrap stats (computed here)
BOOTSTRAP_CSV = os.path.join(FIGURES_DIR, f"in_domain_baseline_bootstrap_ci_{LLM_USED}.csv")

TAXONOMIES = ['ekman_basic_emotions', 'plutchik_wheel']

MODELS_DIR = os.path.join(BASE_DIR, "models")

# Baseline point-estimate CSVs (in-domain macro F1, single split).
PATH_BASELINE_GEN = "results/probes_generated_prompts_Qwen2.5-14B-Instruct/full_probing_metrics_Qwen2.5-14B-Instruct_final_F1.csv"
PATH_BASELINE_HUMAN = "results/probes_human_centric_Qwen2.5-14B-Instruct/full_probing_metrics_Qwen2.5-14B-Instruct_final_F1.csv"

# Split reconstruction (must match train_linear_probes_on_annotations.py).
MIN_SAMPLES_REQUIRED = 5
SPLIT_RANDOM_STATE = 42
ACTIVATION_COL = "last_token_activation"

# Bootstrap settings (match interdataset_cross_test.py).
BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 42
USE_CACHED_BOOTSTRAP = True  # reuse the cached stats CSV if it exists

# Percentiles stored from the bootstrap distribution.
PERCENTILES = [2.5, 5, 16, 25, 75, 84, 95, 97.5]


def _pct_col(p):
    """Column name for a stored percentile, e.g. 2.5 -> 'f1_p2_5'."""
    return f"f1_p{str(p).replace('.', '_')}"


# In-domain baselines: how to reconstruct each one.
IN_DOMAIN = {
    "Generated": {
        "dataset": "generated_prompts",
        "data_path": os.path.join(
            BASE_DIR, "data", "03_activations",
            "generated_prompts_Qwen2.5-14B-Instruct_20251220_225401_FINAL.pkl",
        ),
        "baseline_csv": PATH_BASELINE_GEN,
        "color": "#1f77b4",
        "label": "Train: Generated $\\to$ Test: Generated",
    },
    "Human": {
        "dataset": "human_centric",
        "data_path": os.path.join(
            BASE_DIR, "data", "03_activations",
            "MERGED_andyzou_emotion_query_Qwen2.5-14B-Instruct_20260618_152745_FINAL.pkl",
        ),
        "baseline_csv": PATH_BASELINE_HUMAN,
        "color": "#ff7f0e",
        "label": "Train: Human $\\to$ Test: Human",
    },
}


# =========================================================================
# Dispersion bands. `band(d)` takes a DataFrame with bootstrap stats and
# returns (lower, upper) Series, or (None, None) if the needed columns are
# absent (e.g. SD requested but the old cross-domain CSV has no std). ONE
# figure is produced per entry.
# =========================================================================
def _band_ci(d, lo_p=2.5, hi_p=97.5):
    lo, hi = _pct_col(lo_p), _pct_col(hi_p)
    if lo in d.columns and hi in d.columns:
        return d[lo], d[hi]
    if "f1_lower" in d.columns and "f1_upper" in d.columns:  # old CSV (95% only)
        return d["f1_lower"], d["f1_upper"]
    return None, None


def _band_sd(d, k):
    if "f1_std" in d.columns and "f1_mean" in d.columns and d["f1_std"].notna().any():
        return d["f1_mean"] - k * d["f1_std"], d["f1_mean"] + k * d["f1_std"]
    return None, None


DISPERSION_METRICS = [
    {"key": "ci95",  "label": "95% percentile CI [2.5, 97.5]",
     "band": lambda d: _band_ci(d)},
    {"key": "sd196", "label": r"Mean $\pm$ 1.96$\cdot$SD (normal approx.)",
     "band": lambda d: _band_sd(d, 1.96)},
    {"key": "sd1",   "label": r"Mean $\pm$ 1 SD",
     "band": lambda d: _band_sd(d, 1.0)},
    {"key": "sd2",   "label": r"Mean $\pm$ 2 SD",
     "band": lambda d: _band_sd(d, 2.0)},
]


# =========================================================================
# In-domain bootstrap (Option 1: test-set resampling, no retraining)
# =========================================================================
def bootstrap_f1(y_true, y_pred, n_iterations=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED):
    """Bootstrap distribution of macro F1 (stratified when possible)."""
    rng = np.random.RandomState(seed)
    _, counts = np.unique(y_true, return_counts=True)
    stratify_possible = np.all(counts >= 2)

    boot = np.empty(n_iterations)
    for i in range(n_iterations):
        iter_seed = rng.randint(0, 2**32 - 1)
        if stratify_possible:
            y_t, y_p = resample(y_true, y_pred, replace=True,
                                stratify=y_true, random_state=iter_seed)
        else:
            y_t, y_p = resample(y_true, y_pred, replace=True, random_state=iter_seed)
        boot[i] = f1_score(y_t, y_p, average="macro", zero_division=0)
    return boot


def summarize_boot(boot):
    """Summary stats: mean, std (bootstrap SE) and percentiles."""
    out = {"f1_mean": float(np.mean(boot)), "f1_std": float(np.std(boot, ddof=1))}
    for p in PERCENTILES:
        out[_pct_col(p)] = float(np.percentile(boot, p))
    out["f1_lower"] = out[_pct_col(2.5)]
    out["f1_upper"] = out[_pct_col(97.5)]
    return out


def _build_test_split(sub, target, layer):
    """Reconstruct the exact (X_test, y_test) used in baseline training."""
    X_list, y_list = [], []
    for act_row, label in zip(sub["activations"], sub[target]):
        try:
            act = act_row.iloc[layer][ACTIVATION_COL]
        except Exception:
            continue
        if not isinstance(act, np.ndarray):
            continue
        if act.ndim > 1:
            act = act.squeeze()
        if act.shape == (MODEL_DIM,):
            X_list.append(act)
            y_list.append(label)

    if not X_list:
        return None, None
    X = np.stack(X_list)
    y_real = np.array(y_list)

    # Drop rare classes (identical to training).
    counts = Counter(y_real)
    drop = [c for c, n in counts.items() if n < MIN_SAMPLES_REQUIRED]
    if drop:
        keep = ~np.isin(y_real, drop)
        X, y_real = X[keep], y_real[keep]

    if len(X) == 0 or len(np.unique(y_real)) < 2:
        return None, None

    try:
        _, X_test, _, y_test = train_test_split(
            X, y_real, test_size=0.2,
            random_state=SPLIT_RANDOM_STATE, stratify=y_real,
        )
    except ValueError:
        return None, None
    return X_test, y_test


def compute_in_domain_bootstrap():
    """Test-set bootstrap stats for the in-domain baselines (no retraining)."""
    rows = []

    for train_source, cfg in IN_DOMAIN.items():
        data_path = cfg["data_path"]
        dataset = cfg["dataset"]

        if not os.path.exists(data_path):
            print(f"Advertencia: no se encontró el pkl de activaciones para "
                  f"'{train_source}': {data_path}. Se omite su CI in-domain.")
            continue
        if not os.path.exists(cfg["baseline_csv"]):
            print(f"Advertencia: no se encontró el baseline CSV para "
                  f"'{train_source}': {cfg['baseline_csv']}. Se omite su CI in-domain.")
            continue

        nested = pd.read_pickle(data_path)
        base_df = pd.read_csv(cfg["baseline_csv"])

        for target in TAXONOMIES:
            if target not in nested.columns:
                continue
            layers = sorted(base_df[base_df["taxonomy"] == target]["layer"].unique())
            if not layers:
                continue

            mask = nested[target].apply(lambda x: isinstance(x, list) and len(x) > 0)
            sub = nested[mask].copy()
            sub[target] = sub[target].str[0]

            for layer in tqdm(layers, desc=f"Bootstrap {train_source}/{target}"):
                model_path = os.path.join(
                    MODELS_DIR, f"{dataset}_{LLM_USED}_{target}_layer_{layer}.joblib"
                )
                if not os.path.exists(model_path):
                    continue

                X_test, y_test = _build_test_split(sub, target, layer)
                if X_test is None:
                    continue

                clf = joblib.load(model_path)
                y_pred = clf.predict(X_test)
                boot = bootstrap_f1(y_test, y_pred)

                row = {"train_source": train_source, "taxonomy": target,
                       "layer": int(layer)}
                row.update(summarize_boot(boot))
                rows.append(row)

    boot_df = pd.DataFrame(rows)
    boot_df.to_csv(BOOTSTRAP_CSV, index=False)
    print(f"Stats in-domain guardadas: {BOOTSTRAP_CSV}")
    return boot_df


# =========================================================================
# Plotting
# =========================================================================
def _draw_series(x_line, y_line, band_df, metric, color, label, linestyle):
    """Draw one line and (if available) its dispersion band. Returns y-values."""
    plt.plot(x_line, y_line, label=label, color=color, linewidth=2, linestyle=linestyle)
    ys = list(np.asarray(y_line, dtype=float))
    if band_df is not None and not band_df.empty:
        lower, upper = metric["band"](band_df)
        if lower is not None:
            plt.fill_between(band_df["layer"], lower, upper, color=color, alpha=0.18)
            ys += list(np.asarray(lower, dtype=float))
            ys += list(np.asarray(upper, dtype=float))
    return ys


def _finalize(tax, y_values, title_suffix, save_name):
    """Common axis/legend styling. Y-axis anchored at 0 (the floor)."""
    plt.title(f"Cross-Dataset Robustness (Stratified Bootstrap 10k){title_suffix}\n"
              f"Taxonomy: {tax} | Metric: Macro F1", fontsize=14)
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Macro F1-Score", fontsize=12)
    plt.legend(loc="lower right")

    arr = np.asarray(y_values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size:
        y_max = float(np.max(arr))
        pad = max(y_max * 0.08, 0.02)
        plt.ylim(0, min(1.0, y_max + pad))  # 0 always shown; only top adjusts

    plt.grid(True, alpha=0.3)
    save_path = os.path.join(FIGURES_DIR, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Gráfico guardado: {save_path}")


def plot_figure_a(subset, tax):
    """Original figure: cross-domain 95% CI bands + in-domain point estimates."""
    plt.figure(figsize=(12, 7))
    ys = []
    ci = DISPERSION_METRICS[0]  # ci95

    gh = subset[subset["train_source"] == "Generated"].sort_values("layer")
    ys += _draw_series(gh["layer"], gh["f1_mean"], gh, ci, "#1f77b4",
                       "Train: Generated $\\to$ Test: Human", "-")
    hg = subset[subset["train_source"] == "Human"].sort_values("layer")
    ys += _draw_series(hg["layer"], hg["f1_mean"], hg, ci, "#ff7f0e",
                       "Train: Human $\\to$ Test: Generated", "-")

    for cfg in IN_DOMAIN.values():
        if not os.path.exists(cfg["baseline_csv"]):
            print(f"Advertencia: no se encontró baseline en {cfg['baseline_csv']}")
            continue
        base = pd.read_csv(cfg["baseline_csv"])
        base = base[base["taxonomy"] == tax].sort_values("layer")
        ys += _draw_series(base["layer"], base["macro_f1"], None, ci,
                           cfg["color"], cfg["label"], "--")

    _finalize(tax, ys, "", f"cross_test_{tax}.png")


def plot_metric(subset, tax, boot_df, metric):
    """One figure per dispersion metric: all four series get the same band."""
    plt.figure(figsize=(12, 7))
    ys = []

    gh = subset[subset["train_source"] == "Generated"].sort_values("layer")
    ys += _draw_series(gh["layer"], gh["f1_mean"], gh, metric, "#1f77b4",
                       "Train: Generated $\\to$ Test: Human", "-")
    hg = subset[subset["train_source"] == "Human"].sort_values("layer")
    ys += _draw_series(hg["layer"], hg["f1_mean"], hg, metric, "#ff7f0e",
                       "Train: Human $\\to$ Test: Generated", "-")

    for train_source, cfg in IN_DOMAIN.items():
        if not os.path.exists(cfg["baseline_csv"]):
            continue
        base = pd.read_csv(cfg["baseline_csv"])
        base = base[base["taxonomy"] == tax].sort_values("layer")
        b = boot_df[(boot_df["train_source"] == train_source) &
                    (boot_df["taxonomy"] == tax)].sort_values("layer")
        # Plot the bootstrap mean (same estimator the band is built from) so the
        # dashed line tracks the band center. Fall back to the single-split
        # point estimate only when no bootstrap stats are available.
        if not b.empty:
            x_line, y_line, band = b["layer"], b["f1_mean"], b
        else:
            x_line, y_line, band = base["layer"], base["macro_f1"], None
        ys += _draw_series(x_line, y_line, band, metric,
                           cfg["color"], cfg["label"], "--")

    _finalize(tax, ys, f" — band: {metric['label']}",
              f"cross_test_{tax}_{metric['key']}.png")


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    if not os.path.exists(INPUT_CSV):
        print(f"Error: No se encontró el archivo de resultados en {INPUT_CSV}")
        print("Ejecuta primero el script de procesamiento (interdataset_cross_test.py).")
        exit()

    print(f"Cargando resultados de: {INPUT_CSV}")
    results_df = pd.read_csv(INPUT_CSV)

    if "f1_std" not in results_df.columns:
        print("AVISO: el CSV cross-domain es del formato viejo (sin 'f1_std'). "
              "Las bandas SD de las series cross-domain se omitirán (solo línea). "
              "Re-ejecuta interdataset_cross_test.py para habilitarlas.")

    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 12})

    # In-domain bootstrap stats (cached); recompute if old format / missing.
    required_cols = {"f1_mean", "f1_std", _pct_col(2.5), _pct_col(97.5)}
    boot_df = None
    if USE_CACHED_BOOTSTRAP and os.path.exists(BOOTSTRAP_CSV):
        try:
            cached = pd.read_csv(BOOTSTRAP_CSV)
        except pd.errors.EmptyDataError:
            cached = None
            print(f"Caché in-domain vacía/corrupta ({BOOTSTRAP_CSV}); recalculando...")
        if cached is not None and required_cols.issubset(cached.columns):
            boot_df = cached
            print(f"Stats in-domain cargadas desde caché: {BOOTSTRAP_CSV}")
        elif cached is not None:
            print("Caché in-domain en formato viejo; recalculando...")
    if boot_df is None:
        print("Calculando stats in-domain (bootstrap estratificado, sin reentrenar)...")
        boot_df = compute_in_domain_bootstrap()

    for tax in TAXONOMIES:
        subset = results_df[results_df["taxonomy"] == tax]
        if subset.empty:
            print(f"No hay datos para la taxonomía: {tax}")
            continue

        plot_figure_a(subset, tax)               # original figure (95% CI)
        for metric in DISPERSION_METRICS:        # one figure per dispersion metric
            plot_metric(subset, tax, boot_df, metric)

    print("\nVisualización completada.")
