"""Walk-forward cross-validation for the two-stage calibration pipeline.

Expanding window: train on days [1..d], test on day d+1.
Minimum training window: 7 days.

Reports:
  - Out-of-sample log-loss for each model (Gaussian, Fixed-t)
  - Parameter stability across folds
  - Comparison: in-sample vs out-of-sample performance
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from pricing.models.gaussian import GaussianModel
from pricing.models.fixed_t import FixedTModel
from pricing.calibrate import calibrate_vol, calibrate_tail_fixed, log_loss


MIN_TRAIN_DAYS = 7


def _make_fixed_t(vol_params):
    """Create FixedTModel with injected vol params (bypass JSON loading)."""
    m = object.__new__(FixedTModel)
    m.c = vol_params["c"]
    m.beta = vol_params["beta"]
    m.alpha = vol_params["alpha"]
    m.lam = vol_params["lam"]
    return m


def _predict_ll(model, params, df):
    """Predict on a dataset and return log-loss."""
    S = df["S"].values.astype(np.float64)
    K = df["K"].values.astype(np.float64)
    tau = df["tau"].values.astype(np.float64)
    y = df["y"].values.astype(np.float64)
    features = {f: df[f].values.astype(np.float64) for f in model.required_features()}
    p = model.predict(params, S, K, tau, features)
    return log_loss(y, p)


def _market_ll(model, params, df):
    """Per-market log-loss (equal weight per market)."""
    S = df["S"].values.astype(np.float64)
    K = df["K"].values.astype(np.float64)
    tau = df["tau"].values.astype(np.float64)
    y = df["y"].values.astype(np.float64)
    features = {f: df[f].values.astype(np.float64) for f in model.required_features()}
    p = model.predict(params, S, K, tau, features)
    losses = -(y * np.log(np.clip(p, 1e-9, None)) + (1 - y) * np.log(np.clip(1 - p, 1e-9, None)))
    market_ids = df["market_id"].values
    unique = np.unique(market_ids)
    return float(np.mean([np.mean(losses[market_ids == m]) for m in unique]))


def run_cv():
    # Load dataset
    dataset = pd.read_parquet("pricing/output/calibration_dataset.parquet")
    dataset["day"] = dataset["market_id"].str[4:12]
    days = sorted(dataset["day"].unique())
    n_days = len(days)

    print(f"Dataset: {len(dataset):,} rows, {dataset['market_id'].nunique()} markets, {n_days} days")
    print(f"Walk-forward CV: train on [1..d], test on d+1  (min train = {MIN_TRAIN_DAYS} days)")
    print(f"Test folds: {n_days - MIN_TRAIN_DAYS}\n")

    # Storage
    results = []
    vol_params_history = defaultdict(list)
    nu_history = []

    y_all = dataset["y"].values.astype(np.float64)
    baseline_rate = y_all.mean()

    for fold_idx in range(MIN_TRAIN_DAYS, n_days):
        train_days = days[:fold_idx]
        test_day = days[fold_idx]

        train = dataset[dataset["day"].isin(train_days)]
        test = dataset[dataset["day"] == test_day]

        n_train_markets = train["market_id"].nunique()
        n_test_markets = test["market_id"].nunique()

        # --- Stage 1: QLIKE ---
        model_gauss = GaussianModel()
        result_vol = calibrate_vol(model_gauss, train, objective="qlike",
                                   verbose=False, output_dir=None)
        vp = result_vol.params

        for k, v in vp.items():
            vol_params_history[k].append(v)

        # --- Stage 2: Fixed-nu ---
        model_fixed = _make_fixed_t(vp)
        result_fixed = calibrate_tail_fixed(model_fixed, train,
                                            verbose=False, output_dir=None)
        nu_history.append(result_fixed.params["nu"])

        # --- Evaluate on test ---
        ll_train_gauss = _market_ll(model_gauss, vp, train)
        ll_test_gauss = _market_ll(model_gauss, vp, test)

        ll_train_fixed = _market_ll(model_fixed, result_fixed.params, train)
        ll_test_fixed = _market_ll(model_fixed, result_fixed.params, test)

        ll_test_baseline = log_loss(
            test["y"].values.astype(np.float64),
            np.full(len(test), train["y"].values.astype(np.float64).mean())
        )

        results.append({
            "fold": fold_idx - MIN_TRAIN_DAYS,
            "test_day": test_day,
            "n_train_days": len(train_days),
            "n_train_markets": n_train_markets,
            "n_test_markets": n_test_markets,
            "ll_train_gauss": ll_train_gauss,
            "ll_test_gauss": ll_test_gauss,
            "ll_train_fixed": ll_train_fixed,
            "ll_test_fixed": ll_test_fixed,
            "ll_test_baseline": ll_test_baseline,
            "vol_params": vp.copy(),
            "nu_fixed": result_fixed.params["nu"],
        })

        imp_fixed = (ll_test_baseline - ll_test_fixed) / ll_test_baseline * 100
        overfit_fixed = ll_test_fixed - ll_train_fixed

        print(f"  Fold {fold_idx - MIN_TRAIN_DAYS:2d}  test={test_day}  "
              f"train={len(train_days):2d}d  "
              f"Gauss: {ll_test_gauss:.4f}  "
              f"Fixed-t: {ll_test_fixed:.4f}  "
              f"base: {ll_test_baseline:.4f}  "
              f"imp: {imp_fixed:+.1f}%  "
              f"gap: {overfit_fixed:+.4f}")

    # =====================================================================
    # Summary
    # =====================================================================
    df_res = pd.DataFrame(results)

    print(f"\n{'='*70}")
    print(f"CROSS-VALIDATION SUMMARY ({len(df_res)} folds)")
    print(f"{'='*70}")

    for model_name, train_col, test_col in [
        ("Gaussian", "ll_train_gauss", "ll_test_gauss"),
        ("Fixed-t", "ll_train_fixed", "ll_test_fixed"),
    ]:
        train_mean = df_res[train_col].mean()
        test_mean = df_res[test_col].mean()
        test_se = df_res[test_col].std() / np.sqrt(len(df_res))
        base_mean = df_res["ll_test_baseline"].mean()
        imp = (base_mean - test_mean) / base_mean * 100
        gap = test_mean - train_mean
        print(f"\n  {model_name}:")
        print(f"    Train LL (mean):  {train_mean:.6f}")
        print(f"    Test  LL (mean):  {test_mean:.6f} +/- {test_se:.6f}")
        print(f"    Baseline (mean):  {base_mean:.6f}")
        print(f"    OOS improvement:  {imp:+.1f}%")
        print(f"    Overfit gap:      {gap:+.6f}  ({gap/train_mean*100:+.1f}%)")

    # Full-sample comparison
    print(f"\n  Full-sample (for reference):")
    model_gauss_full = GaussianModel()
    res_full = calibrate_vol(model_gauss_full, dataset, verbose=False, output_dir=None)
    ll_full_gauss = _market_ll(model_gauss_full, res_full.params, dataset)
    model_fixed_full = _make_fixed_t(res_full.params)
    res_fixed_full = calibrate_tail_fixed(model_fixed_full, dataset, verbose=False, output_dir=None)
    ll_full_fixed = _market_ll(model_fixed_full, res_fixed_full.params, dataset)
    ll_full_base = log_loss(y_all, np.full(len(y_all), baseline_rate))
    imp_full = (ll_full_base - ll_full_fixed) / ll_full_base * 100
    print(f"    Fixed-t LL: {ll_full_fixed:.6f}  ({imp_full:+.1f}% vs baseline)")

    # =====================================================================
    # Parameter stability
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"PARAMETER STABILITY")
    print(f"{'='*70}")

    print(f"\n  Vol params (Stage 1):")
    for k in ["c", "beta", "alpha", "lam"]:
        vals = np.array(vol_params_history[k])
        print(f"    {k:6s}: mean={np.mean(vals):+.4f}  std={np.std(vals):.4f}  "
              f"range=[{np.min(vals):+.4f}, {np.max(vals):+.4f}]  "
              f"CV={np.std(vals)/abs(np.mean(vals))*100:.1f}%")

    print(f"\n  Fixed-nu (Stage 2):")
    vals = np.array(nu_history)
    print(f"    nu    : mean={np.mean(vals):.2f}  std={np.std(vals):.2f}  "
          f"range=[{np.min(vals):.2f}, {np.max(vals):.2f}]  "
          f"CV={np.std(vals)/np.mean(vals)*100:.1f}%")

    # =====================================================================
    # Diagnostic plot
    # =====================================================================
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle("Walk-Forward Cross-Validation Diagnostics", fontsize=14, fontweight="bold")

    c_gauss = "#95a5a6"
    c_fixed = "#27ae60"
    c_base = "#e74c3c"

    # Panel 1: OOS log-loss by fold
    ax = axes[0, 0]
    x = np.arange(len(df_res))
    ax.plot(x, df_res["ll_test_gauss"], "o-", color=c_gauss, ms=4, label="Gaussian")
    ax.plot(x, df_res["ll_test_fixed"], "s-", color=c_fixed, ms=4, label="Fixed-t")
    ax.plot(x, df_res["ll_test_baseline"], "x--", color=c_base, ms=4, label="Baseline")
    ax.set_xlabel("Fold (test day)")
    ax.set_ylabel("OOS Log-Loss")
    ax.set_title("Out-of-Sample LL by Fold")
    ax.legend(fontsize=7)
    ax.set_xticks(x[::3])
    ax.set_xticklabels([df_res["test_day"].iloc[i][4:] for i in range(0, len(df_res), 3)],
                       rotation=45, fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 2: Train vs Test LL (overfit diagnostic)
    ax = axes[0, 1]
    ax.scatter(df_res["ll_train_fixed"], df_res["ll_test_fixed"],
               c=c_fixed, s=30, zorder=3, label="Fixed-t")
    ax.scatter(df_res["ll_train_gauss"], df_res["ll_test_gauss"],
               c=c_gauss, s=30, zorder=3, label="Gaussian")
    lims = [min(df_res["ll_train_fixed"].min(), df_res["ll_test_fixed"].min()) - 0.02,
            max(df_res["ll_train_fixed"].max(), df_res["ll_test_fixed"].max()) + 0.02]
    ax.plot(lims, lims, "k--", lw=1, alpha=0.5)
    ax.set_xlabel("Train LL")
    ax.set_ylabel("Test LL")
    ax.set_title("Train vs Test LL (diagonal = no overfit)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 3: OOS improvement over baseline
    ax = axes[0, 2]
    imp_gauss_oos = (df_res["ll_test_baseline"] - df_res["ll_test_gauss"]) / df_res["ll_test_baseline"] * 100
    imp_fixed_oos = (df_res["ll_test_baseline"] - df_res["ll_test_fixed"]) / df_res["ll_test_baseline"] * 100
    ax.bar(x - 0.15, imp_gauss_oos, width=0.3, color=c_gauss, label="Gaussian")
    ax.bar(x + 0.15, imp_fixed_oos, width=0.3, color=c_fixed, label="Fixed-t")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Fold")
    ax.set_ylabel("OOS improvement vs baseline (%)")
    ax.set_title("OOS % Improvement per Fold")
    ax.legend(fontsize=7)
    ax.set_xticks(x[::3])
    ax.set_xticklabels([df_res["test_day"].iloc[i][4:] for i in range(0, len(df_res), 3)],
                       rotation=45, fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 4: Vol parameter trajectories
    ax = axes[1, 0]
    colors_p = ["#e74c3c", "#2980b9", "#27ae60", "#8e44ad"]
    for i, k in enumerate(["c", "beta", "alpha", "lam"]):
        vals = np.array(vol_params_history[k])
        ax.plot(range(len(vals)), vals, "o-", ms=3, color=colors_p[i], label=k)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Parameter value")
    ax.set_title("Vol Parameters (Stage 1) Stability")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 5: nu trajectory
    ax = axes[1, 1]
    ax.plot(range(len(nu_history)), nu_history, "o-", ms=4, color=c_fixed, label="Fixed nu")
    ax.set_xlabel("Fold")
    ax.set_ylabel("nu")
    ax.set_title("Fixed-nu Stability")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 6: Summary table
    ax = axes[1, 2]
    ax.axis("off")

    base_mean = df_res["ll_test_baseline"].mean()
    rows = [
        ["", "Gaussian", "Fixed-t"],
        ["OOS LL", f"{df_res['ll_test_gauss'].mean():.4f}",
         f"{df_res['ll_test_fixed'].mean():.4f}"],
        ["OOS vs base",
         f"{(base_mean - df_res['ll_test_gauss'].mean()) / base_mean * 100:+.1f}%",
         f"{(base_mean - df_res['ll_test_fixed'].mean()) / base_mean * 100:+.1f}%"],
        ["Train LL", f"{df_res['ll_train_gauss'].mean():.4f}",
         f"{df_res['ll_train_fixed'].mean():.4f}"],
        ["Overfit gap",
         f"{(df_res['ll_test_gauss'].mean() - df_res['ll_train_gauss'].mean()):+.4f}",
         f"{(df_res['ll_test_fixed'].mean() - df_res['ll_train_fixed'].mean()):+.4f}"],
        ["", "", ""],
        ["Full-sample", f"{ll_full_gauss:.4f}", f"{ll_full_fixed:.4f}"],
        ["Full vs base", "",  f"{imp_full:+.1f}%"],
        ["N params", "4", "4+1"],
    ]
    table = ax.table(cellText=rows, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    for j in range(3):
        table[0, j].set_text_props(fontweight="bold")
        table[0, j].set_facecolor("#ecf0f1")
    for i in range(1, len(rows)):
        table[i, 1].set_facecolor("#fadbd8")
        table[i, 2].set_facecolor("#d5f5e3")
    ax.set_title("Summary", pad=20)

    plt.tight_layout()
    out_path = "pricing/output/cross_validation.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")
    plt.close()


if __name__ == "__main__":
    run_cv()
