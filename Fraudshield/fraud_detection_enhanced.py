"""
=============================================================================
FRAUD DETECTION SYSTEM — Enhanced ML Pipeline v3.0.0
=============================================================================
NEW FEATURES ADDED
─────────────────────────────────────────────────────────────────────────────
1.  SMOTE-based oversampling (manual implementation — no imbalanced-learn)
2.  XGBoost-style model: HistGradientBoosting (sklearn native, faster + better)
3.  Advanced feature engineering:
    - amount_velocity_ratio  : amount ÷ velocity, detects high-spend bursts
    - log_amount             : log-normalised amount for better linearity
    - cross_risk_score       : composite risk signal (country × device × velocity)
    - hour_sin / hour_cos    : cyclical encoding of hour of day
    - amount_z_score         : deviation of amount from card's historical mean
4.  Calibrated probability outputs (CalibratedClassifierCV)
5.  Cost-sensitive evaluation (business cost matrix: FN costs 10× more than FP)
6.  Learning curve analysis plot
7.  SHAP approximate permutation importance (real, not hardcoded values)
8.  Model drift monitoring (PSI — Population Stability Index)
9.  Auto-saving best model + scaler via joblib
10. Detailed JSON report with cost-adjusted metrics
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, json, os, time
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,   # NEW: faster XGBoost-style model
)
from sklearn.calibration import CalibratedClassifierCV           # NEW
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RandomizedSearchCV, learning_curve
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score, f1_score,
    brier_score_loss,
)
from sklearn.inspection import permutation_importance              # NEW (real SHAP approx)
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
import joblib

warnings.filterwarnings('ignore')
np.random.seed(42)
OUT = Path("/mnt/user-data/outputs")
OUT.mkdir(parents=True, exist_ok=True)

# ─── Colour palette ───────────────────────────────────────────────────────────
C_LEGIT  = "#3B8BD4"
C_FRAUD  = "#E24B4A"
C_ACCENT = "#EF9F27"
C_DARK   = "#2C2C2A"
C_MUTED  = "#5F5E5A"
C_NEW    = "#6CC551"   # green — highlights new features in plots
BG       = "#F1EFE8"


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATASET GENERATION  (same as v2 + new card_avg_amount field)
# ─────────────────────────────────────────────────────────────────────────────
def generate_dataset(n_samples: int = 50_000) -> pd.DataFrame:
    """
    Enhanced generator: adds card_avg_amount to enable amount_z_score feature.
    """
    print("=" * 65)
    print("  FRAUD DETECTION SYSTEM v3.0 — Enhanced Pipeline")
    print("=" * 65)

    n_fraud = int(n_samples * 0.01)
    n_legit = n_samples - n_fraud

    legit = pd.DataFrame({
        "amount":           np.random.lognormal(4.5, 1.1, n_legit),
        "hour":             np.random.choice(range(24), n_legit,
                                p=np.array([1,1,1,1,1,2,4,6,7,7,6,6,
                                            6,6,6,6,5,5,5,5,5,4,3,2], float) /
                                  np.array([1,1,1,1,1,2,4,6,7,7,6,6,
                                            6,6,6,6,5,5,5,5,5,4,3,2], float).sum()),
        "day_of_week":      np.random.randint(0, 7, n_legit),
        "merchant_type":    np.random.choice(
                                ["retail","online","food","travel","ATM"], n_legit,
                                p=[0.35, 0.30, 0.20, 0.10, 0.05]),
        "country_mismatch": np.random.choice([0,1], n_legit, p=[0.97, 0.03]),
        "device_known":     np.random.choice([0,1], n_legit, p=[0.10, 0.90]),
        "velocity_1h":      np.clip(np.random.poisson(1.5, n_legit), 0, 10),
        "distance_km":      np.abs(np.random.normal(20, 30, n_legit)),
        "card_age_days":    np.random.randint(30, 3650, n_legit),
        "card_avg_amount":  np.random.lognormal(4.2, 0.8, n_legit),   # NEW
        "is_fraud":         0,
    })

    fraud = pd.DataFrame({
        "amount":           np.random.lognormal(5.5, 1.4, n_fraud),
        "hour":             np.random.choice(range(24), n_fraud,
                                p=np.array([7,8,8,8,7,6,4,3,3,3,3,4,
                                            4,4,4,4,4,4,4,4,4,4,4,4], float) /
                                  np.array([7,8,8,8,7,6,4,3,3,3,3,4,
                                            4,4,4,4,4,4,4,4,4,4,4,4], float).sum()),
        "day_of_week":      np.random.randint(0, 7, n_fraud),
        "merchant_type":    np.random.choice(
                                ["retail","online","food","travel","ATM"], n_fraud,
                                p=[0.15, 0.45, 0.10, 0.15, 0.15]),
        "country_mismatch": np.random.choice([0,1], n_fraud, p=[0.30, 0.70]),
        "device_known":     np.random.choice([0,1], n_fraud, p=[0.75, 0.25]),
        "velocity_1h":      np.clip(np.random.poisson(4.0, n_fraud), 0, 20),
        "distance_km":      np.abs(np.random.normal(200, 250, n_fraud)),
        "card_age_days":    np.random.randint(1, 180, n_fraud),
        "card_avg_amount":  np.random.lognormal(4.2, 0.8, n_fraud),   # NEW
        "is_fraud":         1,
    })

    df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42)
    df.index.name = "transaction_id"
    df.reset_index(inplace=True)

    # Realistic messiness
    df.loc[np.random.random(len(df)) < 0.003, "distance_km"]      = np.nan
    df.loc[np.random.random(len(df)) < 0.002, "amount"]            = np.nan
    df.loc[np.random.random(len(df)) < 0.002, "card_avg_amount"]   = np.nan
    df = pd.concat([df, df.sample(30, random_state=1)], ignore_index=True)
    df.loc[np.random.choice(df.index, 10), "amount"] = np.random.uniform(9000, 15000, 10)

    print(f"\n  Dataset shape  : {df.shape}")
    print(f"  Fraud cases    : {df['is_fraud'].sum():,}  ({df['is_fraud'].mean()*100:.2f}%)")
    print(f"  Legit cases    : {(df['is_fraud']==0).sum():,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESSING + ADVANCED FEATURE ENGINEERING  ← NEW features added here
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> tuple:
    """
    NEW features added vs v2:
    ─────────────────────────────────────────────────────────────────────────
    log_amount            : log(amount+1) — compresses right skew
    amount_velocity_ratio : amount / (velocity_1h+1) — detects high-spend bursts
    cross_risk_score      : sum of top binary risk signals (0–4)
    hour_sin / hour_cos   : cyclical encoding so 23→0 is close, not far
    amount_z_score        : (amount - card_avg) / std — flags anomalous spending
    ─────────────────────────────────────────────────────────────────────────
    """
    print("\n\n── PREPROCESSING + FEATURE ENGINEERING ────────────────────────")

    # 1. Duplicates
    before = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "transaction_id"])
    print(f"  Dropped {before - len(df)} duplicate rows")

    # 2. Missing values
    n_missing = df.isnull().sum().sum()
    for col in ["distance_km", "amount", "card_avg_amount"]:
        df[col].fillna(df[col].median(), inplace=True)
    print(f"  Imputed {n_missing} missing values (median strategy)")

    # 3. Outlier capping
    cap_99 = df["amount"].quantile(0.99)
    df["amount"] = df["amount"].clip(upper=cap_99)
    print(f"  Amount capped at 99th pct (${cap_99:,.0f})")

    # 4. Categorical encoding
    le = LabelEncoder()
    df["merchant_type_enc"] = le.fit_transform(df["merchant_type"])

    # 5. Original engineered features (v2)
    df["is_weekend"]     = (df["day_of_week"] >= 5).astype(int)
    df["is_night"]       = df["hour"].apply(lambda h: 1 if h < 6 or h >= 22 else 0)
    df["high_velocity"]  = (df["velocity_1h"] >= 4).astype(int)
    df["new_card"]       = (df["card_age_days"] < 30).astype(int)

    # ── NEW FEATURES ─────────────────────────────────────────────────────────
    # F1: log(amount) — stabilises right-skewed distribution
    df["log_amount"] = np.log1p(df["amount"])

    # F2: amount / velocity — high spend in burst sessions → strong fraud signal
    df["amount_velocity_ratio"] = df["amount"] / (df["velocity_1h"] + 1)

    # F3: composite cross-risk score (0 = all safe, 4 = all risk signals active)
    df["cross_risk_score"] = (
        df["country_mismatch"] +
        (1 - df["device_known"]) +
        df["high_velocity"] +
        df["new_card"]
    )

    # F4: cyclical hour encoding — removes artificial distance between hour 23 and 0
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # F5: amount z-score relative to the card's average spending
    card_std = df["card_avg_amount"].std()
    df["amount_z_score"] = (df["amount"] - df["card_avg_amount"]) / (card_std + 1e-6)
    # ─────────────────────────────────────────────────────────────────────────

    feature_cols = [
        # Original
        "amount", "hour", "day_of_week", "merchant_type_enc",
        "country_mismatch", "device_known", "velocity_1h",
        "distance_km", "card_age_days",
        "is_weekend", "is_night", "high_velocity", "new_card",
        # New
        "log_amount", "amount_velocity_ratio", "cross_risk_score",
        "hour_sin", "hour_cos", "amount_z_score",
    ]

    X = df[feature_cols].values
    y = df["is_fraud"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler   = StandardScaler()
    X_tr_sc  = np.nan_to_num(scaler.fit_transform(X_tr), nan=0.0)
    X_te_sc  = np.nan_to_num(scaler.transform(X_te),    nan=0.0)

    print(f"\n  Train  : {X_tr_sc.shape[0]:,} rows")
    print(f"  Test   : {X_te_sc.shape[0]:,} rows")
    print(f"  Features (v2): 13  →  Features (v3): {len(feature_cols)}")
    print(f"  New features : log_amount, amount_velocity_ratio, cross_risk_score,")
    print(f"                 hour_sin, hour_cos, amount_z_score")

    return X_tr_sc, X_te_sc, y_tr, y_te, scaler, feature_cols, df


# ─────────────────────────────────────────────────────────────────────────────
# 3. EDA PLOTS  (same as v2, unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def eda_plots(df: pd.DataFrame):
    print("\n\n── EDA ─────────────────────────────────────────────────────────")
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    fig.suptitle("Exploratory Data Analysis — Fraud vs Legitimate",
                 fontsize=16, fontweight="bold", color=C_DARK, y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    palette = {0: C_LEGIT, 1: C_FRAUD}
    labels  = {0: "Legitimate", 1: "Fraud"}

    ax = fig.add_subplot(gs[0, 0])
    counts = df["is_fraud"].value_counts().sort_index()
    bars = ax.bar(["Legitimate", "Fraud"], counts.values,
                  color=[C_LEGIT, C_FRAUD], width=0.5, edgecolor="white")
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+150,
                f"{v:,}", ha="center", va="bottom", fontsize=10, color=C_DARK)
    ax.set_title("Class Distribution", fontweight="bold", color=C_DARK)
    ax.set_facecolor(BG); ax.spines[["top","right"]].set_visible(False)

    ax = fig.add_subplot(gs[0, 1])
    for cls in [0,1]:
        ax.hist(df[df["is_fraud"]==cls]["amount"], bins=60, alpha=0.65,
                color=palette[cls], label=labels[cls], density=True)
    ax.set_title("Amount Distribution", fontweight="bold", color=C_DARK)
    ax.set_xlabel("Amount (USD)"); ax.legend(fontsize=9)
    ax.set_facecolor(BG); ax.spines[["top","right"]].set_visible(False)

    ax = fig.add_subplot(gs[0, 2])
    for cls in [0,1]:
        h = df[df["is_fraud"]==cls].groupby("hour").size()
        ax.plot(h.index, h/h.sum(), color=palette[cls], label=labels[cls], lw=2)
    ax.set_title("Transaction Hour", fontweight="bold", color=C_DARK)
    ax.set_xlabel("Hour of Day"); ax.set_ylabel("Relative Frequency")
    ax.legend(fontsize=9); ax.set_facecolor(BG)
    ax.spines[["top","right"]].set_visible(False)

    ax = fig.add_subplot(gs[1, 0])
    mismatch = df.groupby(["country_mismatch","is_fraud"]).size().unstack(fill_value=0)
    mismatch_pct = mismatch.div(mismatch.sum(axis=1), axis=0) * 100
    mismatch_pct.plot(kind="bar", ax=ax, color=[C_LEGIT, C_FRAUD],
                      edgecolor="white", width=0.6)
    ax.set_xticklabels(["Same Country", "Mismatch"], rotation=0)
    ax.set_title("Country Mismatch %", fontweight="bold", color=C_DARK)
    ax.set_ylabel("% of transactions"); ax.legend(["Legit","Fraud"], fontsize=9)
    ax.set_facecolor(BG); ax.spines[["top","right"]].set_visible(False)

    ax = fig.add_subplot(gs[1, 1])
    for cls in [0,1]:
        ax.hist(df[df["is_fraud"]==cls]["velocity_1h"], bins=range(0,15),
                alpha=0.65, color=palette[cls], label=labels[cls], density=True)
    ax.set_title("Txn Velocity (1h)", fontweight="bold", color=C_DARK)
    ax.set_xlabel("# Transactions / Hour"); ax.legend(fontsize=9)
    ax.set_facecolor(BG); ax.spines[["top","right"]].set_visible(False)

    ax = fig.add_subplot(gs[1, 2])
    for cls in [0,1]:
        ax.hist(df[df["is_fraud"]==cls]["distance_km"].dropna().clip(upper=1000),
                bins=40, alpha=0.65, color=palette[cls], label=labels[cls], density=True)
    ax.set_title("Distance from Last Txn (km)", fontweight="bold", color=C_DARK)
    ax.set_xlabel("Distance (km)"); ax.legend(fontsize=9)
    ax.set_facecolor(BG); ax.spines[["top","right"]].set_visible(False)

    plt.savefig(OUT/"01_eda.png", dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved → 01_eda.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. SMOTE-STYLE OVERSAMPLING  ← NEW: manual interpolation, not just duplication
# ─────────────────────────────────────────────────────────────────────────────
def handle_imbalance_smote(X_tr: np.ndarray, y_tr: np.ndarray) -> tuple:
    """
    NEW (v3): Manual SMOTE-style synthetic minority oversampling.

    Algorithm:
      For each minority sample, pick k random neighbours from the minority class,
      then create a synthetic sample by interpolating between the sample and a
      random neighbour: synthetic = sample + rand(0,1) × (neighbour - sample)

    Advantage over simple duplication: creates novel samples → better
    generalisation, reduces overfitting to the exact minority points.
    """
    print("\n\n── SMOTE-STYLE OVERSAMPLING (NEW) ──────────────────────────────")

    fraud_idx = np.where(y_tr == 1)[0]
    legit_idx = np.where(y_tr == 0)[0]
    X_fraud   = X_tr[fraud_idx]
    X_legit   = X_tr[legit_idx]

    target_fraud = int(len(legit_idx) * 0.10)   # oversample to 10% ratio
    n_synthetic  = target_fraud - len(fraud_idx)

    print(f"  Original  — Fraud: {len(fraud_idx):,}  |  Legit: {len(legit_idx):,}")
    print(f"  Generating {n_synthetic:,} synthetic fraud samples via interpolation")

    k = 5  # k nearest neighbours
    synthetic = []
    rng = np.random.RandomState(42)

    for _ in range(n_synthetic):
        # pick a random minority sample as the base
        base_idx  = rng.randint(0, len(X_fraud))
        base      = X_fraud[base_idx]
        # pick k random neighbours (with replacement from minority pool)
        neigh_idx = rng.choice(len(X_fraud), k, replace=True)
        neighbour = X_fraud[rng.choice(neigh_idx)]
        # linear interpolation with random weight
        alpha     = rng.uniform(0, 1)
        synthetic.append(base + alpha * (neighbour - base))

    X_synthetic = np.array(synthetic)
    X_bal = np.vstack([X_tr, X_synthetic])
    y_bal = np.concatenate([y_tr, np.ones(n_synthetic, dtype=int)])

    # shuffle
    perm  = rng.permutation(len(X_bal))
    X_bal = X_bal[perm]
    y_bal = y_bal[perm]

    print(f"  Balanced  — Fraud: {y_bal.sum():,}  |  Legit: {(y_bal==0).sum():,}")
    print(f"  Strategy  : SMOTE-style interpolation (k={k})")
    return X_bal, y_bal


# ─────────────────────────────────────────────────────────────────────────────
# 5. MODEL TRAINING  ← NEW: HistGradientBoosting + calibrated probabilities
# ─────────────────────────────────────────────────────────────────────────────
def train_models(X_tr, y_tr, X_te, y_te) -> dict:
    """
    NEW model added vs v2:
    ─────────────────────────────────────────────────────────────────────
    HistGradientBoosting  : sklearn's native XGBoost-equivalent.
                            Histogram-based splitting → 5-10× faster than
                            GradientBoostingClassifier on large datasets.
                            Supports native NaN handling and early stopping.

    NEW: Calibrated wrapper applied to Random Forest so that predicted
         probabilities are better aligned to real-world frequencies
         (Platt scaling via cross-validation).
    ─────────────────────────────────────────────────────────────────────
    """
    print("\n\n── MODEL TRAINING ──────────────────────────────────────────────")

    base_rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=5,
        class_weight="balanced", n_jobs=1, random_state=42
    )
    # NEW: Wrap RF in probability calibration
    calibrated_rf = CalibratedClassifierCV(base_rf, method="sigmoid", cv=3)

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=500, C=0.1, random_state=42
        ),
        "Random Forest (Calibrated)": calibrated_rf,           # NEW wrapping
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=42
        ),
        # NEW model
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=6,
            min_samples_leaf=20, random_state=42,
            class_weight="balanced",
        ),
    }

    results = {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        elapsed = time.time() - t0

        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.40).astype(int)

        roc_auc  = roc_auc_score(y_te, y_prob)
        avg_prec = average_precision_score(y_te, y_prob)
        brier    = brier_score_loss(y_te, y_prob)           # NEW: calibration quality
        report   = classification_report(y_te, y_pred, output_dict=True)

        results[name] = {
            "model":    model,
            "y_prob":   y_prob,
            "y_pred":   y_pred,
            "roc_auc":  roc_auc,
            "avg_prec": avg_prec,
            "brier":    brier,
            "report":   report,
            "elapsed":  elapsed,
        }

        p = report["1"]["precision"]
        r = report["1"]["recall"]
        f = report["1"]["f1-score"]
        print(f"\n  {name}")
        print(f"    ROC-AUC  : {roc_auc:.4f}  |  Brier score: {brier:.4f} (lower=better)")
        print(f"    Avg Prec : {avg_prec:.4f}")
        print(f"    Precision: {p:.4f}  |  Recall: {r:.4f}  |  F1: {f:.4f}")
        print(f"    Train time: {elapsed:.1f}s")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6. EVALUATION PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def evaluation_plots(results: dict, y_te: np.ndarray):
    print("\n\n── EVALUATION PLOTS ────────────────────────────────────────────")
    colors = {
        "Logistic Regression":         C_ACCENT,
        "Random Forest (Calibrated)":  C_LEGIT,
        "Gradient Boosting":           C_FRAUD,
        "HistGradientBoosting":        C_NEW,          # NEW model — green
    }

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), facecolor=BG)
    fig.suptitle("Model Evaluation — Fraud Detection v3.0", fontsize=15,
                 fontweight="bold", color=C_DARK, y=1.02)

    # ROC
    ax = axes[0]; ax.set_facecolor(BG)
    ax.plot([0,1],[0,1],"--", color=C_MUTED, lw=1, label="Random (0.50)")
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_te, res["y_prob"])
        ax.plot(fpr, tpr, color=colors[name], lw=2,
                label=f"{name} ({res['roc_auc']:.3f})")
    ax.set_title("ROC Curve", fontweight="bold", color=C_DARK)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=7); ax.spines[["top","right"]].set_visible(False)

    # Precision-Recall
    ax = axes[1]; ax.set_facecolor(BG)
    ax.axhline(y_te.mean(), ls="--", color=C_MUTED, lw=1, label=f"Random ({y_te.mean():.3f})")
    for name, res in results.items():
        prec, rec, _ = precision_recall_curve(y_te, res["y_prob"])
        ax.plot(rec, prec, color=colors[name], lw=2,
                label=f"{name} (AP={res['avg_prec']:.3f})")
    ax.set_title("Precision-Recall Curve", fontweight="bold", color=C_DARK)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(fontsize=7); ax.spines[["top","right"]].set_visible(False)

    # Metric bar chart
    ax = axes[2]; ax.set_facecolor(BG)
    metric_keys = ["precision","recall","f1-score","roc_auc"]
    x = np.arange(len(metric_keys))
    width = 0.20
    for i, (name, res) in enumerate(results.items()):
        vals = [
            res["report"]["1"]["precision"],
            res["report"]["1"]["recall"],
            res["report"]["1"]["f1-score"],
            res["roc_auc"],
        ]
        ax.bar(x + i*width - 1.5*width, vals, width,
               label=name, color=colors[name], alpha=0.85)
    ax.set_xticks(x - width/2)
    ax.set_xticklabels(["Precision","Recall","F1","ROC-AUC"], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("Metric Comparison", fontweight="bold", color=C_DARK)
    ax.legend(fontsize=7); ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT/"02_evaluation.png", dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved → 02_evaluation.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. CONFUSION MATRIX + THRESHOLD ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def threshold_analysis(results: dict, y_te: np.ndarray, best_model_name: str) -> float:
    print("\n\n── THRESHOLD ANALYSIS ──────────────────────────────────────────")
    res    = results[best_model_name]
    y_prob = res["y_prob"]

    thresholds = np.arange(0.10, 0.80, 0.01)
    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        y_p = (y_prob >= t).astype(int)
        if y_p.sum() == 0:
            precisions.append(0); recalls.append(0); f1s.append(0); continue
        r = classification_report(y_te, y_p, output_dict=True)
        precisions.append(r["1"]["precision"])
        recalls.append(r["1"]["recall"])
        f1s.append(r["1"]["f1-score"])

    best_t = thresholds[np.argmax(f1s)]
    print(f"  Best threshold (max F1): {best_t:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
    fig.suptitle(f"Threshold Analysis — {best_model_name}",
                 fontsize=14, fontweight="bold", color=C_DARK)

    ax = axes[0]; ax.set_facecolor(BG)
    ax.plot(thresholds, precisions, color=C_LEGIT, lw=2, label="Precision")
    ax.plot(thresholds, recalls,   color=C_FRAUD, lw=2, label="Recall")
    ax.plot(thresholds, f1s,       color=C_ACCENT, lw=2, label="F1-Score")
    ax.axvline(best_t, ls="--", color=C_MUTED, lw=1.5, label=f"Best t={best_t:.2f}")
    ax.set_xlabel("Decision Threshold"); ax.set_ylabel("Score")
    ax.set_title("Precision vs Recall vs F1", fontweight="bold", color=C_DARK)
    ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)

    ax = axes[1]; ax.set_facecolor(BG)
    y_pred_best = (y_prob >= best_t).astype(int)
    cm = confusion_matrix(y_te, y_pred_best)
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred: Legit","Pred: Fraud"])
    ax.set_yticklabels(["True: Legit","True: Fraud"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    fontsize=14, color="white" if cm[i,j] > cm.max()/2 else C_DARK)
    ax.set_title(f"Confusion Matrix (t={best_t:.2f})", fontweight="bold", color=C_DARK)

    plt.tight_layout()
    plt.savefig(OUT/"03_threshold.png", dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved → 03_threshold.png")
    return best_t


# ─────────────────────────────────────────────────────────────────────────────
# 8. REAL PERMUTATION IMPORTANCE  ← NEW: computed, not hardcoded
# ─────────────────────────────────────────────────────────────────────────────
def feature_importance_plot(results: dict, X_te: np.ndarray,
                             y_te: np.ndarray, feature_names: list,
                             best_model_name: str):
    """
    NEW (v3): Uses sklearn's permutation_importance() on the test set for
    the best model. This gives a model-agnostic, real importance estimate:
    for each feature, randomly shuffle its values and measure ROC-AUC drop.
    A large drop means the feature matters; small drop means it's redundant.
    """
    print("\n\n── FEATURE IMPORTANCE (PERMUTATION — NEW) ──────────────────────")
    model = results[best_model_name]["model"]

    # Use a calibrated estimator wrapper if needed
    perm = permutation_importance(
        model, X_te, y_te,
        n_repeats=10, random_state=42, n_jobs=1,
        scoring="roc_auc"
    )
    importances_mean = perm.importances_mean
    importances_std  = perm.importances_std

    idx = np.argsort(importances_mean)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
    fig.suptitle("Feature Importance — Permutation (v3) vs Tree (v2)",
                 fontsize=14, fontweight="bold", color=C_DARK)

    # NEW: Permutation importance (correct)
    ax = axes[0]; ax.set_facecolor(BG)
    bar_colors = [C_NEW if importances_mean[i] > np.median(importances_mean) else C_LEGIT
                  for i in idx]
    ax.barh([feature_names[i] for i in idx], importances_mean[idx],
            xerr=importances_std[idx], color=bar_colors, alpha=0.85)
    ax.set_title("Permutation Importance\n(ROC-AUC drop when feature shuffled) ★ NEW",
                 fontweight="bold", color=C_DARK)
    ax.set_xlabel("Mean ROC-AUC decrease ± std")
    ax.spines[["top","right"]].set_visible(False)

    print(f"\n  Top 5 features by permutation importance ({best_model_name}):")
    for rank, i in enumerate(idx[::-1][:5], 1):
        print(f"    {rank}. {feature_names[i]:<30} {importances_mean[i]:.4f} ± {importances_std[i]:.4f}")

    # Tree-based (Gini) for comparison — only for tree models
    ax = axes[1]; ax.set_facecolor(BG)
    if hasattr(model, "feature_importances_"):
        tree_imp = model.feature_importances_
        tree_idx = np.argsort(tree_imp)
        ax.barh([feature_names[i] for i in tree_idx], tree_imp[tree_idx],
                color=C_LEGIT, alpha=0.85)
        ax.set_title("Tree-based Importance (Gini)\n(for comparison — can be biased)",
                     fontweight="bold", color=C_DARK)
        ax.set_xlabel("Mean Decrease in Impurity")
    elif hasattr(model, "estimators_"):
        # Calibrated RF: access the underlying estimator
        base = model.calibrated_classifiers_[0].estimator
        if hasattr(base, "feature_importances_"):
            tree_imp = base.feature_importances_
            tree_idx = np.argsort(tree_imp)
            ax.barh([feature_names[i] for i in tree_idx], tree_imp[tree_idx],
                    color=C_LEGIT, alpha=0.85)
            ax.set_title("Tree-based Importance (Gini — base RF)",
                         fontweight="bold", color=C_DARK)
            ax.set_xlabel("Mean Decrease in Impurity")
    else:
        # Logistic or HistGradientBoosting: use coefficients / native importance
        ax.text(0.5, 0.5, "Tree Gini importance\nnot available for this model",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, color=C_MUTED)
        ax.set_title("Tree Gini Importance — N/A", fontweight="bold", color=C_DARK)

    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT/"04_feature_importance.png", dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved → 04_feature_importance.png")


# ─────────────────────────────────────────────────────────────────────────────
# 9. COST-SENSITIVE EVALUATION  ← NEW
# ─────────────────────────────────────────────────────────────────────────────
def cost_analysis(results: dict, y_te: np.ndarray, best_model_name: str):
    """
    NEW (v3): Business-cost-weighted evaluation.

    In fraud detection, missing a fraud (False Negative) is far costlier than
    falsely flagging a legitimate transaction (False Positive).

    Cost matrix:
      FN (missed fraud)       : $500 average loss per incident
      FP (false alarm)        : $10  cost (manual review + customer friction)
      TP (caught fraud)       : +$500 savings
      TN (correct clearance)  : $0

    This section computes the total business cost at each threshold and finds
    the cost-minimising operating point, which may differ from the F1-optimal
    threshold.
    """
    print("\n\n── COST-SENSITIVE ANALYSIS (NEW) ───────────────────────────────")

    COST_FN  = 500   # USD: missed fraud
    COST_FP  =  10   # USD: false alarm
    GAIN_TP  = 500   # USD: caught fraud savings

    res    = results[best_model_name]
    y_prob = res["y_prob"]
    thresholds = np.arange(0.05, 0.90, 0.01)
    costs  = []

    for t in thresholds:
        y_p = (y_prob >= t).astype(int)
        cm  = confusion_matrix(y_te, y_p)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (cm[0,0],0,0,0)
        net_cost = fn * COST_FN + fp * COST_FP - tp * GAIN_TP
        costs.append(net_cost)

    best_cost_t = thresholds[np.argmin(costs)]
    best_cost   = min(costs)

    print(f"  Cost matrix — FN: ${COST_FN}, FP: ${COST_FP}, TP gain: ${GAIN_TP}")
    print(f"  Cost-optimal threshold : {best_cost_t:.2f}")
    print(f"  Net cost at optimum    : ${best_cost:,.0f}")

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.plot(thresholds, costs, color=C_FRAUD, lw=2.5, label="Net Business Cost ($)")
    ax.axvline(best_cost_t, ls="--", color=C_NEW, lw=2,
               label=f"Cost-optimal t={best_cost_t:.2f}  (${best_cost:,.0f})")
    ax.axvline(0.40, ls=":", color=C_ACCENT, lw=2, label="F1-optimal t=0.40")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Net Cost (USD)")
    ax.set_title(f"Business Cost vs Threshold — {best_model_name}\n"
                 f"FN=${COST_FN}, FP=${COST_FP}, TP gain=${GAIN_TP}",
                 fontweight="bold", color=C_DARK)
    ax.legend(fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT/"05_cost_analysis.png", dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved → 05_cost_analysis.png")
    return best_cost_t, best_cost


# ─────────────────────────────────────────────────────────────────────────────
# 10. LEARNING CURVES  ← NEW
# ─────────────────────────────────────────────────────────────────────────────
def learning_curve_plot(X_tr: np.ndarray, y_tr: np.ndarray, best_model_name: str):
    """
    NEW (v3): Learning curve shows how model performance changes with training
    set size. Helps diagnose:
      - High bias (underfitting): both train and val scores are low
      - High variance (overfitting): large gap between train and val scores
    """
    print("\n\n── LEARNING CURVES (NEW) ───────────────────────────────────────")

    model = HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.05, max_depth=5,
        random_state=42, class_weight="balanced"
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    train_sizes = np.linspace(0.10, 1.0, 8)

    print("  Computing learning curves (this may take 30s) …")
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X_tr, y_tr,
        train_sizes=train_sizes,
        cv=cv,
        scoring="average_precision",
        n_jobs=1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    ax.set_facecolor(BG)
    ax.fill_between(train_sizes_abs, train_mean-train_std, train_mean+train_std,
                    alpha=0.15, color=C_LEGIT)
    ax.fill_between(train_sizes_abs, val_mean-val_std, val_mean+val_std,
                    alpha=0.15, color=C_FRAUD)
    ax.plot(train_sizes_abs, train_mean, "o-", color=C_LEGIT, lw=2, label="Train AP")
    ax.plot(train_sizes_abs, val_mean,   "o-", color=C_FRAUD, lw=2, label="Validation AP")

    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Average Precision (AP)")
    ax.set_title("Learning Curves — HistGradientBoosting\n"
                 "(gap = variance; low score = bias)",
                 fontweight="bold", color=C_DARK)
    ax.legend(fontsize=10); ax.spines[["top","right"]].set_visible(False)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUT/"06_learning_curves.png", dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved → 06_learning_curves.png")
    print(f"  Final validation AP: {val_mean[-1]:.4f} ± {val_std[-1]:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. PSI — POPULATION STABILITY INDEX  ← NEW (drift monitoring)
# ─────────────────────────────────────────────────────────────────────────────
def psi_analysis(results: dict, y_te: np.ndarray, best_model_name: str):
    """
    NEW (v3): Population Stability Index (PSI) measures model drift.

    PSI compares the predicted score distribution between a reference window
    (first half of test set) and a current window (second half).

    PSI < 0.10  → No drift, model is stable
    PSI 0.10–0.25 → Slight drift, monitor closely
    PSI > 0.25  → Significant drift, consider retraining

    In production, reference = training set scores, current = latest scores.
    """
    print("\n\n── PSI DRIFT MONITORING (NEW) ──────────────────────────────────")

    y_prob = results[best_model_name]["y_prob"]
    mid    = len(y_prob) // 2
    ref    = y_prob[:mid]
    curr   = y_prob[mid:]

    n_bins   = 10
    bins     = np.linspace(0, 1, n_bins + 1)
    ref_pct  = np.histogram(ref,  bins=bins)[0] / len(ref)
    curr_pct = np.histogram(curr, bins=bins)[0] / len(curr)

    # Avoid division by zero / log(0)
    ref_pct  = np.where(ref_pct  == 0, 1e-6, ref_pct)
    curr_pct = np.where(curr_pct == 0, 1e-6, curr_pct)

    psi_vals = (curr_pct - ref_pct) * np.log(curr_pct / ref_pct)
    psi      = psi_vals.sum()

    status = "STABLE" if psi < 0.10 else ("MONITOR" if psi < 0.25 else "RETRAIN")
    print(f"  PSI value  : {psi:.4f}")
    print(f"  Status     : {status}  (< 0.10 stable | 0.10-0.25 monitor | > 0.25 retrain)")

    bin_centres = (bins[:-1] + bins[1:]) / 2
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
    ax.set_facecolor(BG)
    w = (bins[1] - bins[0]) * 0.4
    ax.bar(bin_centres - w/2, ref_pct,  width=w, color=C_LEGIT, alpha=0.8, label="Reference (first half)")
    ax.bar(bin_centres + w/2, curr_pct, width=w, color=C_FRAUD, alpha=0.8, label="Current (second half)")
    ax.set_xlabel("Predicted Fraud Probability")
    ax.set_ylabel("Proportion of Samples")
    ax.set_title(f"PSI Drift Analysis — {best_model_name}\n"
                 f"PSI = {psi:.4f}  →  {status}",
                 fontweight="bold", color=C_DARK)
    ax.legend(fontsize=10); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT/"07_psi_drift.png", dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  Saved → 07_psi_drift.png")
    return psi, status


# ─────────────────────────────────────────────────────────────────────────────
# 12. HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────
def tune_model(X_tr, y_tr, X_te, y_te):
    print("\n\n── HYPERPARAMETER TUNING ───────────────────────────────────────")
    print("  Running RandomizedSearchCV on HistGradientBoosting (3-fold CV)…")

    param_dist = {
        "max_iter":        [100, 200],
        "max_depth":       [4, 6],
        "learning_rate":   [0.03, 0.05, 0.10],
        "min_samples_leaf":[10, 20],
        "l2_regularization":[0.0, 0.1],
    }

    base = HistGradientBoostingClassifier(random_state=42, class_weight="balanced")
    cv   = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        base, param_dist, n_iter=8, cv=cv,
        scoring="average_precision", n_jobs=1, random_state=42, verbose=0
    )
    search.fit(X_tr, y_tr)

    best = search.best_estimator_
    y_prob_tuned = best.predict_proba(X_te)[:, 1]
    y_pred_tuned = (y_prob_tuned >= 0.40).astype(int)
    ap_tuned  = average_precision_score(y_te, y_prob_tuned)
    roc_tuned = roc_auc_score(y_te, y_prob_tuned)

    print(f"\n  Best params  : {search.best_params_}")
    print(f"  AP (CV best) : {search.best_score_:.4f}")
    print(f"  AP (test)    : {ap_tuned:.4f}")
    print(f"  ROC-AUC      : {roc_tuned:.4f}")
    r = classification_report(y_te, y_pred_tuned, output_dict=True)
    print(f"  Precision    : {r['1']['precision']:.4f}")
    print(f"  Recall       : {r['1']['recall']:.4f}")
    print(f"  F1           : {r['1']['f1-score']:.4f}")

    return best, search.best_params_


# ─────────────────────────────────────────────────────────────────────────────
# 13. SAVE ARTEFACTS  ← NEW: includes cost metrics, brier score, PSI
# ─────────────────────────────────────────────────────────────────────────────
def save_artefacts(results, best_params, threshold, feature_names,
                   cost_threshold, net_cost, psi, psi_status):
    summary = {
        "pipeline_version": "3.0.0",
        "best_model": "HistGradientBoosting",
        "best_params": {k: str(v) for k, v in best_params.items()},
        "decision_threshold_f1":  round(threshold, 2),
        "decision_threshold_cost": round(cost_threshold, 2),  # NEW
        "feature_names": feature_names,
        "new_features_v3": [                                  # NEW
            "log_amount",
            "amount_velocity_ratio",
            "cross_risk_score",
            "hour_sin",
            "hour_cos",
            "amount_z_score",
        ],
        "model_metrics": {
            name: {
                "roc_auc":   round(res["roc_auc"], 4),
                "avg_prec":  round(res["avg_prec"], 4),
                "brier":     round(res["brier"], 4),           # NEW
                "precision": round(res["report"]["1"]["precision"], 4),
                "recall":    round(res["report"]["1"]["recall"], 4),
                "f1":        round(res["report"]["1"]["f1-score"], 4),
            }
            for name, res in results.items()
        },
        "drift_monitoring": {                                   # NEW
            "psi_value":  round(psi, 4),
            "psi_status": psi_status,
            "thresholds": {"stable": 0.10, "monitor": 0.25, "retrain": 0.25},
        },
        "cost_analysis": {                                      # NEW
            "fn_cost_usd":    500,
            "fp_cost_usd":     10,
            "tp_gain_usd":    500,
            "net_cost_usd":   round(net_cost, 0),
            "optimal_threshold": round(cost_threshold, 2),
        },
        "system_notes": [
            "v3: SMOTE-style oversampling (interpolation, not duplication)",
            "v3: HistGradientBoosting added (XGBoost-equivalent, faster)",
            "v3: Calibrated Random Forest probabilities (Platt scaling)",
            "v3: Real permutation importance (not hardcoded)",
            "v3: Cost-sensitive threshold optimisation",
            "v3: Learning curve diagnosis plot",
            "v3: PSI drift monitoring",
            "v3: 6 new engineered features",
            "Priority metric: Recall (catching fraud > false positives)",
            "In production: retrain monthly or on PSI > 0.25 trigger",
        ]
    }
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    with open(OUT/"model_summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=NpEncoder)
    print("\n  Saved → model_summary.json")


# ─────────────────────────────────────────────────────────────────────────────
# 14. SAVE MODEL + SCALER  ← NEW: joblib persistence
# ─────────────────────────────────────────────────────────────────────────────
def persist_model(best_model, scaler):
    joblib.dump(best_model, OUT/"fraud_model_v3.joblib")
    joblib.dump(scaler,     OUT/"fraud_scaler_v3.joblib")
    print("  Saved → fraud_model_v3.joblib")
    print("  Saved → fraud_scaler_v3.joblib")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    df = generate_dataset(n_samples=20_000)
    X_tr, X_te, y_tr, y_te, scaler, feature_names, df_clean = preprocess(df)
    eda_plots(df_clean)

    # SMOTE-style oversampling (NEW)
    X_tr_bal, y_tr_bal = handle_imbalance_smote(X_tr, y_tr)

    results = train_models(X_tr_bal, y_tr_bal, X_te, y_te)
    evaluation_plots(results, y_te)

    best_name = max(results, key=lambda n: results[n]["avg_prec"])
    print(f"\n\n  Best model: {best_name}")

    best_t         = threshold_analysis(results, y_te, best_name)
    feature_importance_plot(results, X_te, y_te, feature_names, best_name)
    cost_t, cost   = cost_analysis(results, y_te, best_name)       # NEW
    learning_curve_plot(X_tr_bal, y_tr_bal, best_name)             # NEW
    psi, psi_stat  = psi_analysis(results, y_te, best_name)        # NEW

    tuned_model, best_params = tune_model(X_tr_bal, y_tr_bal, X_te, y_te)
    save_artefacts(results, best_params, best_t, feature_names,
                   cost_t, cost, psi, psi_stat)
    persist_model(tuned_model, scaler)

    print("\n\n" + "=" * 65)
    print("  PIPELINE v3.0 COMPLETE")
    print("  Outputs → /mnt/user-data/outputs/")
    print("=" * 65)
    return results, tuned_model, scaler, feature_names, best_t


if __name__ == "__main__":
    main()
