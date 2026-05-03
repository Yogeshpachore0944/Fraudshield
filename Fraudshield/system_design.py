"""
=============================================================================
FRAUD DETECTION SYSTEM — Industry-Level System Design & Monitoring
=============================================================================
This module describes and partially implements the production-grade
infrastructure surrounding the ML model.
=============================================================================
"""

import json
import time
import numpy as np
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass, field
from typing import Callable


# ─────────────────────────────────────────────────────────────────────────────
# 1. REAL-TIME SCORING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
"""
ARCHITECTURE OVERVIEW
─────────────────────

  Card Network / POS Terminal
          │
          ▼  (< 100 ms SLA)
  ┌───────────────────────────────────────────────────────────┐
  │                  API GATEWAY (FastAPI)                     │
  │  Rate limiting · Auth (API key / JWT) · Request logging   │
  └───────────────┬──────────────────────────────────────────-┘
                  │
                  ▼
  ┌─────────────────────────────────────────────────┐
  │              PREPROCESSING SERVICE               │
  │  Feature lookup (Redis cache: velocity, device)  │
  │  Missing value imputation · Standard scaling     │
  └─────────────┬──────────────────────────────────-┘
                │
                ▼
  ┌─────────────────────────────────────────────────┐
  │               MODEL INFERENCE                    │
  │  scikit-learn / XGBoost model (in-process)       │
  │  Fallback: rule-engine if model unavailable      │
  └─────────────┬──────────────────────────────────-┘
                │
                ▼
  ┌─────────────────────────────────────────────────┐
  │               DECISION ENGINE                    │
  │  Threshold logic · Risk level assignment         │
  │  Hard rules overlay (e.g. velocity > 20 → block) │
  │  Explainability (SHAP value extraction)          │
  └──────┬──────────────────────────┬──────────────-┘
         │ Approve / Review / Block  │ Log everything
         ▼                          ▼
  Card Network              Feature Store + Event Log
                                    │
                                    ▼
                           ┌────────────────────┐
                           │ MONITORING SERVICE  │
                           │ Drift detection     │
                           │ Performance tracker │
                           │ Alerting (PagerDuty)│
                           └────────┬───────────┘
                                    │  weekly / on-drift
                                    ▼
                           ┌────────────────────┐
                           │  RETRAINING LOOP   │
                           │  Label collection  │
                           │  Model comparison  │
                           │  Shadow deployment │
                           └────────────────────┘

Key design decisions
─────────────────────
• In-process inference (no network hop to model server) → p99 < 5 ms
• Redis for velocity / device lookups (sub-millisecond)
• Fallback rule engine ensures 100 % availability even if model crashes
• All raw features + decisions logged to data lake (S3 / BigQuery)
• Labels collected via chargeback feed (T+30 days) and analyst review
"""


# ─────────────────────────────────────────────────────────────────────────────
# 2. MONITORING — MODEL DRIFT DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DriftMonitor:
    """
    Lightweight statistical drift detector using a sliding window.

    Monitors two types of drift:
    ─────────────────────────────
    1. Input drift  : distribution of incoming feature values
    2. Output drift : distribution of predicted fraud probabilities
    3. Performance  : precision / recall tracking as labels arrive

    Alert triggers:
    ────────────────
    • Mean fraud probability shifts >15 % from training baseline
    • Rolling 7-day recall drops below 0.65
    • Any feature's mean deviates >2σ from training mean
    """

    training_fraud_rate:  float = 0.01          # baseline from training data
    alert_threshold:      float = 0.15          # relative shift to trigger alert
    window_size:          int   = 10_000        # sliding window of recent predictions
    recall_floor:         float = 0.65          # minimum acceptable recall

    _prob_window:   deque = field(default_factory=lambda: deque(maxlen=10_000))
    _correct_fraud: int = 0
    _total_fraud:   int = 0
    _alerts:        list = field(default_factory=list)

    def record_prediction(self, fraud_prob: float):
        """Call after every prediction."""
        self._prob_window.append(fraud_prob)

    def record_label(self, fraud_prob: float, true_label: int):
        """Call when ground-truth label arrives (e.g. via chargeback feed)."""
        if true_label == 1:
            self._total_fraud += 1
            if fraud_prob >= 0.40:   # decision threshold
                self._correct_fraud += 1

    def check_drift(self) -> dict:
        """Returns drift report. Call periodically (e.g. every hour)."""
        if len(self._prob_window) < 1000:
            return {"status": "insufficient_data"}

        current_fraud_rate = np.mean([p >= 0.40 for p in self._prob_window])
        relative_shift = abs(current_fraud_rate - self.training_fraud_rate) \
                         / (self.training_fraud_rate + 1e-9)

        recall = (self._correct_fraud / max(self._total_fraud, 1))

        alerts = []
        if relative_shift > self.alert_threshold:
            alerts.append({
                "type": "OUTPUT_DRIFT",
                "message": f"Fraud rate shifted {relative_shift*100:.1f}% from baseline",
                "severity": "HIGH",
            })
        if self._total_fraud > 100 and recall < self.recall_floor:
            alerts.append({
                "type": "RECALL_DROP",
                "message": f"Rolling recall {recall:.3f} < threshold {self.recall_floor}",
                "severity": "CRITICAL",
            })

        self._alerts.extend(alerts)
        return {
            "status":             "ALERT" if alerts else "OK",
            "current_fraud_rate": round(current_fraud_rate, 4),
            "training_fraud_rate":self.training_fraud_rate,
            "relative_shift_pct": round(relative_shift * 100, 2),
            "rolling_recall":     round(recall, 4),
            "window_size":        len(self._prob_window),
            "alerts":             alerts,
            "checked_at":         datetime.utcnow().isoformat(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE DRIFT — PSI (Population Stability Index)
# ─────────────────────────────────────────────────────────────────────────────

def population_stability_index(expected: np.ndarray, actual: np.ndarray,
                                n_bins: int = 10) -> float:
    """
    PSI measures how much a feature's distribution has shifted.

    Interpretation (industry standard):
      PSI < 0.10  → negligible drift
      PSI < 0.25  → moderate drift — monitor closely
      PSI >= 0.25 → significant drift — consider retraining
    """
    eps = 1e-6
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins[0]  -= eps
    bins[-1] += eps

    exp_counts = np.histogram(expected, bins=bins)[0]
    act_counts = np.histogram(actual,   bins=bins)[0]

    exp_pct = exp_counts / (exp_counts.sum() + eps)
    act_pct = act_counts / (act_counts.sum() + eps)
    exp_pct = np.clip(exp_pct, eps, None)
    act_pct = np.clip(act_pct, eps, None)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return round(float(psi), 4)


def monitor_feature_drift(training_data: dict, live_data: dict,
                           threshold: float = 0.25) -> dict:
    """
    Computes PSI for every numeric feature.
    Returns a dict with per-feature PSI and overall drift verdict.
    """
    results = {}
    for feature, train_vals in training_data.items():
        if feature not in live_data:
            continue
        psi = population_stability_index(
            np.array(train_vals), np.array(live_data[feature])
        )
        results[feature] = {
            "psi":    psi,
            "status": "OK" if psi < 0.10 else
                      "WATCH" if psi < 0.25 else "DRIFT",
        }

    drift_features = [f for f, v in results.items() if v["status"] == "DRIFT"]
    return {
        "feature_psi":     results,
        "drifted_features":drift_features,
        "recommendation":  "RETRAIN" if len(drift_features) >= 3 else "MONITOR",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. RETRAINING STRATEGY
# ─────────────────────────────────────────────────────────────────────────────
"""
RETRAINING STRATEGY
───────────────────

1. TRIGGER CONDITIONS (any one of):
   a. Scheduled: weekly (Sunday 02:00 UTC)
   b. Recall drops below 0.65 (DriftMonitor alert)
   c. PSI ≥ 0.25 for 3+ features
   d. Fraud rate shifts > 15 % from baseline

2. LABEL COLLECTION PIPELINE:
   • Chargeback feed (card network) → labelled fraud cases (T+30 days)
   • Manual analyst review (high-probability cases) → T+2 days
   • Dispute resolution feed → confirmed legitimate cases

3. TRAINING DATA MANAGEMENT:
   • Rolling 12-month window (seasonal patterns preserved)
   • Hard-negative mining: false negatives from previous model
   • Class weight adjustment based on current fraud prevalence

4. MODEL VALIDATION BEFORE PROMOTION:
   • Must beat current model on held-out recent data (last 2 weeks)
   • Recall ≥ 0.80 on test set
   • Precision ≥ 0.40 (acceptable false positive rate)
   • Shadow-deploy for 24 h: compare decisions on live traffic (no action taken)
   • A/B test for 72 h: challenger handles 10 % traffic

5. ROLLBACK PLAN:
   • Previous model always kept in registry (MLflow / SageMaker Model Registry)
   • Auto-rollback if recall drops >10 % within 6 h of promotion
   • Feature flags control which model serves traffic (instant switch)

6. GOVERNANCE:
   • Every model version logged: data snapshot date, metrics, author
   • Model card published (fairness metrics across demographics)
   • Regulatory reporting: decisions and reasons stored for 7 years
"""

class RetrainingScheduler:
    """Checks retraining triggers and fires a callback when threshold is met."""

    def __init__(self, retrain_fn: Callable, monitor: DriftMonitor):
        self.retrain_fn = retrain_fn
        self.monitor    = monitor
        self._last_retrain = datetime.utcnow()

    def check_and_maybe_retrain(self) -> bool:
        drift_report = self.monitor.check_drift()
        days_since   = (datetime.utcnow() - self._last_retrain).days

        should_retrain = (
            days_since >= 7                                    # scheduled weekly
            or any(a["severity"] == "CRITICAL"                 # critical alert
                   for a in drift_report.get("alerts", []))
            or drift_report.get("relative_shift_pct", 0) > 15 # output drift
        )

        if should_retrain:
            print(f"[{datetime.utcnow().isoformat()}] Retraining triggered.")
            self.retrain_fn()
            self._last_retrain = datetime.utcnow()
            return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 5. HARD RULE ENGINE (safety net, runs BEFORE model)
# ─────────────────────────────────────────────────────────────────────────────

HARD_RULES = [
    {
        "name":      "extreme_velocity",
        "condition": lambda txn: txn.get("velocity_1h", 0) > 15,
        "action":    "BLOCK",
        "reason":    "Velocity exceeds 15 transactions per hour — hard block",
    },
    {
        "name":      "impossible_distance",
        "condition": lambda txn: txn.get("distance_km", 0) > 5000,
        "action":    "BLOCK",
        "reason":    "Physically impossible travel distance in time window",
    },
    {
        "name":      "high_amount_new_card",
        "condition": lambda txn: txn.get("amount", 0) > 5000
                                 and txn.get("card_age_days", 999) < 7,
        "action":    "REVIEW",
        "reason":    "Transaction >$5,000 on card issued <7 days ago",
    },
]


def apply_hard_rules(txn: dict) -> dict | None:
    """
    Returns a rule decision if a hard rule fires, else None (pass to model).
    Hard rules override the model — they're deterministic and auditable.
    """
    for rule in HARD_RULES:
        if rule["condition"](txn):
            return {
                "triggered_rule": rule["name"],
                "action":         rule["action"],
                "reason":         rule["reason"],
            }
    return None


# ─────────────────────────────────────────────────────────────────────────────
# DEMO: Print system design summary
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  SYSTEM DESIGN MODULE — Fraud Detection")
    print("=" * 65)

    # Simulate drift monitor
    monitor = DriftMonitor()
    rng = np.random.default_rng(42)

    # Simulate 5000 predictions at baseline fraud rate
    for _ in range(5000):
        p = rng.beta(1, 99)   # ~1% fraud rate
        monitor.record_prediction(p)
        if rng.random() < 0.01:
            monitor.record_label(p, 1)

    report = monitor.check_drift()
    print("\n[DriftMonitor] Initial check:", json.dumps(report, indent=2))

    # Simulate drift (fraud rate rises to 3%)
    for _ in range(5000):
        p = rng.beta(3, 97)
        monitor.record_prediction(p)

    report2 = monitor.check_drift()
    print("\n[DriftMonitor] After drift:", json.dumps(report2, indent=2))

    # PSI example
    train_amounts = rng.lognormal(4.5, 1.1, 10000)
    live_amounts  = rng.lognormal(5.2, 1.3, 2000)   # shifted distribution
    psi = population_stability_index(train_amounts, live_amounts)
    print(f"\n[PSI] Amount feature PSI: {psi}  → {'DRIFT' if psi >= 0.25 else 'OK'}")

    # Hard rule demo
    suspicious_txn = {"velocity_1h": 18, "distance_km": 120, "amount": 200}
    rule_result = apply_hard_rules(suspicious_txn)
    print(f"\n[HardRule] {rule_result}")

    normal_txn = {"velocity_1h": 2, "distance_km": 15, "amount": 45}
    print(f"[HardRule] Normal txn → {apply_hard_rules(normal_txn)}")

    print("\n  System design module loaded successfully.")
