"""
=============================================================================
FRAUD DETECTION API — FastAPI Deployment Layer  (Updated v2.0.0)
=============================================================================
Run:
  Step 1 (first time only):  python fraud_detection_system.py
  Step 2:                    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

Docs: http://localhost:8000/docs   (Swagger UI, auto-generated)
=============================================================================

Changes from v1.0.0
-------------------
  - Fixed: top-level `app` now exposed so uvicorn api_server:app works
  - Fixed: Pydantic v2 compatibility (schema_extra → model_config + json_schema_extra,
            @validator → @field_validator)
  - Added: model persistence via joblib (train once, reload on restart)
  - Added: /reload_model endpoint to hot-swap model without restarting
  - Added: structured startup logging
  - Added: request_count and last_request_time in /health
  - Improved: model_info reads from model_summary.json safely
"""

from __future__ import annotations

import os
import sys
import time
import json
import uuid
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fraud_api")

# ── FastAPI imports ────────────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field, field_validator, model_validator
    import uvicorn
    _FASTAPI_AVAILABLE = True
except ImportError as _err:
    _FASTAPI_AVAILABLE = False
    log.error("FastAPI not installed: %s", _err)
    log.error("Fix: pip install fastapi uvicorn pydantic")
    # Stub classes so the module still parses for syntax checking
    class BaseModel:          # type: ignore
        pass
    class Field:              # type: ignore
        def __init__(self, *a, **kw): pass
    def field_validator(*a, **kw):  # type: ignore
        return lambda f: f
    def model_validator(*a, **kw):  # type: ignore
        return lambda f: f

# ── joblib for model persistence ───────────────────────────────────────────────
try:
    import joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False
    log.warning("joblib not found. Model will not be saved/loaded from disk.")
    log.warning("Fix: pip install joblib")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR          = Path(__file__).parent
# ── DATABASE SETUP ───────────────────────────────────────────────────────────
import sqlite3

DB_PATH = BASE_DIR / "transactions.db"

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS transactions (
    id TEXT PRIMARY KEY,
    amount REAL,
    fraud_probability REAL,
    is_fraud INTEGER,
    risk_level TEXT,
    timestamp TEXT
)
""")
conn.commit()

MODEL_PATH        = BASE_DIR / "fraud_model.joblib"
SCALER_PATH       = BASE_DIR / "fraud_scaler.joblib"
MODEL_SUMMARY_PATH = BASE_DIR / "model_summary.json"


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS  (Pydantic v2 compatible)
# ─────────────────────────────────────────────────────────────────────────────

VALID_MERCHANT_TYPES = {"retail", "online", "food", "travel", "ATM"}

class TransactionRequest(BaseModel):
    """
    Single transaction submitted for real-time fraud scoring.
    All monetary amounts in USD. Boolean fields as 0/1 integers.
    """
    amount:           float = Field(..., gt=0,       description="Transaction amount in USD")
    hour:             int   = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week:      int   = Field(..., ge=0, le=6,  description="0=Monday … 6=Sunday")
    merchant_type:    str   = Field(...,              description="retail | online | food | travel | ATM")
    country_mismatch: int   = Field(..., ge=0, le=1,  description="1 if card country ≠ txn country")
    device_known:     int   = Field(..., ge=0, le=1,  description="1 if device previously seen")
    velocity_1h:      int   = Field(..., ge=0,        description="# transactions past 1 hour")
    distance_km:      float = Field(..., ge=0,        description="Distance from previous txn (km)")
    card_age_days:    int   = Field(..., ge=0,        description="Card age in days")

    # ── Pydantic v2 style validator ───────────────────────────────────────────
    @field_validator("merchant_type")
    @classmethod
    def validate_merchant(cls, v: str) -> str:
        if v not in VALID_MERCHANT_TYPES:
            raise ValueError(f"merchant_type must be one of {VALID_MERCHANT_TYPES}")
        return v

    # ── Pydantic v2 style config (replaces inner Config class) ───────────────
    model_config = {
        "json_schema_extra": {
            "example": {
                "amount":           2340.50,
                "hour":             2,
                "day_of_week":      5,
                "merchant_type":    "online",
                "country_mismatch": 1,
                "device_known":     0,
                "velocity_1h":      5,
                "distance_km":      850.0,
                "card_age_days":    14,
            }
        }
    }


class FraudDecision(BaseModel):
    """Fraud scoring response returned to the caller."""
    transaction_id:    str
    is_fraud:          bool
    fraud_probability: float
    risk_level:        str          # LOW | MEDIUM | HIGH | CRITICAL
    decision_reasons:  list[str]
    processing_ms:     float
    timestamp:         str
    model_version:     str


class HealthResponse(BaseModel):
    status:            str
    model_loaded:      bool
    uptime_seconds:    float
    requests_served:   int
    last_request_time: Optional[str] = None


class BatchRequest(BaseModel):
    transactions: list[TransactionRequest]


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

MERCHANT_MAP = {"retail": 0, "online": 1, "food": 2, "travel": 3, "ATM": 4}


class FraudModel:
    """Lightweight inference wrapper around a trained sklearn model + scaler."""

    VERSION   = "2.0.0"
    THRESHOLD = 0.40    # probability threshold tuned for max-F1

    def __init__(self, model, scaler):
        self.model   = model
        self.scaler  = scaler
        self._start  = time.time()
        self._count  = 0
        self._last_request: Optional[str] = None

    # ── Feature engineering ───────────────────────────────────────────────────
    def _engineer_features(self, req: TransactionRequest) -> np.ndarray:
        """Replicate feature engineering performed during training."""
        merchant_enc  = MERCHANT_MAP.get(req.merchant_type, 0)
        is_weekend    = 1 if req.day_of_week >= 5 else 0
        is_night      = 1 if req.hour < 6 or req.hour >= 22 else 0
        high_velocity = 1 if req.velocity_1h >= 4 else 0
        new_card      = 1 if req.card_age_days < 30 else 0

        row = np.array([[
            req.amount, req.hour, req.day_of_week, merchant_enc,
            req.country_mismatch, req.device_known, req.velocity_1h,
            req.distance_km, req.card_age_days,
            is_weekend, is_night, high_velocity, new_card,
        ]])
        return self.scaler.transform(row)

    # ── Risk bucketing ────────────────────────────────────────────────────────
    @staticmethod
    def _risk_level(prob: float) -> str:
        if prob < 0.20: return "LOW"
        if prob < 0.40: return "MEDIUM"
        if prob < 0.70: return "HIGH"
        return "CRITICAL"

    # ── Human-readable reasons ────────────────────────────────────────────────
    @staticmethod
    def _reasons(req: TransactionRequest) -> list[str]:
        reasons = []
        if req.country_mismatch == 1:
            reasons.append("Transaction country differs from card country")
        if req.device_known == 0:
            reasons.append("Unknown device detected")
        if req.velocity_1h >= 4:
            reasons.append(f"High transaction velocity ({req.velocity_1h} txns/hr)")
        if req.distance_km > 300:
            reasons.append(f"Large geographic distance ({req.distance_km:.0f} km)")
        if req.card_age_days < 30:
            reasons.append("Card is newly issued (<30 days)")
        if req.amount > 1000:
            reasons.append(f"High transaction amount (${req.amount:,.0f})")
        if req.hour < 5 or req.hour >= 22:
            reasons.append("Transaction at unusual hour (late night)")
        if not reasons:
            reasons.append("No significant risk signals detected")
        return reasons

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(self, req: TransactionRequest) -> dict:
        t0       = time.perf_counter()
        X        = self._engineer_features(req)
        prob     = float(self.model.predict_proba(X)[0, 1])
        is_fraud = prob >= self.THRESHOLD
        elapsed  = (time.perf_counter() - t0) * 1000
        self._count += 1
        self._last_request = datetime.now(timezone.utc).isoformat()

        return {
            "transaction_id":    str(uuid.uuid4()),
            "is_fraud":          is_fraud,
            "fraud_probability": round(prob, 4),
            "risk_level":        self._risk_level(prob),
            "decision_reasons":  self._reasons(req),
            "processing_ms":     round(elapsed, 2),
            "timestamp":         self._last_request,
            "model_version":     self.VERSION,
        }

    @property
    def uptime(self) -> float:
        return time.time() - self._start


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING / TRAINING HELPER
# ─────────────────────────────────────────────────────────────────────────────

def load_or_train_model():
    """
    Try to load a pre-trained model from disk.
    If not found, train a new one via fraud_detection_system.py and save it.
    Returns (model, scaler).
    """
    # ── Try loading saved model ───────────────────────────────────────────────
    if _JOBLIB_AVAILABLE and MODEL_PATH.exists() and SCALER_PATH.exists():
        log.info("Loading saved model from %s …", MODEL_PATH)
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        log.info("Model loaded successfully ✓")
        return model, scaler

    # ── No saved model → train from scratch ──────────────────────────────────
    log.info("No saved model found. Running training pipeline …")
    log.info("(This may take 60-120 seconds on first run)")

    # Add current directory to path so we can import the training module
    sys.path.insert(0, str(BASE_DIR))
    try:
        from fraud_detection_system import main as run_pipeline
    except ImportError:
        raise RuntimeError(
            "Cannot import fraud_detection_system.py — make sure it is in the "
            "same directory as api_server.py and run:\n"
            "  python fraud_detection_system.py\n"
            "then restart the server."
        )

    results, tuned_model, scaler, feature_names, threshold = run_pipeline()

    # Pick best model by average precision
    best_name = max(results, key=lambda n: results[n]["avg_prec"])
    model     = results[best_name]["model"]
    log.info("Training complete. Best model: %s  (avg_prec=%.4f)", best_name,
             results[best_name]["avg_prec"])

    # ── Persist to disk ───────────────────────────────────────────────────────
    if _JOBLIB_AVAILABLE:
        joblib.dump(model,  MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        log.info("Model saved → %s", MODEL_PATH)
        log.info("Scaler saved → %s", SCALER_PATH)
    else:
        log.warning("joblib not available — model will NOT be saved to disk.")

    return model, scaler


# ─────────────────────────────────────────────────────────────────────────────
# APP FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def create_app(model, scaler) -> "FastAPI":  # type: ignore
    if not _FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI not installed. Run: pip install fastapi uvicorn")

    engine = FraudModel(model, scaler)

    app = FastAPI(
        title="Fraud Detection API",
        description=(
            "Real-time ML-powered fraud scoring for financial transactions.\n\n"
            "**Quick start:** POST a transaction to `/predict` and receive an "
            "instant fraud score with human-readable reasons.\n\n"
            "**Batch scoring:** Send up to 500 transactions at once via `/batch_predict`."
        ),
        version=FraudModel.VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ─── Routes ───────────────────────────────────────────────────────────────

    @app.get("/", tags=["System"])
    def root():
        """Redirect hint — visit /docs for the interactive UI."""
        return {
            "message": "Fraud Detection API is running.",
            "docs":    "http://localhost:8000/docs",
            "health":  "http://localhost:8000/health",
        }

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    def health():
        """Liveness probe — returns server status and basic stats."""
        return {
            "status":            "ok",
            "model_loaded":      True,
            "uptime_seconds":    round(engine.uptime, 1),
            "requests_served":   engine._count,
            "last_request_time": engine._last_request,
        }

    @app.post("/predict", response_model=FraudDecision, tags=["Scoring"])
    def predict(transaction: TransactionRequest):
        """
        Score a single transaction in real-time.

        Returns a fraud probability (0–1), risk level (LOW/MEDIUM/HIGH/CRITICAL),
        and plain-English reasons suitable for fraud analysts or automated systems.
        """
        try:
            result = engine.predict(transaction)
            # ── STORE TRANSACTION ─────────────────────────────────────
            cursor.execute(
    "INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?)",
    (
        result["transaction_id"],
        transaction.amount,
        result["fraud_probability"],
        int(result["is_fraud"]),
        result["risk_level"],
        result["timestamp"],
    )
)
            conn.commit()
            log.info(
                "txn=%s  prob=%.4f  risk=%s  fraud=%s",
                result["transaction_id"][:8],
                result["fraud_probability"],
                result["risk_level"],
                result["is_fraud"],
            )
            return result
        except Exception as exc:
            log.exception("Prediction error: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/batch_predict", response_model=list[FraudDecision], tags=["Scoring"])
    def batch_predict(body: BatchRequest):
        """
        Score a batch of transactions (max 500 per call).

        Useful for nightly batch-scoring or replaying queued events.
        Send a JSON body: `{"transactions": [ {...}, {...} ]}`
        """
        transactions = body.transactions
        if len(transactions) == 0:
            raise HTTPException(status_code=400, detail="transactions list is empty")
        if len(transactions) > 500:
            raise HTTPException(
                status_code=400,
                detail=f"Max 500 transactions per batch, got {len(transactions)}"
            )
        results = []
        for txn in transactions:
            try:
                results.append(engine.predict(txn))
            except Exception as exc:
                log.exception("Batch prediction error: %s", exc)
                raise HTTPException(status_code=500, detail=str(exc))
        log.info("Batch scored %d transactions", len(results))
        return results

    @app.get("/model_info", tags=["System"])
    def model_info():
        """Returns model metadata, feature list, and performance metrics."""
        if not MODEL_SUMMARY_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail=(
                    "model_summary.json not found. "
                    "Run python fraud_detection_system.py first."
                ),
            )
        try:
            with open(MODEL_SUMMARY_PATH) as f:
                return json.load(f)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=500, detail=f"Bad model_summary.json: {exc}")

    @app.post("/reload_model", tags=["System"])
    def reload_model():
        """
        Hot-reload the model from disk without restarting the server.
        Use after re-training: python fraud_detection_system.py → POST /reload_model
        """
        nonlocal engine
        if not (_JOBLIB_AVAILABLE and MODEL_PATH.exists() and SCALER_PATH.exists()):
            raise HTTPException(
                status_code=404,
                detail="No saved model found on disk. Train first with fraud_detection_system.py",
            )
        try:
            new_model  = joblib.load(MODEL_PATH)
            new_scaler = joblib.load(SCALER_PATH)
            engine = FraudModel(new_model, new_scaler)
            log.info("Model hot-reloaded successfully ✓")
            return {"status": "ok", "message": "Model reloaded successfully", "version": FraudModel.VERSION}
        except Exception as exc:
            log.exception("Reload failed: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

    return app

    @app.get("/history", tags=["Data"])
    def get_history():
        cursor.execute("SELECT * FROM transactions ORDER BY timestamp DESC")
        rows = cursor.fetchall()

    return [
        {
            "id": row[0],
            "amount": row[1],
            "fraud_probability": row[2],
            "is_fraud": bool(row[3]),
            "risk_level": row[4],
            "timestamp": row[5],
        }
        for row in rows
    ]


# ─────────────────────────────────────────────────────────────────────────────
# TOP-LEVEL `app`  ← this is what uvicorn api_server:app looks for
# ─────────────────────────────────────────────────────────────────────────────

if not _FASTAPI_AVAILABLE:
    log.error("Cannot create app — FastAPI is not installed.")
    app = None  # type: ignore
else:
    log.info("=" * 60)
    log.info("  Fraud Detection API  v%s", FraudModel.VERSION)
    log.info("=" * 60)
    _model, _scaler = load_or_train_model()
    app = create_app(_model, _scaler)
    log.info("API ready → http://127.0.0.1:8000/docs")
    log.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT  (python api_server.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if app is None:
        print("Install dependencies: pip install fastapi uvicorn joblib scikit-learn numpy pandas")
        sys.exit(1)
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,   # set True during development
        log_level="info",
    )