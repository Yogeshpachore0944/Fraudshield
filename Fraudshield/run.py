"""
=============================================================================
  FRAUD DETECTION — ONE-CLICK LAUNCHER
=============================================================================
  Just run:  python run.py

  This script will automatically:
    1. Install all required packages
    2. Train the fraud detection model (skipped if already trained)
    3. Save the model to disk
    4. Start the API server at http://localhost:8000
    5. Open the browser to the Swagger UI docs

  After first run, the model is saved — restarting is instant.
=============================================================================
"""

import subprocess
import sys
import os
import time
from pathlib import Path

BASE_DIR    = Path(__file__).parent
MODEL_PATH  = BASE_DIR / "fraud_model.joblib"
SCALER_PATH = BASE_DIR / "fraud_scaler.joblib"

REQUIRED_PACKAGES = [
    "fastapi",
    "uvicorn[standard]",
    "pydantic",
    "joblib",
    "scikit-learn",
    "numpy",
    "pandas",
    "imbalanced-learn",
    "matplotlib",
    "seaborn",
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def banner(text: str, char: str = "="):
    line = char * 60
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}")


def step(n: int, text: str):
    print(f"\n[Step {n}] {text}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Install dependencies
# ─────────────────────────────────────────────────────────────────────────────

def install_packages():
    step(1, "Installing / verifying required packages …")
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade"] + REQUIRED_PACKAGES
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("  ⚠  pip output:\n", result.stderr[-1000:])
        print("  Some packages may have failed to install — continuing anyway.")
    else:
        print("  ✓  All packages ready.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Train & save model
# ─────────────────────────────────────────────────────────────────────────────

def train_and_save():
    step(2, "Checking for saved model …")

    if MODEL_PATH.exists() and SCALER_PATH.exists():
        print("  ✓  Saved model found — skipping training.")
        print(f"     Model  : {MODEL_PATH}")
        print(f"     Scaler : {SCALER_PATH}")
        print("     (Delete these files to force a retrain.)")
        return

    print("  No saved model found. Starting training pipeline …")
    print("  ⏳  This takes ~60-120 seconds on first run.\n")

    # Add base dir to path so imports work
    sys.path.insert(0, str(BASE_DIR))

    try:
        from fraud_detection_system import main as run_pipeline
    except ImportError:
        print("\n  ✗  ERROR: fraud_detection_system.py not found!")
        print(f"     Make sure it is in the same folder as run.py:")
        print(f"     {BASE_DIR}")
        sys.exit(1)

    try:
        import joblib
    except ImportError:
        print("\n  ✗  ERROR: joblib not found even after install attempt.")
        print("     Try manually:  pip install joblib")
        sys.exit(1)

    print("  Training started …\n")
    t0 = time.time()

    results, tuned_model, scaler, feature_names, best_t = run_pipeline()

    elapsed = time.time() - t0
    print(f"\n  ✓  Training complete in {elapsed:.1f}s")

    # Pick best model by average precision
    best_name  = max(results, key=lambda n: results[n]["avg_prec"])
    best_model = results[best_name]["model"]
    print(f"  ✓  Best model: {best_name}  "
          f"(avg_precision={results[best_name]['avg_prec']:.4f})")

    # Save
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler,     SCALER_PATH)
    print(f"  ✓  Model saved  → {MODEL_PATH}")
    print(f"  ✓  Scaler saved → {SCALER_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Start API server
# ─────────────────────────────────────────────────────────────────────────────

def start_server():
    step(3, "Starting API server …")

    api_file = BASE_DIR / "api_server.py"
    if not api_file.exists():
        print(f"\n  ✗  ERROR: api_server.py not found in {BASE_DIR}")
        sys.exit(1)

    print("  ✓  Launching uvicorn …")
    print()
    banner("Server is starting — open your browser at:", char="-")
    print()
    print("      http://localhost:8000/docs    ← Swagger UI (test API here)")
    print("      http://localhost:8000/health  ← Health check")
    print("      http://localhost:8000/predict ← POST endpoint")
    print()
    print("  Press CTRL+C to stop the server.")
    banner("", char="-")
    print()

    # Small delay so the banner is readable before uvicorn logs flood the screen
    time.sleep(1)

    # Open browser automatically
    try:
        import webbrowser
        webbrowser.open("http://localhost:8000/docs")
    except Exception:
        pass  # Not critical if browser won't open

    # Launch uvicorn — this blocks until CTRL+C
    os.chdir(BASE_DIR)  # so api_server.py can find model files by relative path
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "api_server:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
    ])


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    banner("FRAUD DETECTION API — ONE-CLICK LAUNCHER")
    print()
    print("  Files expected in this folder:")
    print(f"    ✔  run.py                    ← this file")
    print(f"    ✔  fraud_detection_system.py ← training pipeline")
    print(f"    ✔  api_server.py             ← FastAPI server")
    print(f"    ✔  model_summary.json        ← model metadata")

    install_packages()
    train_and_save()
    start_server()
