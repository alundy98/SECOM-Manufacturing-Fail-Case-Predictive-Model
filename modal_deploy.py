# modal_deploy.py
from pathlib import Path
import modal
import os
import subprocess

app = modal.App("secom-failure-dashboard")

project_dir = Path(__file__).parent

# All your local model and visualization files
local_files = [
    "app.py",
    "xgb_model.pkl",
    "scaler.pkl",
    "feature_names.pkl",
    "feature_importance.csv",
    "feature_importance.png",
    "confusion_matrix.npy",
    "confusion_matrix.png",
    "y_test.npy",
    "y_pred.npy",
]

# Define base image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "streamlit",
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "matplotlib",
        "seaborn",
        "joblib",
    )
)

# Mount local project files to container
for f in local_files:
    image = image.add_local_file(project_dir / f, f"/root/{f}")

# Streamlit entry point
@app.function(image=image, timeout=600)
@modal.web_server(port=8501)
def serve():
    cmd = [
        "streamlit",
        "run",
        "/root/app.py",
        "--server.port=8501",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
    ]
    subprocess.run(cmd, check=True)
