# Predictive Maintenance MLOps Platform

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> A production-ready machine learning platform for industrial equipment failure prediction, featuring validation-tuned ensemble models (XGBoost + LSTM), real-time API serving, automated retraining, and comprehensive MLOps infrastructure.

![Project Status](https://img.shields.io/badge/status-in%20development-yellow)

---

## 🎯 Project Overview

This platform demonstrates end-to-end MLOps capabilities by predicting equipment failures 30 cycles in advance using NASA's Turbofan Engine Degradation dataset. The system combines classical machine learning with deep learning, deployed as a containerized microservices architecture.

### Key Features

- **Ensemble Models**: Validation-tuned XGBoost + LSTM blending with fallback policy
- **GPU Acceleration**: PyTorch LSTM training optimized for NVIDIA RTX 5060 Mobile (8GB VRAM)
- **REST API**: FastAPI service with <50ms latency (p95)
- **Explainability**: SHAP values for model interpretability
- **Drift Detection**: Automated feature and prediction drift monitoring
- **Auto-Retraining**: Triggered by drift or performance degradation
- **Monitoring**: Prometheus + Grafana dashboards with Slack alerting
- **MLOps Pipeline**: Prefect orchestration with MLflow tracking

### Architecture Highlights

See [**Detailed Architecture Diagrams**](assets/architecture.md) for complete reference. Below is the system architecture:

```mermaid
graph TB
    subgraph DL["Data Layer"]
        DB[(PostgreSQL<br/>Data Lake)]
        FS[(Redis<br/>Feature Store)]
        MLFLOW[(MLflow<br/>Artifacts)]
    end
    
    subgraph ML["Model Layer"]
        XGB["XGBoost<br/>validation-tuned weight<br/>F2: 0.9915"]
        LSTM["LSTM + Attention<br/>validation-tuned weight<br/>F2: 0.8750"]
    end
    
    subgraph IL["Inference Layer"]
        API["FastAPI<br/>REST API"]
        CACHE["Request Cache"]
    end
    
    subgraph TP["Training Pipeline"]
        PREPROCESS["Data Preprocessing"]
        TRAIN["Model Training"]
        EVAL["Evaluation"]
    end
    
    subgraph MA["Monitoring"]
        PROM["Prometheus"]
        GRAFANA["Grafana"]
    end
    
    DB -->|Load| PREPROCESS
    PREPROCESS -->|Split| TRAIN
    TRAIN -->|Save| MLFLOW
    MLFLOW -->|Load| XGB
    MLFLOW -->|Load| LSTM
    XGB -->|Ensemble| API
    LSTM -->|Ensemble| API
    API -->|Cache| CACHE
    API -->|Query| FS
    TRAIN -->|Metrics| EVAL
    EVAL -->|Log| PROM
    PROM -->|Visualize| GRAFANA

    style XGB fill:#A5D6A7
    style LSTM fill:#90CAF9
    style API fill:#FFD54F
```

**Current Benchmark (Test Set, updated 2026-02-07):**
| Metric | XGBoost Baseline | LSTM Temporal | Tuned Ensemble (0.775/0.225) |
|--------|-------------------|---------------|-------------------------------|
| F2 Score | 0.9923 | 0.8637 | **0.9932** |
| Precision | 0.9707 | 0.8520 | **0.9748** |
| Recall | **0.9978** | 0.8667 | **0.9978** |
| ROC-AUC | **0.9999** | 0.9869 | 0.9993 |

**Known Decision**
- Primary metric is F2.
- Production selection uses validation-gated fallback (from `data/models/ensemble_metrics.json`):
  - Use ensemble only if validation F2 gain is at least `min_f2_gain_for_ensemble` (currently `0.005`) over XGBoost reference.
  - Otherwise select XGBoost.
- Serving policy precedence:
  - Default source: `selected_model` and `selected_threshold` from `data/models/ensemble_metrics.json`.
  - Emergency override: `MODEL_OVERRIDE=xgboost|ensemble` (and optional `MODEL_THRESHOLD_OVERRIDE=0.0..1.0`).
- Current selected production model: **XGBoost** (`selected_model: "xgboost"`).

### Training Pipeline

```mermaid
graph LR
    A["Raw Data<br/>100 engines"] 
    B["Feature<br/>Engineering<br/>128 features"]
    C["Feature<br/>Selection<br/>Top 40"]
    D["Sequence<br/>Creation<br/>30 cycles"]
    E["Train/Val/Test<br/>Split 70/15/15"]
    F["Model Training<br/>GPU-accelerated"]
    G["Evaluation<br/>& Metrics"]
    H["Model Registry<br/>MLflow"]
    
    A-->B-->C-->D-->E-->F-->G-->H
    
    style F fill:#F3E5F5
    style H fill:#FFFDE7
```

---

## 📋 Quick Start

### Prerequisites

- **Hardware**: NVIDIA GPU with 8GB+ VRAM (RTX 5060 Mobile or better)
- **Software**: 
  - Windows 10/11 with WSL2 (or Linux/macOS)
  - [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
  - [NVIDIA CUDA 13.0](https://developer.nvidia.com/cuda-toolkit)
  - [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for deployment)

### Installation (10 minutes)

1. **Clone the repository**
   ```powershell
   git clone https://github.com/yourusername/predictive-maintenance-mlops.git
   cd predictive-maintenance-mlops
   ```

2. **Run automated setup script**
   ```powershell
   .\setup_env.ps1
   ```

   This script will:
   - Create conda environment `pred-maint`
   - Install Python 3.11
   - Install PyTorch with CUDA 13.0
   - Install all project dependencies
   - Create `.env` configuration file
   - Verify GPU setup

3. **Activate the environment**
   ```powershell
   conda activate pred-maint
   ```

4. **Verify GPU is working**
   ```powershell
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   # Expected output: CUDA available: True
   ```

### Manual Installation (if script fails)

<details>
<summary>Click to expand manual setup instructions</summary>

```powershell
# Create conda environment
conda create -n pred-maint python=3.11 -y
conda activate pred-maint

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=13.0 -c pytorch -c nvidia -y

# Install project dependencies
pip install -r requirements/base.txt
pip install -r requirements/training.txt
pip install -r requirements/serving.txt
pip install -r requirements/dev.txt

# Setup environment file
cp .env.example .env
# Edit .env with your configuration
```

</details>

---

## 🚀 Usage

### 1. Exploratory Data Analysis

```powershell
# Launch Jupyter Lab
jupyter lab

# Open notebooks/01_eda.ipynb (create if doesn't exist)
```

**EDA Goals:**
- Load NASA Turbofan dataset from `turbofan_ed_dataset/`
- Analyze sensor patterns and failure modes
- Visualize degradation trends
- Identify most predictive sensors

### 2. Train Models Locally

```powershell
# Train XGBoost baseline
python -m src.training.trainer --model xgboost --data-dir data

# Train LSTM temporal model
python -m src.training.trainer --model lstm --data-dir data --epochs 100 --batch-size 32

# Run ensemble selection (validation-tuned blending + fallback policy)
python -m src.training.trainer --model ensemble --data-dir data --min-f2-gain 0.005

# Run full pipeline sequentially: XGBoost -> LSTM -> Ensemble
python -m src.training.trainer --model all --data-dir data

# Phase 3 orchestration entrypoint (local runner; Prefect mode when installed)
python -m src.pipelines.prefect_flow --engine local --data-dir data --epochs 100 --batch-size 32
```

### 3. Start API Server

```powershell
# Development mode (auto-reload)
uvicorn src.serving.api:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn src.serving.api:app --workers 3 --host 0.0.0.0 --port 8000
```

Access API documentation: http://localhost:8000/docs

### 3b. Start With Docker Compose

```powershell
docker compose up --build
```

This starts:
- `api` on `http://localhost:8000`
- `redis` on `localhost:6379`
- `postgres` on `localhost:5432`

### 4. Make Predictions

```powershell
# Using curl
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "equipment_id": "engine_001",
    "sequence": [
      {"op_setting_1": 0.0, "op_setting_2": 0.0, "...": 0.0},
      {"op_setting_1": 0.1, "op_setting_2": 0.0, "...": 0.0}
    ]
  }'

# Using Python
import requests
response = requests.post("http://localhost:8000/predict", json={...})
print(response.json())
```

`sequence` must contain 30 timesteps (or the configured sequence length), and each timestep must include all features listed in `data/models/feature_names.json`.

### 4b. Explain Predictions

```powershell
curl -X POST "http://localhost:8000/explain?top_k=10" ^
  -H "Content-Type: application/json" ^
  -d "{ \"equipment_id\": \"engine_001\", \"sequence\": [...] }"
```

`/explain` returns top feature contribution scores from the XGBoost model (`pred_contribs`), which are SHAP-compatible local attributions.

### 5. Run Tests

```powershell
# All tests
pytest

# Unit tests only
pytest -m unit

# With coverage report
pytest --cov=src --cov-report=html
# Open htmlcov/index.html to view coverage
```

Current baseline: `pytest -q` passes with total coverage above 85%.

### 6. Run Drift Detection

```powershell
python -m src.monitoring.drift_detection ^
  --reference data/processed/train_features_FD001.csv ^
  --current data/processed/train_features_FD001.csv
```

Writes `data/models/drift_report.json` with feature and prediction drift summary.

---

## 📊 Dataset

**Source**: [NASA Turbofan Engine Degradation Simulation](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)

**Location**: `turbofan_ed_dataset/`

**Details**:
- **Training samples**: ~70,000 cycles (100 engines)
- **Sensors**: 21 time-series measurements
- **Operating conditions**: 3 settings
- **Failure modes**: Multiple degradation patterns

**Files**:
- `train_FD001.txt` - Training data (simplest failure mode)
- `test_FD001.txt` - Test data
- `RUL_FD001.txt` - Ground truth Remaining Useful Life

---

## 🏗️ Project Structure

```
predictive-maintenance-mlops/
├── data/                      # Data storage (gitignored)
│   ├── raw/                   # Original dataset
│   ├── processed/             # Cleaned data
│   ├── features/              # Engineered features
│   └── models/                # Trained models
│
├── notebooks/                 # Jupyter notebooks
│   ├── 01_eda.ipynb          # Exploratory analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_xgboost.ipynb
│   ├── 04_lstm_model.ipynb
│   └── 05_ensemble.ipynb
│
├── src/                       # Source code
│   ├── data/                  # Data processing
│   ├── features/              # Feature engineering
│   ├── models/                # ML models
│   ├── training/              # Training pipeline
│   ├── serving/               # API service
│   ├── monitoring/            # Drift detection
│   ├── pipelines/             # Orchestration
│   └── utils/                 # Utilities
│
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── load/                  # Load tests
│
├── configs/                   # Configuration files
├── docker/                    # Docker setup
├── deploy/                    # Deployment configs
├── docs/                      # Documentation
├── scripts/                   # Utility scripts
│
├── requirements/              # Dependencies
│   ├── base.txt              # Core packages
│   ├── training.txt          # ML training
│   ├── serving.txt           # API serving
│   └── dev.txt               # Development tools
│
├── .env.example              # Environment variables template
├── setup_env.ps1             # Automated setup script
├── pyproject.toml            # Project metadata
├── pytest.ini                # Test configuration
└── README.md                 # This file
```

---


## 🛠️ Tech Stack

- **Language**: Python 3.11
- **ML Frameworks**: TensorFlow/Keras 2.15+, XGBoost 2.0+, Scikit-learn, imbalanced-learn
- **API**: FastAPI, Uvicorn
- **Model Artifacts**: JSON, H5, Pickle (`data/models/*`)
- **Experiment Tracking**: MLflow (notebook workflow + local `mlruns`)
- **Orchestration**: Prefect-compatible local flow runner (`src/pipelines/prefect_flow.py`)
- **Drift Detection**: Statistical drift checks (KS + PSI + prediction mean shift) via `src/monitoring/drift_detection.py`
- **Monitoring (Planned)**: Prometheus, Grafana
- **Containerization (Planned)**: Docker, Docker Compose
- **Testing**: pytest
- **Explainability**: SHAP
- **GPU**: CUDA 13.0
