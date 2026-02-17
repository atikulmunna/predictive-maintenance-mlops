# Current MLOps Architecture

## System Architecture (Implemented)

```mermaid
graph TB
    subgraph Data["Data & Artifacts"]
        RAW["Raw Dataset<br/>NASA Turbofan"]
        PROC["Processed Features<br/>data/processed/*.csv"]
        MODELS["Model Artifacts<br/>data/models/*.json/*.pkl/*.h5"]
        MLRUNS["MLflow Local Runs<br/>mlruns/ (optional)"]
    end

    subgraph Train["Training & Orchestration"]
        TRAINER["trainer.py<br/>xgboost | lstm | ensemble | all"]
        FLOW["prefect_flow.py<br/>local/prefect engines"]
        DRIFT["drift_detection.py"]
        RETRAIN["retraining_pipeline.py"]
    end

    subgraph Serve["Serving & UI"]
        API["FastAPI<br/>/health /predict /explain"]
        UI["Streamlit Dashboard<br/>streamlit_app.py"]
    end

    RAW --> PROC
    PROC --> TRAINER
    TRAINER --> MODELS
    FLOW --> TRAINER
    MODELS --> API
    API --> UI
    PROC --> DRIFT
    DRIFT --> MODELS
    DRIFT --> RETRAIN
    RETRAIN --> FLOW
    TRAINER --> MLRUNS

    style API fill:#FFD54F
    style UI fill:#90CAF9
    style FLOW fill:#A5D6A7
```

## Training Pipeline (Implemented)

```mermaid
graph LR
    A["Raw Data<br/>NASA Turbofan"] 
    B["Feature Engineering<br/>notebooks/02 + data/processed"]
    C["XGBoost Training<br/>xgboost_pipeline.py"]
    D["LSTM Training<br/>lstm_pipeline.py"]
    E["Ensemble Selection<br/>ensemble_pipeline.py"]
    F["Selection Policy Artifact<br/>ensemble_metrics.json"]
    G["Orchestration Summary<br/>pipeline_run_summary.json"]

    A --> B --> C
    B --> D
    C --> E
    D --> E
    E --> F
    F --> G
```

## Model Comparison and Selection Policy (Current)

```mermaid
graph TB
    XGB["XGBoost Baseline<br/>Primary production fallback"]
    LSTM["LSTM Temporal<br/>Secondary model"]
    ENS["Validation-Tuned Ensemble<br/>Candidate model"]
    GATE["Selection Gate<br/>Use ensemble only if val F2 gain >= min_f2_gain"]
    SELECT["selected_model + selected_threshold<br/>stored in ensemble_metrics.json"]

    XGB --> GATE
    LSTM --> ENS
    XGB --> ENS
    ENS --> GATE
    GATE --> SELECT

    style XGB fill:#A5D6A7
    style ENS fill:#FFD54F
    style SELECT fill:#90CAF9
```

## Inference Data Flow (Current)

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit / Client
    participant API as FastAPI
    participant M as ModelBundle
    participant A as data/models artifacts

    User->>UI: Submit sequence
    UI->>API: POST /predict or /explain
    API->>M: Validate sequence schema
    M->>A: Load feature_names + scalers + models
    M->>M: XGBoost / LSTM inference
    M->>M: Apply selected_model policy
    API-->>UI: JSON response
```

## Deployment Architecture (Current)

```mermaid
graph TB
    subgraph Local["Local / Dev Runtime"]
        TRAIN["Python CLI & Notebooks"]
        API["FastAPI Service"]
        ST["Streamlit Dashboard"]
    end

    subgraph Compose["Docker Compose Runtime"]
        API_C["api container"]
        REDIS["redis container"]
        PG["postgres container"]
    end

    TRAIN --> API
    API --> ST
    API_C --> REDIS
    API_C --> PG
```
