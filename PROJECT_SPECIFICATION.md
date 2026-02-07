# Predictive Maintenance MLOps Platform
## Project Specification (Personal Portfolio Edition)

**Last Updated:** February 7, 2026

---

## 🎯 Project Vision

Build a **production-ready** predictive maintenance platform that demonstrates end-to-end MLOps capabilities, leveraging deep learning with GPU acceleration. This project serves as a portfolio piece showcasing real-world ML engineering skills.

**Key Differentiators:**
- GPU-accelerated LSTM with PyTorch (RTX 5060 Mobile, 8GB VRAM)
- Real MLOps pipeline (not just a Jupyter notebook)
- Explainable predictions with SHAP
- Automated retraining with drift detection
- Docker-based deployment ready for cloud

---

## 💻 Hardware & Environment

**Development Machine:**
- GPU: NVIDIA RTX 5060 Mobile (8GB VRAM)
- CUDA: 13.1
- Driver: 591.74
- Available VRAM for Training: ~7GB (after OS overhead)

**Performance Targets:**
- LSTM training: 5-10 minutes per epoch
- Batch inference: 100 predictions < 2 seconds
- API latency: < 50ms (p95)

---

## 📋 Project Phases (12-Week Timeline)

### **Phase 1: Foundation (Weeks 1-3)**
**Goal:** Working baseline with data pipeline and simple model

**Deliverables:**
- ✅ EDA notebook with dataset understanding
- ✅ Feature engineering pipeline (30-50 features)
- ✅ XGBoost baseline model (target F2 > 0.75)
- ✅ MLflow experiment tracking
- ✅ Basic project structure

**Tech Stack:**
- Python 3.11
- Pandas, NumPy, Scikit-learn
- XGBoost
- MLflow
- Jupyter Lab

### **Phase 2: Deep Learning & API (Weeks 4-6)**
**Goal:** Production API with ensemble model

**Deliverables:**
- ✅ PyTorch LSTM model (GPU-accelerated)
- ✅ Ensemble: XGBoost (0.6) + LSTM (0.4)
- ✅ FastAPI prediction service
- ✅ SHAP explainability
- ✅ Docker Compose setup
- ✅ Test coverage > 80%

**Tech Stack:**
- PyTorch 2.x + CUDA 13.1
- FastAPI + Uvicorn
- Redis (feature store)
- PostgreSQL (data lake)
- Docker + Docker Compose

### **Phase 3: MLOps Pipeline (Weeks 7-9)**
**Goal:** Automated training and deployment

**Deliverables:**
- ✅ Automated training pipeline
- ✅ Model registry (MLflow)
- ✅ Drift detection (Evidently AI)
- ✅ Automated retraining trigger
- ✅ Model versioning and rollback
- ✅ CI/CD with GitHub Actions

**Tech Stack:**
- Prefect (lightweight orchestration)
- Evidently AI (drift detection)
- Great Expectations (data validation)
- GitHub Actions (CI/CD)

### **Phase 4: Monitoring & Polish (Weeks 10-12)**
**Goal:** Production-grade monitoring and documentation

**Deliverables:**
- ✅ Prometheus + Grafana dashboards
- ✅ Alerting (Slack webhook)
- ✅ Load testing (Locust)
- ✅ Complete documentation
- ✅ Demo video (5-8 minutes)
- ✅ Technical blog post

**Tech Stack:**
- Prometheus (metrics)
- Grafana (visualization)
- Locust (load testing)
- MkDocs (documentation)

---

## 🤖 Machine Learning Design

### **Problem Statement**

**Binary Classification:** Predict equipment failure in next 30 cycles
**Regression:** Estimate Remaining Useful Life (RUL)

**Dataset:** NASA Turbofan Engine Degradation
- Training samples: ~70,000 cycles
- Engines: 100 units
- Sensors: 21 time-series measurements
- Operating conditions: 3 settings

### **Target Metrics**

| Metric | Target | Rationale |
|--------|--------|-----------|
| **F2 Score** | > 0.80 | Prioritize recall (2x weight) - missing failures costly |
| **Precision** | > 0.65 | Acceptable false alarm rate (~35%) |
| **Recall** | > 0.85 | Catch 85%+ of actual failures |
| **AUC-ROC** | > 0.90 | Overall discrimination ability |
| **RMSE (RUL)** | < 20 cycles | RUL prediction accuracy |

**Business Context:**
- False Negative Cost: $50,000 (unplanned downtime)
- False Positive Cost: $2,000 (unnecessary inspection)
- Target: 10:1 cost ratio justifies recall focus

### **Model Architecture**

#### **Model 1: XGBoost (Weight: 0.60)**

```python
XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=5,  # class imbalance
    gpu_id=0,  # GPU acceleration
    tree_method='gpu_hist'
)
```

**Strengths:**
- Handles tabular features excellently
- Fast inference (<2ms per prediction)
- Interpretable feature importance
- Robust to feature scaling

**Training Time:** ~5 minutes
**Expected F2:** 0.76-0.78

#### **Model 2: LSTM (Weight: 0.40)**

```python
class LSTMPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=21, hidden_size=128, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
```

**Input:** Last 50 cycles × 21 sensors
**Strengths:**
- Captures temporal dependencies
- Learns degradation patterns
- Benefits from GPU (8GB VRAM sufficient)

**Training Time:** ~8 minutes (10 epochs on GPU)
**Expected F2:** 0.74-0.76

#### **Ensemble Strategy**

**Weighted Average:**
```python
prediction = 0.6 * xgb_proba + 0.4 * lstm_proba
```

**Calibration:** Platt scaling on validation set
**Expected Ensemble F2:** 0.82-0.85

**Why this split?**
- XGBoost excels at cross-sectional patterns
- LSTM captures temporal degradation
- XGBoost is faster, gets higher weight

---

## 🔧 Feature Engineering

### **Feature Groups (Total: ~50 features)**

#### **1. Rolling Statistics (21 sensors × 3 windows = 63 → top 20)**
```python
windows = [10, 25, 50]  # cycles
aggregations = ['mean', 'std', 'max']

# Example for sensor_2 (temperature)
features = [
    'sensor_2_mean_10',   # short-term trend
    'sensor_2_std_25',    # medium-term volatility
    'sensor_2_max_50'     # long-term peak
]
```

**Selection:** Keep top 20 by XGBoost importance

#### **2. Lag Features (sensors = 10)**
```python
# Most degradation-sensitive sensors
critical_sensors = [2, 3, 4, 7, 11, 12, 13, 15, 17, 21]

for sensor in critical_sensors:
    features.append(f'sensor_{sensor}_lag_1')   # previous cycle
    features.append(f'sensor_{sensor}_lag_10')  # 10 cycles ago
```

**Output:** 10 sensors × 1 lag = 10 features

#### **3. Rate of Change (10 features)**
```python
for sensor in critical_sensors:
    # Degradation velocity
    features.append(f'sensor_{sensor}_roc_10')  # change over 10 cycles
```

#### **4. Domain Features (10 features)**
```python
# Temperature stress indicator
'temp_stress': (sensor_2 + sensor_3 + sensor_4) / 3

# Vibration anomaly
'vibration_anomaly': sensor_11 > sensor_11_mean_50 + 2*std

# Cycles since anomaly
'cycles_since_anomaly': count cycles since vibration spike

# Operating regime (one-hot encoded)
'regime_low', 'regime_medium', 'regime_high'

# Degradation score (composite)
'degradation_score': weighted sum of sensor deviations
```

**Total Features:** 20 + 10 + 10 + 10 = **50 features**

### **Feature Store Design**

**Storage:** Redis (in-memory for <5ms access)

```python
# Key structure
feature_key = f"features:equipment_{id}:timestamp_{ts}"

# Value: JSON with features + metadata
{
    "features": [...],  # 50-element array
    "computed_at": "2026-02-07T10:00:00Z",
    "version": "v1.2"
}
```

**TTL:** 90 days (training needs historical features)
**Backup:** PostgreSQL (persistent storage)

---

## 🏗️ System Architecture

### **Service Stack**

```
┌─────────────────────────────────────────────────────────────┐
│                     CLIENT APPLICATIONS                      │
│         (Maintenance Dashboard, API Consumers)               │
└────────────────────────┬────────────────────────────────────┘
                         │
                    [HTTPS/REST]
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   PREDICTION API                             │
│         FastAPI + Uvicorn (3 replicas)                       │
│    /predict  /batch-predict  /explain  /health              │
└───┬─────────────────────────────────────────────┬───────────┘
    │                                             │
    │ (features)                            (log predictions)
    ▼                                             ▼
┌───────────────┐                          ┌──────────────┐
│     REDIS     │                          │ PostgreSQL   │
│ Feature Store │                          │  Predictions │
│  (< 5ms read) │                          │   + Actuals  │
└───────────────┘                          └──────────────┘
         ▲                                        │
         │                                        │
         │ (write features)              (read training data)
         │                                        │
┌────────┴──────────────────────────────────────┴───────────┐
│                  TRAINING PIPELINE                         │
│    (Prefect DAG, runs weekly + on-demand)                  │
│                                                             │
│  [Data Prep] → [Feature Eng] → [Train] → [Evaluate]       │
│        ↓            ↓             ↓          ↓             │
│    [Validate]   [Store]      [MLflow]   [Register]        │
└───────────────────────┬────────────────────────────────────┘
                        │
                    [artifacts]
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   MLFLOW SERVER                              │
│     Experiment Tracking + Model Registry                     │
│     Backend: PostgreSQL  |  Artifacts: Local filesystem      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              MONITORING STACK                                │
│   Prometheus → Grafana → Alertmanager → Slack               │
│  (metrics)    (dashboards)  (alerts)    (notifications)     │
└─────────────────────────────────────────────────────────────┘
```

### **Docker Compose Services**

```yaml
services:
  # Core Data Layer
  postgres:
    image: postgres:15-alpine
    volumes: pgdata
    ports: 5432
    resources: 2 CPU, 4GB RAM
  
  redis:
    image: redis:7-alpine
    ports: 6379
    resources: 1 CPU, 2GB RAM
  
  # MLOps Layer
  mlflow:
    build: ./mlflow
    ports: 5000
    depends_on: postgres
    resources: 1 CPU, 2GB RAM
  
  # Application Layer
  api:
    build: ./src/api
    ports: 8000
    replicas: 3
    depends_on: [postgres, redis, mlflow]
    resources: 2 CPU, 4GB RAM per replica
    environment:
      - CUDA_VISIBLE_DEVICES=-1  # CPU inference (fast enough)
  
  # Monitoring Layer
  prometheus:
    image: prom/prometheus:latest
    ports: 9090
    volumes: ./prometheus.yml
    resources: 1 CPU, 2GB RAM
  
  grafana:
    image: grafana/grafana:latest
    ports: 3000
    depends_on: prometheus
    resources: 1 CPU, 1GB RAM
  
  # Training (on-demand, not always running)
  trainer:
    build: ./src/training
    depends_on: [postgres, redis, mlflow]
    deploy: manual
    environment:
      - CUDA_VISIBLE_DEVICES=0  # Use GPU
    resources: 4 CPU, 8GB RAM, 1 GPU
```

**Total Resource Usage:**
- Idle: ~12GB RAM, minimal CPU
- Training: +8GB RAM, GPU active

---

## 🔄 MLOps Pipeline

### **Pipeline Orchestration: Prefect**

**Why Prefect over Airflow?**
- Lighter weight (no separate webserver)
- Better for personal projects
- Native Python API
- Easier debugging

### **Pipeline DAGs**

#### **1. Data Ingestion Pipeline**
**Trigger:** Manual (hourly in production)
**Duration:** ~2 minutes

```python
@flow
def data_ingestion_pipeline():
    # 1. Extract from source (simulated IoT)
    raw_data = extract_sensor_data()
    
    # 2. Validate with Great Expectations
    validation_results = validate_data(raw_data)
    
    if not validation_results.success:
        send_alert("Data validation failed")
        return
    
    # 3. Store in PostgreSQL
    store_raw_data(raw_data)
    
    # 4. Trigger feature pipeline
    feature_engineering_pipeline()
```

#### **2. Feature Engineering Pipeline**
**Trigger:** After data ingestion
**Duration:** ~3 minutes for 100 equipment

```python
@flow
def feature_engineering_pipeline():
    # 1. Load recent data (last 100 cycles per equipment)
    data = load_data_for_features()
    
    # 2. Compute features
    features = compute_features(data)
    
    # 3. Validate feature distributions
    validate_features(features)
    
    # 4. Store in Redis + PostgreSQL
    store_features(features)
```

#### **3. Training Pipeline**
**Trigger:** Weekly (Sunday 2 AM) OR on-demand OR drift detected
**Duration:** ~15 minutes

```python
@flow
def training_pipeline(trigger_reason: str):
    with mlflow.start_run():
        # 1. Load training data (last 60 days)
        X_train, y_train = load_training_data()
        X_val, y_val = load_validation_data()
        
        # 2. Train XGBoost
        xgb_model = train_xgboost(X_train, y_train)
        xgb_metrics = evaluate_model(xgb_model, X_val, y_val)
        
        # 3. Train LSTM (GPU)
        lstm_model = train_lstm(X_train, y_train, device='cuda')
        lstm_metrics = evaluate_model(lstm_model, X_val, y_val)
        
        # 4. Create ensemble
        ensemble = create_ensemble(xgb_model, lstm_model)
        ensemble_metrics = evaluate_model(ensemble, X_val, y_val)
        
        # 5. Log to MLflow
        mlflow.log_params(...)
        mlflow.log_metrics(ensemble_metrics)
        
        # 6. Model validation gate
        if ensemble_metrics['f2_score'] > 0.78:
            # 7. Register model
            model_uri = mlflow.register_model(
                model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
                name="predictive_maintenance_ensemble"
            )
            
            # 8. Transition to staging
            transition_model_stage(model_uri, stage="Staging")
            
            # 9. Run shadow deployment tests
            shadow_test_results = run_shadow_tests(ensemble)
            
            # 10. Promote to production if tests pass
            if shadow_test_results.pass_rate > 0.95:
                transition_model_stage(model_uri, stage="Production")
                send_alert(f"New model deployed: {model_uri}")
            else:
                send_alert(f"Shadow tests failed, keeping old model")
        else:
            send_alert(f"Model F2 {ensemble_metrics['f2_score']:.3f} < 0.78, not deploying")
```

#### **4. Drift Detection Pipeline**
**Trigger:** Every 6 hours
**Duration:** ~1 minute

```python
@flow
def drift_detection_pipeline():
    # 1. Load reference data (last 30 days)
    reference_data = load_reference_data()
    
    # 2. Load current data (last 6 hours)
    current_data = load_current_data()
    
    # 3. Feature drift (Evidently)
    feature_drift_report = generate_feature_drift_report(
        reference_data, current_data
    )
    
    # 4. Prediction drift (PSI)
    prediction_drift = calculate_psi(
        reference_predictions, current_predictions
    )
    
    # 5. Check thresholds
    if feature_drift_report.get_drift_share() > 0.3:
        send_alert("⚠️ Feature drift detected: 30% of features drifted")
        # Trigger retraining
        training_pipeline.apply_async(trigger_reason="feature_drift")
    
    if prediction_drift > 0.25:
        send_alert("⚠️ Prediction drift detected: PSI = {prediction_drift:.3f}")
        training_pipeline.apply_async(trigger_reason="prediction_drift")
    
    # 6. Store drift metrics
    store_drift_metrics(feature_drift_report, prediction_drift)
```

---

## 🚀 API Design

### **Endpoints**

#### **1. Health Check**
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "v1.3.2",
  "services": {
    "postgres": "ok",
    "redis": "ok",
    "mlflow": "ok"
  },
  "uptime_seconds": 86400
}
```

#### **2. Single Prediction**
```http
POST /predict
Content-Type: application/json

{
  "equipment_id": "engine_042",
  "sensor_readings": {
    "sensor_1": 518.67,
    "sensor_2": 642.83,
    ...
    "sensor_21": 23.42
  },
  "operational_settings": {
    "setting_1": -0.0007,
    "setting_2": -0.0004,
    "setting_3": 100.0
  },
  "timestamp": "2026-02-07T10:00:00Z"
}
```

**Response (< 50ms):**
```json
{
  "equipment_id": "engine_042",
  "prediction": {
    "failure_probability": 0.82,
    "risk_level": "HIGH",
    "confidence": 0.91,
    "predicted_rul_cycles": 15,
    "predicted_rul_days": 3.75,
    "should_inspect": true
  },
  "explanation": {
    "top_features": [
      {
        "feature": "sensor_11_mean_25",
        "importance": 0.23,
        "value": 47.82,
        "impact": "increases_risk"
      },
      {
        "feature": "sensor_2_roc_10",
        "importance": 0.19,
        "value": 15.3,
        "impact": "increases_risk"
      },
      {
        "feature": "degradation_score",
        "importance": 0.15,
        "value": 0.78,
        "impact": "increases_risk"
      }
    ],
    "visualization_url": "/shap/engine_042/latest"
  },
  "model_version": "v1.3.2",
  "timestamp": "2026-02-07T10:00:01Z",
  "latency_ms": 42
}
```

#### **3. Batch Prediction**
```http
POST /batch-predict
Content-Type: application/json

{
  "equipment_ids": ["engine_001", "engine_002", ..., "engine_100"],
  "timestamp": "2026-02-07T10:00:00Z"
}
```

**Response (< 2 seconds for 100 equipment):**
```json
{
  "predictions": [
    { "equipment_id": "engine_001", "failure_probability": 0.15, ... },
    { "equipment_id": "engine_002", "failure_probability": 0.67, ... },
    ...
  ],
  "total_count": 100,
  "high_risk_count": 8,
  "processing_time_ms": 1847
}
```

#### **4. Feedback Submission**
```http
POST /feedback
Content-Type: application/json

{
  "equipment_id": "engine_042",
  "prediction_id": "pred_20260207_100001_042",
  "actual_outcome": {
    "failure_occurred": true,
    "failure_timestamp": "2026-02-10T08:30:00Z",
    "failure_type": "compressor_blade_break",
    "inspection_notes": "Visible crack on blade 7"
  }
}
```

**Response:**
```json
{
  "feedback_id": "fb_20260210_100530_042",
  "status": "recorded",
  "prediction_accuracy": {
    "predicted_rul_days": 3.75,
    "actual_rul_days": 3.15,
    "error_days": 0.60
  },
  "model_updated": false,
  "next_training_scheduled": "2026-02-14T02:00:00Z"
}
```

---

## 📊 Monitoring & Observability

### **Metrics to Track**

#### **Model Performance**
```python
# Prometheus metrics
model_prediction_total = Counter('predictions_total', 'Total predictions')
model_prediction_latency = Histogram('prediction_latency_seconds')
model_confidence = Histogram('prediction_confidence')
model_f2_score = Gauge('model_f2_score_current')
model_precision = Gauge('model_precision_current')
model_recall = Gauge('model_recall_current')
model_false_alarm_rate = Gauge('model_false_alarm_rate')
```

#### **Data Quality**
```python
data_missing_rate = Gauge('data_missing_value_rate')
data_outlier_count = Counter('data_outliers_total')
feature_drift_score = Gauge('feature_drift_score', ['feature_name'])
prediction_drift_psi = Gauge('prediction_drift_psi')
```

#### **System Health**
```python
api_requests_total = Counter('api_requests_total', ['endpoint', 'status'])
api_latency = Histogram('api_latency_seconds', ['endpoint'])
api_errors_total = Counter('api_errors_total', ['endpoint', 'error_type'])
db_connection_pool_size = Gauge('db_connections_active')
redis_cache_hit_rate = Gauge('redis_cache_hit_rate')
```

### **Grafana Dashboards**

#### **Dashboard 1: Model Performance**
- F2 / Precision / Recall over time (line chart)
- Confusion matrix (heatmap, updated daily)
- Prediction distribution (histogram)
- Confidence intervals (box plot)
- False alarm rate trend

#### **Dashboard 2: API Health**
- Request rate (requests/second)
- Latency percentiles (p50, p95, p99)
- Error rate (%)
- Success rate by endpoint
- Response time heatmap

#### **Dashboard 3: Data & Drift**
- Feature drift scores (bar chart, top 10)
- Prediction drift PSI (gauge)
- Missing value rate (line chart)
- Data ingestion lag (seconds)
- Training data freshness

#### **Dashboard 4: Business KPIs**
- Prevented failures (count)
- Cost savings estimate ($)
- Equipment under monitoring
- High-risk equipment list (table)
- Monthly maintenance schedule

### **Alerting Rules**

**Critical Alerts (Slack + Email):**
```yaml
- Model F2 score < 0.75 for 24 hours
- API down (no successful requests for 5 minutes)
- Prediction drift PSI > 0.40
- Training pipeline failed 2x in a row
- Database connection pool exhausted
```

**Warning Alerts (Slack only):**
```yaml
- Model F2 score < 0.78
- API p95 latency > 100ms for 10 minutes
- Prediction drift PSI > 0.25
- Feature drift detected in >30% of features
- Cache hit rate < 70%
- Data ingestion delayed > 1 hour
```

---

## 🧪 Testing Strategy

### **Test Coverage Target: 85%**

#### **Unit Tests (pytest)**
```
tests/
├── test_features.py          # Feature engineering functions
├── test_models.py            # Model prediction logic
├── test_data_validation.py   # Great Expectations suites
├── test_utils.py             # Helper functions
└── test_drift_detection.py   # Drift calculation
```

**Example:**
```python
def test_rolling_mean_feature():
    """Test rolling mean calculation for sensor data."""
    data = pd.DataFrame({
        'sensor_2': [10, 20, 30, 40, 50],
        'cycle': [1, 2, 3, 4, 5]
    })
    
    result = compute_rolling_mean(data, window=3, sensor='sensor_2')
    
    assert result.iloc[2] == 20.0  # (10+20+30)/3
    assert result.iloc[4] == 40.0  # (30+40+50)/3

def test_xgboost_prediction_shape():
    """Test XGBoost model output shape."""
    model = load_model('xgboost_v1.pkl')
    X_test = np.random.rand(10, 50)  # 10 samples, 50 features
    
    predictions = model.predict_proba(X_test)
    
    assert predictions.shape == (10, 2)  # 2 classes
    assert np.all((predictions >= 0) & (predictions <= 1))
```

#### **Integration Tests**
```
tests/integration/
├── test_api_endpoints.py     # End-to-end API tests
├── test_training_pipeline.py # Full training workflow
├── test_database.py          # Database operations
└── test_feature_store.py     # Redis operations
```

**Example:**
```python
@pytest.mark.integration
def test_predict_endpoint_e2e(test_client):
    """Test full prediction flow through API."""
    payload = {
        "equipment_id": "test_engine",
        "sensor_readings": {...},  # valid sensor data
        "operational_settings": {...}
    }
    
    response = test_client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data['prediction']['failure_probability'] <= 1
    assert 'explanation' in data
    assert data['latency_ms'] < 50
```

#### **Load Tests (Locust)**
```python
class PredictionUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def predict(self):
        payload = generate_random_sensor_data()
        self.client.post("/predict", json=payload)
    
    @task(3)  # 3x more frequent
    def health_check(self):
        self.client.get("/health")
```

**Load Test Targets:**
- 1000 concurrent users
- 10-minute sustained load
- Success rate > 99.9%
- p95 latency < 50ms

---

## 📁 Project Structure

```
predictive-maintenance-mlops/
│
├── data/                          # Data directory (gitignored)
│   ├── raw/                       # NASA Turbofan dataset
│   ├── processed/                 # Cleaned data
│   ├── features/                  # Engineered features
│   └── models/                    # Trained model artifacts
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_xgboost.ipynb
│   ├── 04_lstm_model.ipynb
│   └── 05_ensemble.ipynb
│
├── src/                           # Source code
│   ├── __init__.py
│   │
│   ├── data/                      # Data processing
│   │   ├── __init__.py
│   │   ├── loader.py             # Data loading utilities
│   │   ├── validator.py          # Great Expectations
│   │   └── preprocessor.py       # Data cleaning
│   │
│   ├── features/                  # Feature engineering
│   │   ├── __init__.py
│   │   ├── builder.py            # Feature computation
│   │   ├── store.py              # Redis feature store
│   │   └── selector.py           # Feature selection
│   │
│   ├── models/                    # ML models
│   │   ├── __init__.py
│   │   ├── xgboost_model.py      # XGBoost wrapper
│   │   ├── lstm_model.py         # PyTorch LSTM
│   │   ├── ensemble.py           # Ensemble logic
│   │   └── explainer.py          # SHAP wrapper
│   │
│   ├── training/                  # Training pipeline
│   │   ├── __init__.py
│   │   ├── trainer.py            # Training orchestration
│   │   ├── evaluator.py          # Model evaluation
│   │   └── hyperparameter_tuner.py  # Optuna
│   │
│   ├── serving/                   # Prediction service
│   │   ├── __init__.py
│   │   ├── api.py                # FastAPI app
│   │   ├── predictor.py          # Prediction logic
│   │   └── schemas.py            # Pydantic models
│   │
│   ├── monitoring/                # Monitoring & drift
│   │   ├── __init__.py
│   │   ├── drift_detector.py     # Evidently wrapper
│   │   ├── metrics.py            # Prometheus metrics
│   │   └── alerting.py           # Slack notifications
│   │
│   ├── pipelines/                 # Orchestration
│   │   ├── __init__.py
│   │   ├── data_pipeline.py      # Data ingestion
│   │   ├── training_pipeline.py  # Model training
│   │   └── monitoring_pipeline.py # Drift detection
│   │
│   └── utils/                     # Utilities
│       ├── __init__.py
│       ├── config.py             # Configuration management
│       ├── logger.py             # Structured logging
│       └── db.py                 # Database connections
│
├── tests/                         # Test suite
│   ├── unit/
│   ├── integration/
│   └── load/
│
├── configs/                       # Configuration files
│   ├── model_config.yaml
│   ├── training_config.yaml
│   ├── api_config.yaml
│   └── monitoring_config.yaml
│
├── docker/                        # Docker configurations
│   ├── Dockerfile.api
│   ├── Dockerfile.trainer
│   ├── Dockerfile.mlflow
│   └── docker-compose.yml
│
├── deploy/                        # Deployment configs
│   ├── prometheus.yml
│   ├── grafana/
│   │   └── dashboards/
│   └── prefect/
│
├── docs/                          # Documentation
│   ├── api_documentation.md
│   ├── architecture.md
│   ├── deployment_guide.md
│   └── troubleshooting.md
│
├── scripts/                       # Utility scripts
│   ├── setup_db.sh
│   ├── load_data.py
│   ├── run_training.py
│   └── deploy_model.sh
│
├── .github/
│   └── workflows/
│       ├── ci.yml                # Run tests on PR
│       └── cd.yml                # Deploy on main
│
├── requirements/                  # Dependencies
│   ├── base.txt                  # Core deps
│   ├── training.txt              # ML training
│   ├── serving.txt               # API serving
│   └── dev.txt                   # Development tools
│
├── .env.example                   # Environment variables template
├── .gitignore
├── .dockerignore
├── pyproject.toml                # Project metadata
├── pytest.ini                    # Test configuration
├── README.md                     # Project overview
└── PROJECT_SPECIFICATION.md      # This file
```

---

## 🔐 Security & Configuration

### **Environment Variables**

```bash
# Database
DATABASE_URL=postgresql://user:password@postgres:5432/predictive_maintenance
REDIS_URL=redis://redis:6379/0

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_BACKEND_STORE_URI=postgresql://user:password@postgres:5432/mlflow
MLFLOW_ARTIFACT_ROOT=/mlflow/artifacts

# API
API_SECRET_KEY=<generate_with_openssl_rand_hex_32>
API_RATE_LIMIT=100  # requests per minute

# Monitoring
PROMETHEUS_URL=http://prometheus:9090
GRAFANA_ADMIN_PASSWORD=<secure_password>
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

# GPU
CUDA_VISIBLE_DEVICES=0  # Use GPU 0 for training
```

### **Security Best Practices**

1. ✅ Never commit `.env` file (use `.env.example`)
2. ✅ Use secrets management (Docker secrets or Vault in production)
3. ✅ Enable HTTPS for API (Traefik or nginx reverse proxy)
4. ✅ Implement rate limiting (100 req/min per API key)
5. ✅ Use JWT tokens with 1-hour expiration
6. ✅ Sanitize all user inputs (Pydantic validation)
7. ✅ Regular dependency updates (Dependabot)

---

## 🎯 Success Criteria

### **Technical Metrics**

| Metric | Target | Critical? |
|--------|--------|-----------|
| F2 Score | > 0.80 | ✅ YES |
| Precision | > 0.65 | ⚠️ Nice-to-have |
| Recall | > 0.85 | ✅ YES |
| API Latency (p95) | < 50ms | ✅ YES |
| Training Time | < 15 min | ⚠️ Nice-to-have |
| Test Coverage | > 85% | ✅ YES |
| Drift Detection Working | Yes | ✅ YES |
| Automated Retraining | Yes | ✅ YES |

### **Project Deliverables**

- ✅ **Working Platform**: Fully functional system with Docker Compose
- ✅ **Documentation**: API docs, architecture diagrams, setup guide
- ✅ **Demo Video**: 5-8 minute walkthrough (screen recording)
- ✅ **Blog Post**: Technical deep-dive (Medium/Dev.to)
- ✅ **GitHub Repo**: Clean commit history, README, CI/CD badges

### **Portfolio Impact**

**This project demonstrates:**
1. End-to-end ML system design
2. Production MLOps practices
3. Deep learning with GPU (PyTorch)
4. API development (FastAPI)
5. Containerization (Docker)
6. Monitoring & observability
7. Software engineering best practices

**Target Audience:** ML Engineer, MLOps Engineer, Data Scientist (production) roles

---

## 📅 Weekly Milestone Checklist

### **Week 1: Setup & EDA**
- [ ] Clone NASA Turbofan dataset
- [ ] Setup Python environment (Python 3.11 + CUDA 13.1)
- [ ] Exploratory data analysis notebook
- [ ] Understand failure patterns
- [ ] Define initial feature list (30 features)

### **Week 2: Feature Engineering**
- [ ] Implement rolling statistics
- [ ] Implement lag features
- [ ] Implement rate of change features
- [ ] Implement domain features
- [ ] Feature validation tests

### **Week 3: Baseline Model**
- [ ] Train XGBoost baseline
- [ ] Achieve F2 > 0.75
- [ ] Setup MLflow tracking
- [ ] Log experiments
- [ ] Feature importance analysis

### **Week 4: LSTM Model**
- [ ] Implement LSTM architecture (PyTorch)
- [ ] Train on GPU (RTX 5060)
- [ ] Sequence preparation pipeline
- [ ] Model evaluation (F2 > 0.74)
- [ ] GPU utilization monitoring

### **Week 5: Ensemble & API**
- [ ] Implement weighted ensemble
- [ ] Calibrate probabilities
- [ ] Ensemble evaluation (F2 > 0.80)
- [ ] FastAPI project setup
- [ ] Implement /predict endpoint

### **Week 6: API Completion**
- [ ] Implement /batch-predict
- [ ] Implement /health
- [ ] Implement /feedback
- [ ] SHAP integration
- [ ] API tests (unit + integration)

### **Week 7: Docker & Database**
- [ ] PostgreSQL schema design
- [ ] Redis feature store
- [ ] Dockerfiles (api, trainer, mlflow)
- [ ] Docker Compose configuration
- [ ] Local deployment test

### **Week 8: Training Pipeline**
- [ ] Prefect setup
- [ ] Data ingestion DAG
- [ ] Feature engineering DAG
- [ ] Training DAG
- [ ] Pipeline tests

### **Week 9: MLOps Features**
- [ ] Model registry (MLflow)
- [ ] Model versioning
- [ ] Automated retraining logic
- [ ] Shadow deployment
- [ ] Rollback mechanism

### **Week 10: Monitoring**
- [ ] Prometheus metrics
- [ ] Drift detection (Evidently)
- [ ] Grafana dashboards
- [ ] Alerting (Slack webhook)
- [ ] Load testing (Locust)

### **Week 11: Polish & Documentation**
- [ ] Complete README.md
- [ ] API documentation (Swagger)
- [ ] Architecture diagrams
- [ ] Deployment guide
- [ ] Code cleanup & refactoring

### **Week 12: Demo & Release**
- [ ] Record demo video
- [ ] Write technical blog post
- [ ] GitHub release (v1.0.0)
- [ ] LinkedIn/Twitter post
- [ ] Portfolio website update

---

## 🛠️ Technology Stack Summary

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| **Language** | Python | 3.11 | Core development |
| **ML Framework** | PyTorch | 2.2+ | LSTM model (GPU) |
| **Tree Models** | XGBoost | 2.0+ | Gradient boosting (GPU) |
| **API** | FastAPI | 0.110+ | REST API |
| **Orchestration** | Prefect | 2.14+ | Pipeline automation |
| **Tracking** | MLflow | 2.10+ | Experiment tracking |
| **Database** | PostgreSQL | 15 | Data lake |
| **Cache** | Redis | 7 | Feature store |
| **Monitoring** | Prometheus | 2.50+ | Metrics collection |
| **Visualization** | Grafana | 10.3+ | Dashboards |
| **Drift** | Evidently AI | 0.4+ | Data/prediction drift |
| **Validation** | Great Expectations | 0.18+ | Data quality |
| **Testing** | pytest | 8.0+ | Unit/integration tests |
| **Load Testing** | Locust | 2.20+ | Performance testing |
| **Containerization** | Docker | 25+ | Deployment |
| **Explainability** | SHAP | 0.44+ | Model interpretation |
| **Data Science** | pandas, NumPy, scikit-learn | Latest | Data processing |

---

## 🚨 Risk Mitigation

| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|--------|---------------------|
| **Model underperforms (F2 < 0.80)** | Low | High | Start with proven architecture (XGBoost + LSTM); extensive feature engineering; have baseline |
| **GPU memory overflow (>8GB)** | Low | Medium | Batch size tuning; gradient accumulation; model pruning |
| **Training takes too long (>15min)** | Medium | Low | Use GPU; cache intermediate results; profile code |
| **Docker resource constraints** | Low | Medium | Resource limits in compose; monitor with cAdvisor |
| **Time overrun** | Medium | Medium | Prioritize Phase 1-2; defer monitoring to Phase 4 |
| **Dataset issues** | Low | High | Multiple failure modes available; can use FD002-FD004 |
| **Integration complexity** | Low | Medium | Incremental integration; frequent testing |

---

## 📈 Next Steps (Start Here!)

### **Immediate Actions:**

1. **Create Project Structure** (10 minutes)
   ```bash
   mkdir -p {data/{raw,processed,features,models},notebooks,src/{data,features,models,training,serving,monitoring,pipelines,utils},tests/{unit,integration,load},configs,docker,deploy,docs,scripts}
   ```

2. **Setup Python Environment** (15 minutes)
   ```bash
   conda create -n pred-maint python=3.11
   conda activate pred-maint
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   pip install xgboost[gpu] pandas numpy scikit-learn mlflow fastapi
   ```

3. **Load Dataset** (5 minutes)
   - Data already in `turbofan_ed_dataset/`
   - Read `readme.txt`
   - Start with `train_FD001.txt` and `test_FD001.txt`

4. **Create First Notebook** (30 minutes)
   - `notebooks/01_eda.ipynb`
   - Load data, explore sensors, visualize degradation

---

## 💡 Key Decisions Made

1. **GPU Utilization**: Training only (not inference) - API uses CPU for simplicity
2. **Orchestration**: Prefect over Airflow - lighter for personal project
3. **Feature Count**: 50 (not 400) - avoid overfitting, faster iteration
4. **Model Weights**: XGBoost 60%, LSTM 40% - XGBoost more reliable
5. **Monitoring Delay**: Phase 4 - prioritize working model first
6. **Deployment**: Docker Compose - simpler than Kubernetes for portfolio
7. **Dataset**: FD001 primary - simplest failure mode for MVP

---

**Document Version:** 1.0  
**Last Updated:** February 7, 2026  
**Author:** [Your Name]  
**Status:** Ready to Start 🚀
