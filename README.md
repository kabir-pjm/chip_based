# Predictive Anomaly Detection & Automated Response System

> Real-time multi-zone monitoring with dual-model anomaly detection (Statistical + Isolation Forest), automated risk intervention, and full observability stack.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![SQLite](https://img.shields.io/badge/SQLite-Logging-green)
![License](https://img.shields.io/badge/License-MIT-gray)

---

## Problem Statement

Hardware thermal runaway can cause permanent damage within seconds. Traditional threshold-based monitoring is **reactive** — by the time temperature exceeds limits, it's too late.

This system solves that with a **predictive** approach: using time-series feature engineering and machine learning to detect anomalous *trends* before they become critical, then automatically intervening with zero-data-loss process control.

### Why This Architecture Matters Beyond Hardware

The core pattern — **streaming data → feature engineering → anomaly detection → automated response → observability** — is identical to:

| Domain | Streaming Signal | Anomaly | Response |
|--------|-----------------|---------|----------|
| **This Project** | Thermal sensors | Rising temp trend | SIGSTOP/SIGCONT |
| **Fraud Detection** | Transaction stream | Unusual spending pattern | Block card |
| **Trading Risk** | Market data feed | Position limit breach | Auto-hedge |
| **Infra Monitoring** | Server metrics | Memory leak trend | Auto-scale |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     SENSOR LAYER                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                      │
│  │  CPU    │  │  GPU    │  │  SSD    │   (3 thermal zones)  │
│  │ coretemp│  │ amdgpu  │  │  nvme   │                      │
│  └────┬────┘  └────┬────┘  └────┬────┘                      │
│       └─────────┬──┴───────────┘                             │
│                 ▼                                             │
│       ┌─────────────────────┐                                │
│       │ FEATURE ENGINE      │                                │
│       │ • Linear slope      │                                │
│       │ • Acceleration      │                                │
│       │ • Volatility        │                                │
│       │ • Cross-zone spread │                                │
│       └────────┬────────────┘                                │
│                ▼                                             │
│  ┌─────────────────────────────────────┐                     │
│  │      DUAL ANOMALY DETECTION         │                     │
│  │  ┌──────────────┐ ┌──────────────┐  │                     │
│  │  │ Statistical  │ │ Isolation    │  │                     │
│  │  │ Debounced    │ │ Forest (ML)  │  │                     │
│  │  │ Threshold    │ │ Unsupervised │  │                     │
│  │  └──────┬───────┘ └──────┬───────┘  │                     │
│  │         └───────┬────────┘          │                     │
│  │          FUSION ENGINE              │                     │
│  │     (weighted combination)          │                     │
│  └─────────────┬───────────────────────┘                     │
│                ▼                                             │
│  ┌──────────────────────┐  ┌────────────────────┐            │
│  │ PROCESS ACTUATOR     │  │ SQLITE LOGGER      │            │
│  │ SIGSTOP / SIGCONT    │  │ + Event History     │            │
│  │ (zero data loss)     │  │ + CSV Export        │            │
│  └──────────────────────┘  │ + Replay Engine     │            │
│                            └────────────────────┘            │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │              LIVE DASHBOARD                           │    │
│  │  Multi-zone charts │ Pipeline viz │ Process table     │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

---

## Key Technical Decisions

### 1. Dual-Model Anomaly Detection
**Statistical**: Sliding-window linear regression calculates temperature derivative (slope). A debounce buffer with configurable confidence threshold prevents false positives from sensor noise.

**Machine Learning**: Isolation Forest (unsupervised) trained on multi-dimensional feature vectors. Detects complex, non-linear anomaly patterns that simple thresholds miss. Auto-retrains periodically as data distribution shifts.

**Fusion**: Weighted combination of both scores. Statistical model handles known failure modes; ML model catches unknown patterns. Weights shift once ML model accumulates sufficient training data.

### 2. Multi-Zone Monitoring
Each thermal zone (CPU, GPU, SSD) has independent thresholds and characteristics. Cross-zone features (max temperature spread, correlated slope) detect system-wide thermal events.

### 3. Zero-Data-Loss Process Control
Uses POSIX `SIGSTOP` (suspend) instead of `SIGKILL` (terminate). Suspended processes retain all state in memory and resume instantly with `SIGCONT`. No data corruption, no restart overhead.

### 4. Full Observability
- **SQLite logging**: Every reading, every anomaly score, every action — timestamped and queryable
- **Event replay**: Reconstruct any incident from the database at configurable speed
- **CSV export**: Feed data into Jupyter, Pandas, or any analytics tool
- **Live dashboard**: Real-time visualization of the entire pipeline

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/[you]/thermal-governor.git
cd thermal-governor
pip install -r requirements.txt

# Run in simulation mode (no root needed)
python src/governor.py --sim

# Launch dashboard
python src/api_server.py
# Open http://localhost:8080

# Run tests
python -m pytest tests/ -v
```

---

## Project Structure

```
thermal-governor/
├── src/
│   ├── governor.py      # Main governor: sensors, features, detection, actuation
│   ├── database.py      # SQLite logging, replay engine, CSV export
│   └── api_server.py    # REST API + dashboard server
├── dashboard/
│   └── index.html       # Live monitoring dashboard (standalone)
├── tests/
│   └── test_governor.py # Unit tests (features, detection, DB, simulation)
├── data/                # SQLite DB + exports (auto-created)
├── docs/                # Additional documentation
├── requirements.txt
└── README.md
```

---

## Configuration

All parameters are tunable via `GovernorConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 60 | Sliding window for feature calculation |
| `poll_interval` | 0.5s | Sensor read frequency |
| `debounce_threshold` | 0.8 | Confidence required to trigger intervention |
| `ml_contamination` | 0.05 | Expected anomaly rate for Isolation Forest |
| `ml_retrain_interval` | 200 | Ticks between ML model retraining |

Per-zone thresholds (`ZoneConfig`):

| Zone | Safe Temp | Critical Temp | Weight |
|------|-----------|---------------|--------|
| CPU | 80°C | 95°C | 1.0 |
| GPU | 85°C | 100°C | 0.8 |
| SSD | 70°C | 85°C | 0.6 |

---

## Testing

```bash
$ python -m pytest tests/ -v

test_governor.py::TestFeatureEngine::test_slope_rising          PASSED
test_governor.py::TestFeatureEngine::test_slope_stable           PASSED
test_governor.py::TestFeatureEngine::test_acceleration_detection PASSED
test_governor.py::TestFeatureEngine::test_feature_vector_shape   PASSED
test_governor.py::TestAnomalyDetector::test_statistical_normal   PASSED
test_governor.py::TestAnomalyDetector::test_statistical_anomaly  PASSED
test_governor.py::TestAnomalyDetector::test_debounce_prevents_flicker PASSED
test_governor.py::TestDatabase::test_log_and_retrieve           PASSED
test_governor.py::TestDatabase::test_event_logging              PASSED
```

---

## Future Enhancements

- [ ] LSTM-based sequence prediction for longer-horizon forecasting
- [ ] Grafana integration via Prometheus exporter
- [ ] Kubernetes operator for container-level thermal management
- [ ] A/B testing framework for anomaly detection model comparison

---

## License

MIT
