# Member D (`d-rf-hmm`) — PathSnap Map-Matching & Sequential Route Decoder

**Owner:** Member D — `d-rf-hmm` (Random Forest, Road Graph, HMM & Viterbi Decoder)  
**Track:** UE24CS352A Machine Learning Hackathon  

---

## 1. Overview & Architecture

Member D independently develops the **sequential route reconstruction and map-matching core** for PathSnap:
- **`road_graph.py`:** Parses raw `.nodes` and `.arcs` files, applies local equirectangular metric projection, crops networks to bounding boxes, and builds directed topological adjacency structures (successors, predecessors, reverse twins, and bounded shortest-path distances).
- **`random_forest.py`:** Trains a supervised `RandomForestClassifier` on 20 invariant geometric and kinematic features with 5-fold `GroupKFold` cross-validation grouped by `trip_id`. Normalizes candidate scores into calibrated observation emission distributions.
- **`hmm_decoder.py`:** Couples RF candidate emissions with Newson-Krumm transition modeling, incorporating distance discrepancy, U-turn penalties, and time-gap chain breaking.
- **`viterbi.py`:** Numerically stable log-space dynamic programming algorithm with backpointer storage and backtracking.
- **`evaluate_d.py`:** Comprehensive benchmark comparing RF-only vs RF+HMM+Viterbi on identical observations and ground truth across all 10 student-test development trips.
- **`visualize_route.py`:** Multi-layer interactive Folium HTML map generator (raw GPS points, road network, reference route, RF predictions, HMM decoded path).
- **`integration.py`:** Standalone callable interface `decode_route` for Member C's FastAPI backend service.

---

## 2. Directory Layout

```text
member_d/
├── __init__.py
├── config.py                 # Paths, random seed, constants, and hyperparameters
├── road_graph.py             # Road graph construction, topology, distance, successor lookup
├── random_forest.py          # Random Forest classifier, group CV, probability scoring adapter
├── hmm_decoder.py            # HMM state definition, emissions, transition model, gap handling
├── viterbi.py                # Numerically stable log-space Viterbi DP and backtracking
├── evaluate_d.py             # Rigorous comparison harness: RF-only vs RF+HMM+Viterbi
├── visualize_route.py        # Folium interactive HTML multi-layer map generator
├── integration.py            # Decoupled callable interface `decode_route` for Member C's FastAPI
├── tests/                    # 22 automated unit and integration tests (100% pass)
│   ├── test_road_graph.py
│   ├── test_viterbi.py
│   ├── test_hmm_decoder.py
│   ├── test_random_forest.py
│   └── test_integration.py
├── models/
│   └── random_forest_model.joblib # Serialized trained Random Forest pipeline
├── results/
│   ├── rf_training_meta.json          # CV metrics and training parameters
│   ├── student_test_comparison.csv    # Trip-by-trip evaluation comparison (10 trips)
│   ├── student_test_summary.json      # Aggregate benchmark statistics
│   ├── evaluator_test_comparison.csv  # Official evaluator test comparison (10 trips)
│   └── evaluator_test_summary.json    # Official evaluator test aggregate summary
├── maps/
│   ├── trip_00000000.html            # Student test trip 00000000 map
│   ├── trip_00000007.html            # Student test trip 00000007 map (96.7% continuity)
│   ├── trip_00000052.html            # Student test trip 00000052 map
│   ├── trip_00000015.html            # Evaluator test trip 00000015 map (6,854 points)
│   ├── trip_00000023.html            # Evaluator test trip 00000023 map
│   └── trip_00000073.html            # Evaluator test trip 00000073 map (89.4% continuity)
├── MEMBER_D_REPORT.md        # Complete Section 5 technical report contribution
└── README.md                 # Setup, execution, and integration guide (this file)
```

---

## 3. Quickstart & Exact Execution Commands

Run all commands from the repository root:

### 3.1 Run All Unit & Integration Tests (22 tests)
```bash
python -m pytest member_d/tests/ -v
```

### 3.2 Train Random Forest & Run Group Cross-Validation
```bash
python member_d/random_forest.py
```
*Outputs:*
- Trained model saved to: `member_d/models/random_forest_model.joblib`
- Metadata saved to: `member_d/results/rf_training_meta.json`

### 3.3 Run Comparative Evaluation on 10 Student-Test Trips
```bash
python member_d/evaluate_d.py
```
*Outputs:*
- Detailed comparison CSV: `member_d/results/student_test_comparison.csv`
- Aggregate summary JSON: `member_d/results/student_test_summary.json`
- Interactive Folium maps: `member_d/maps/trip_*.html`

---

## 4. Empirical Evaluation Results (Actual Measured Data)

### 4.1 Development Benchmark (10 Student-Test Trips, 19,160 Observations)

| Evaluation Metric | Random Forest Only (Pointwise) | RF + HMM + Viterbi (Sequential) | Impact / Delta |
| :--- | :---: | :---: | :---: |
| **Observation-Level Top-1 Accuracy** | 26.81% | **28.46%** | **$+1.65\%$** (Up to $+6.43\%$) |
| **Route Continuity** | 70.77% | **80.89%** | **$+10.12$ percentage points** |
| **Total Disconnected Breaks** | 1,767 | **671** | **$-1,096$ breaks ($-62.0\%$)** |
| **Candidate Miss Rate** | 65.31% | 65.31% | (Identical candidate pool) |
| **Mean Inference Latency** | 108.5 ms | 1,101.9 ms | **6.08 s total trip latency** |

### 4.2 Official Evaluator Benchmark (10 Reserved Trips, 19,674 Observations)

| Evaluation Metric | Random Forest Only (Pointwise) | RF + HMM + Viterbi (Sequential) | Impact / Delta |
| :--- | :---: | :---: | :---: |
| **Observation-Level Top-1 Accuracy** | 35.14% | **37.41%** | **$+2.28\%$** (Up to $+7.37\%$) |
| **Route Continuity** | 74.07% | **75.48%** | **$+1.41$ percentage points** |
| **Total Disconnected Breaks** | 1,563 | **794** | **$-769$ breaks ($-49.2\%$)** |
| **Candidate Miss Rate** | 58.44% | 58.44% | (Identical candidate pool) |
| **Mean Inference Latency** | 98.4 ms | 1,029.5 ms | **4.96 s total trip latency** |

---

## 5. Integration Guide for Member C (FastAPI Backend)

Member C can import and call `decode_route` directly in `api/main.py` without any dependency on Streamlit, training code, or ground-truth routes.

### Exact Import Statement
```python
from member_d.integration import decode_route
from member_d.road_graph import RoadGraph
```

### Integration Example in FastAPI `/predict`
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from member_d.road_graph import RoadGraph
from member_d.integration import decode_route
from member_d.random_forest import RandomForestModel

app = FastAPI(title="PathSnap API")

# Load model once at startup
rf_model = RandomForestModel.load("member_d/models/random_forest_model.joblib")

class PredictRequest(BaseModel):
    trip_id: str

@app.post("/predict")
def predict_trip(req: PredictRequest):
    trip_dir = f"data/raw/student_test/{req.trip_id}"
    
    # 1. Build road graph for trip
    road_graph = RoadGraph.from_trip_directory(trip_dir, crop=True)
    
    # 2. Score candidates with classifier
    # cands_by_obs: list[list[int]], scores_by_obs: list[list[float]]
    cands_by_obs, scores_by_obs, top1_rf = rf_model.score_candidates(X, candidates)
    
    # 3. Decode with Member D's Viterbi engine
    result = decode_route(
        candidates=cands_by_obs,
        emission_scores=scores_by_obs,
        road_graph=road_graph,
        timestamps=timestamps,
    )
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail="HMM decoding failed")
        
    return {
        "trip_id": req.trip_id,
        "predicted_arcs": result["decoded_route"],
        "sequence_log_score": result["sequence_log_score"],
        "n_breaks": result["n_breaks"],
    }
```
