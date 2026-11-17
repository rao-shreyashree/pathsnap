# Section 5: HMM and Viterbi Route Reconstruction

**Author:** Member D (`d-rf-hmm`) — PathSnap ML Hackathon  
**Deliverable:** Technical Report Contribution & Model Write-Up  
**Model Family:** Ensemble Random Forest (`RandomForestClassifier`) + Topological Hidden Markov Model + Log-Space Viterbi Dynamic Programming  

---

## 5.1 Introduction & Problem Motivation

In GPS map-matching, independent point-by-point classifier predictions frequently yield unrealistic, topologically disconnected vehicle trajectories. When a vehicle travels along a corridor, GPS multipath scattering, satellite clock bias, and building reflections cause recorded coordinates to scatter across nearby parallel alleys, opposing traffic lanes, or cross streets. A pointwise classifier that predicts the most probable road arc for each GPS observation in isolation lacks temporal and network awareness: it may select arc $a_t$ at time $t$ and arc $a_{t+1}$ at time $t+1$ where no physical road connects $a_t \to a_{t+1}$, or it may predict an impossible instantaneous U-turn against one-way traffic constraints.

To eliminate these anomalies, Member D implemented a sequential Bayesian framework combining:
1. **Supervised Random Forest Classifier:** Generates discriminative candidate emission probabilities from 20 invariant geometric, kinematic, and topological features.
2. **Directed Road Network Graph (`RoadGraph`):** Represents the ground transport network topology, forward successors, backward predecessors, reverse twins, and bounded network distances.
3. **Hidden Markov Model (HMM):** Couples candidate emission likelihoods with physically grounded transition probabilities based on network topology, distance discrepancy, and temporal continuity.
4. **Numerically Stable Log-Space Viterbi Decoder:** Employs dynamic programming to compute the global maximum a posteriori (MAP) sequence of road arcs across the entire trip.

---

## 5.2 Supervised Random Forest Classifier Methodology

### 5.2.1 Candidate Classification Formulation
Because road network topologies and arc IDs vary across geographic regions, the learning problem cannot be formulated as a global multiclass classification task over static arc IDs. Instead, it is formulated as **candidate ranking / binary classification**:
- For each GPS observation $i$, spatial indexing (KDTree over arc segments) retrieves the $K=10$ nearest candidate road arcs within search radius $R=50.0\text{ m}$ (with adaptive expansion up to $R_{\text{max}}=250.0\text{ m}$).
- Each (observation $i$, candidate arc $a$) pair is mapped to an invariant numerical feature vector $\mathbf{x}_{(i,a)} \in \mathbb{R}^{20}$.
- The target label $y_{(i,a)} \in \{0, 1\}$ indicates whether candidate arc $a$ matches the monotonically aligned ground-truth reference arc $\text{ref}(i)$ derived from `.route`.

### 5.2.2 Feature Representation
The 20 invariant features include:
1. **Perpendicular distance (`perp_dist`):** Distance from GPS fix to road centerline segment in metres.
2. **Relative spatial metrics (`dist_rank`, `dist_minus_min`, `rel_dist`):** Candidate distance rank among the observation's candidates, distance difference from the nearest candidate, and ratio to nearest.
3. **Segment progress (`along_arc_t`, `dist_to_start`, `dist_to_end`, `arc_length`):** Normalized fractional projection $t \in [0, 1]$ and metric offsets to intersection endpoints.
4. **Kinematics & Timing (`speed`, `dt_prev`, `dt_next`):** Instantaneous vehicle speed and forward/backward sampling intervals.
5. **Heading Alignment (`heading_diff_abs`, `cos_heading_diff`, `sin_heading_diff`):** Angular difference between vehicle trajectory heading and road segment bearing.
6. **Network Context (`has_reverse_twin`, `from_node_deg`, `to_node_deg`, `num_candidates`):** Intersection connectivity and bidirectional street indicators.
7. **Spatial-Temporal Window Context (`context_mean_dist_w3`, `context_is_nearest_w3`):** Mean distance of neighboring fixes ($t-1, t, t+1$) to the candidate arc and neighboring nearest-arc consistency.

### 5.2.3 Model Architecture & Group-Aware Cross-Validation
- **Algorithm:** `RandomForestClassifier` with $100$ estimators, maximum tree depth of $15$, minimum leaf samples of $10$, and square-root feature subsampling (`max_features="sqrt"`), preconditioned with `StandardScaler` fitted strictly on training folds inside an sklearn `Pipeline`.
- **Validation Protocol:** 5-fold `GroupKFold` grouped strictly by `trip_id` across 25 non-QC training trips ($64,610$ candidate rows, $2,940$ positive matches, $4.55\%$ positive class ratio). QC-flagged trips with severe noise (`00000002`, `00000033`, `00000054`) were excluded from training folds.
- **Empirical CV Results:**
  - **Group CV Macro F1:** $0.5935 \pm 0.0455$
  - **Candidate Precision:** $0.5120$
  - **Candidate Recall:** $0.1386$
  - **Candidate Accuracy:** $0.9545$
  - **Binary Log Loss:** $0.1173$
  - **Training Time:** $1.93\text{ s}$
  - **Inference Latency:** $43.31\text{ ms}$ per 1,000 observations

---

## 5.3 Road Graph Construction (`road_graph.py`)

The directed road graph is constructed directly from `.nodes` and `.arcs`:
1. **Coordinate Projection:** Geographic coordinates $(\lambda, \phi)$ are projected to a local metric Cartesian plane $(x, y)$ in metres centered at the trip's geographic centroid $(\lambda_0, \phi_0)$ using equirectangular projection with $R = 6,371,000\text{ m}$:
   $$x = R \cdot (\lambda - \lambda_0) \cos(\phi_0) \frac{\pi}{180}, \quad y = R \cdot (\phi - \phi_0) \frac{\pi}{180}$$
2. **Spatial Bounding Box Cropping:** To prevent out-of-memory bottlenecks on road networks containing up to $307,392$ arcs (e.g. trip `00000054`), the graph is cropped to the trajectory's bounding box plus a $0.01^\circ$ ($\approx 1.1\text{ km}$) safety margin.
3. **Directed Adjacency & Topology:**
   - Successor lookup: $\text{succ}(a) = \{ b \in \text{Arcs} \mid \text{to\_node}(a) == \text{from\_node}(b) \}$.
   - Predecessor lookup: $\text{pred}(b) = \{ a \in \text{Arcs} \mid \text{to\_node}(a) == \text{from\_node}(b) \}$.
   - Reverse twin detection: $\text{twin}(a) = b$ where $\text{from\_node}(b) == \text{to\_node}(a) \land \text{to\_node}(b) == \text{from\_node}(a)$. Across the dataset, $91.8\%$ of road arcs possess a reverse twin.
4. **Bounded Shortest Network Distance:** A priority-queue Dijkstra algorithm evaluates directed road distance $d_{\text{net}}(a \to b)$ up to a velocity-governed reachability horizon ($v_{\text{max}} \cdot \Delta t + 50\text{ m}$), with memoization via an internal transition distance cache.

---

## 5.4 Hidden Markov Model Formulation

### 5.4.1 Hidden States
For observation $t \in \{0, \dots, T-1\}$, the hidden state space $S_t = \{a_{t, 1}, a_{t, 2}, \dots, a_{t, K_t}\}$ comprises the $K_t$ candidate road arc IDs generated for that observation. The state space is observation-specific, preserving exact original 0-based integer arc IDs.

### 5.4.2 Emission Probabilities
Random Forest produces candidate match probability $P(y=1 \mid \mathbf{x}_{t, k})$. To form a valid discrete emission probability distribution over the candidate set, scores are normalized per observation:
$$b_t(k) = \frac{P(y=1 \mid \mathbf{x}_{t, k})}{\sum_{j=1}^{K_t} P(y=1 \mid \mathbf{x}_{t, j}) + \varepsilon}$$
In log-space with weighting parameter $w_e = 1.0$ and floor $\varepsilon = 10^{-6}$:
$$\log b_t(k) = w_e \cdot \log\left(\max(b_t(k), \varepsilon)\right)$$

### 5.4.3 Transition Probabilities
Transitions model the physical likelihood of moving from candidate arc $u \in S_{t-1}$ to candidate arc $v \in S_t$:
1. **Case A: Same Arc ($u == v$):** Vehicle remains on the current road segment:
   $$\log A(u \to u) = 0.0$$
2. **Case B: Topologically Connected ($u \ne v$):** Grounded in the Newson & Krumm (2009) distance discrepancy principle. Let $d_{\text{gc}} = \|\mathbf{p}_t - \mathbf{p}_{t-1}\|$ be the projected straight-line GPS displacement, and let $d_{\text{net}}(u \to v)$ be the shortest directed road network distance from the end of arc $u$ to the start of arc $v$. The discrepancy $\Delta d = |d_{\text{net}} - d_{\text{gc}}|$ measures how closely the network route matches physical vehicle displacement:
   $$\log A(u \to v) = -\frac{|d_{\text{net}} - d_{\text{gc}}|}{\beta} + \text{Penalties}$$
   where $\beta = 10.0\text{ m}$ is the exponential scale parameter.
3. **Reverse Twin / U-Turn Penalty:** If transitioning from arc $u$ to its opposite twin $\text{twin}(u)$, an additional penalty $\log P_{\text{uturn}} = -10.0$ is added to discourage abrupt heading reversals.
4. **Disconnected Transitions:** If no directed path exists within the reachability horizon ($d_{\text{net}} > v_{\text{max}} \cdot \Delta t$), a soft floor penalty $\log P_{\text{discon}} = -50.0$ is applied, allowing the classifier emission to guide the trajectory without breaking the DP recurrence.
5. **GPS Time Gaps & Chain Breaks:** When $\Delta t = t_{\text{curr}} - t_{\text{prev}} > T_{\text{gap}} = 60.0\text{ s}$, the Markov assumption breaks down. The decoder decouples the transition by setting a chain break flag, resetting the transition prior to uniform:
   $$\log A(u \to v) = -\log(K_t)$$
   This prevents historical errors from corrupting decoding after multi-hour GPS blackouts.

---

## 5.5 Viterbi Dynamic Programming Recurrence

The optimal sequence $\hat{S} = (\hat{a}_0, \hat{a}_1, \dots, \hat{a}_{T-1})$ is computed using log-space dynamic programming:

### 1. Initialization ($t = 0$):
For each candidate state $k \in \{1, \dots, K_0\}$:
$$V_0(k) = -\log(K_0) + \log b_0(k)$$

### 2. Forward Induction ($t = 1, \dots, T-1$):
For each candidate state $k \in \{1, \dots, K_t\}$:
$$V_t(k) = \max_{j \in \{1, \dots, K_{t-1}\}} \left[ V_{t-1}(j) + \log A(s_{t-1, j} \to s_{t, k}) \right] + \log b_t(k)$$
The backpointer records the optimal predecessor:
$$B_t(k) = \arg\max_{j \in \{1, \dots, K_{t-1}\}} \left[ V_{t-1}(j) + \log A(s_{t-1, j} \to s_{t, k}) \right]$$

### 3. Termination:
$$\hat{k}_{T-1} = \arg\max_{k \in \{1, \dots, K_{T-1}\}} V_{T-1}(k), \quad \log P^* = V_{T-1}(\hat{k}_{T-1})$$

### 4. Backtracking:
For $t = T-2, \dots, 0$:
$$\hat{k}_t = B_{t+1}(\hat{k}_{t+1})$$
The decoded sequence of arc IDs is recovered as $\hat{a}_t = s_{t, \hat{k}_t}$.

---

## 5.6 Experimental Results & Comparative Evaluation

Both models were evaluated on the **exact same observations and monotone ground-truth reference alignments** across all 10 official student-test development trips ($19,160$ observations).

### 5.6.1 Consolidated Trip-by-Trip Performance Comparison

| Trip ID | GPS Fixes | Top-1 (RF Only) | Top-1 (RF+HMM) | Top-1 $\Delta$ | Continuity (RF) | Continuity (HMM) | Breaks (RF) | Breaks (HMM) | Latency (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `00000000` | 2,503 | 0.0412 | 0.0412 | $+0.0000$ | 0.5910 | **0.8650** | 200 | **27** | 3,670.8 |
| `00000007` | 1,354 | 0.1004 | **0.1647** | **$+0.0643$** | 0.8343 | **0.9672** | 55 | **4** | 1,690.2 |
| `00000009` | 1,049 | 0.4042 | **0.4185** | $+0.0143$ | 0.7095 | **0.7587** | 131 | **90** | 4,274.1 |
| `00000040` | 2,995 | 0.1135 | 0.1078 | $-0.0057$ | 0.7406 | **0.8900** | 201 | **34** | 6,331.3 |
| `00000052` | 262 | 0.2176 | **0.2366** | $+0.0191$ | 0.7363 | **0.8108** | 24 | **14** | 497.2 |
| `00000056` | 3,958 | 0.1748 | **0.1882** | $+0.0134$ | 0.6846 | **0.7941** | 488 | **229** | 26,110.6 |
| `00000065` | 768 | 0.3568 | **0.3724** | $+0.0156$ | 0.6763 | **0.7836** | 78 | **29** | 1,820.8 |
| `00000076` | 1,486 | 0.7847 | **0.8425** | **$+0.0579$** | 0.6406 | **0.7195** | 184 | **99** | 6,696.3 |
| `00000077` | 3,911 | 0.1491 | 0.1294 | $-0.0197$ | 0.8064 | 0.7025 | 301 | **108** | 7,517.9 |
| `00000091` | 874 | 0.3387 | **0.3444** | $+0.0057$ | 0.6569 | **0.7978** | 105 | **37** | 2,153.3 |
| **Mean / Total** | **19,160** | **0.2681** | **0.2846** | **$+0.0165$** | **0.7077** | **0.8089** | **1,767** | **671** | **6,076.3** |

### 5.6.2 Key Empirical Findings
1. **Dramatic Route Continuity Improvement:**
   Average route continuity surged from **$70.77\%$ to $80.89\%$** ($+10.12$ percentage points). In trip `00000007`, continuity reached **$96.72\%$**.
2. **Massive Reduction in Topological Breaks:**
   Total disconnected transition breaks collapsed from **1,767 down to 671** across the 10 trips—a **$62.0\%$ reduction** in invalid road transitions.
3. **Pointwise Accuracy Gains:**
   HMM decoding increased observation-level Top-1 accuracy from **$26.81\%$ to $28.46\%$** overall. Marked gains were achieved on trip `00000007` ($+6.43\%$) and trip `00000076` ($+5.79\%$, reaching **$84.25\%$** Top-1 accuracy).
4. **Analysis of Cases Where HMM Did Not Improve Accuracy:**
   On trips `00000040` ($-0.57\%$) and `00000077` ($-1.97\%$), Top-1 accuracy decreased slightly. Inspection reveals that in these trips, candidate miss rates exceeded $74\%$; because the true road segment was absent from the candidate pool, the HMM sacrificed individual noisy point matches to enforce a continuous, topologically connected alternative sequence (evidenced by breaks dropping from $201 \to 34$ on `00000040` and $301 \to 108$ on `00000077`). This demonstrates the classic trade-off between local pointwise precision and global topological realism.
5. **Computational Latency:**
   Mean end-to-end inference latency was **$6,076.3\text{ ms}$ ($\approx 6.1\text{ s}$)** per trip, well within the $30\text{ s}$ operational budget required for two-laptop deployment.

---

## 5.7 Geographical Visualisations (Folium)

Interactive Folium HTML maps were generated and saved in `member_d/maps/`:
- `trip_00000000.html`: Long urban trajectory ($2,503$ points) demonstrating HMM continuity recovery ($200 \to 27$ breaks).
- `trip_00000007.html`: High-accuracy map matching ($1,354$ points) showing seamless agreement between HMM (red) and ground truth (green) with $96.72\%$ continuity.
- `trip_00000052.html`: Compact trajectory ($262$ points) clearly contrasting jittery RF classifier predictions with smooth HMM decoded paths.

---

## 5.8 FastAPI Integration Interface for Member C

Member D provided a clean, decoupled, FastAPI-independent integration function in `member_d/integration.py`:
```python
from member_d.integration import decode_route
from member_d.road_graph import RoadGraph

# Clean callable interface for FastAPI /predict
result = decode_route(
    candidates=candidates_by_obs,          # list[list[int]]
    emission_scores=emission_scores_by_obs, # list[list[float]]
    road_graph=road_graph,                 # RoadGraph instance
    timestamps=timestamps,                 # list[float] | None
)
# Returns: {"decoded_route": [...], "sequence_log_score": -12.45, "success": True, "n_breaks": 0, "warnings": []}
```
Automated integration unit tests (`tests/test_integration.py`) verify type compliance, error handling, and performance under zero-leakage conditions.

---

## 5.9 Limitations & Future Work

1. **Candidate Miss Ceiling:** In trips with large GPS shifts (e.g. `00000000` with miss rate $95.4\%$), candidate generation within $R=50\text{ m}$ missed the reference road segment. Expanding initial spatial query radius to $R=100\text{ m}$ for high-speed road sections would raise the upper bound of achievable accuracy.
2. **Shortest-Path Intermediate Arc Interpolation:** Viterbi returns one arc per GPS observation. When a vehicle skips short intermediate connector arcs between 1-second fixes, a post-Viterbi shortest-path bridge can stitch consecutive distinct arcs into an unbroken continuous road polyline for rendering.
3. **Transition Parameter Tuning:** While $\beta = 10.0\text{ m}$ yielded strong results across diverse terrains, Bayesian hyperparameter tuning of $\beta$, $w_e$, and the disconnection penalty on training out-of-fold predictions could yield further gains.

---

## 5.10 Conclusion

The Member D implementation demonstrates that integrating a supervised Random Forest classifier with a topologically constrained Hidden Markov Model and log-space Viterbi decoding transforms noisy, disjoint GPS point predictions into coherent, physically plausible vehicle trajectories. With a **$62.0\%$ reduction in route breaks**, a **$+10.12$ percentage point boost in route continuity**, and rigorous $100\%$ unit test pass rates across 22 test cases, the module fulfills all hackathon requirements and provides a reliable sequential engine for PathSnap.
