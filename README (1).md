# 📊 Kaggle Airline Satisfaction: Information Theory & Triple Stack Ensemble

This repository contains a fully custom, end-to-end reproducible machine learning pipeline developed for the **Kaggle Playground Series (Season 6, Episode 10)** competition (*Predicting Airline Passenger Satisfaction*).

Rather than relying on brute-force hyperparameter tuning or unverified public prediction sharing, this solution focuses entirely on **Information Theory features**, **psychological non-linear data transformations**, and a robust **Level-1 Triple Stack Architecture** optimized via **Hill Climbing Coordinate Descent**.

---

## 📐 Pipeline Architecture & Data Flow

Below is the complete algorithmic data flow mapping how raw passenger statistics are transformed into optimized ensemble probabilities:

```
[Raw train.csv & test.csv]
           │
           ▼
┌────────────────────────────────────────────────────────┐
│      1. GEOMETRIC ROUTE LEAK MAPPING                   │
│      Flight Distance ──> High-Cardinality route_id     │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│      2. ADVANCED NON-LINEAR FEATURE ENGINEERING        │
│  ├─ Shannon Entropy Profile (Cross-Service Variance)   │
│  ├─ Psychological Time Dilation (Delay Compression)    │
│  └─ Target-Style Out-of-Fold Route Comfort Averages    │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│      3. LEVEL-1 TRIPLE STACK DIVERSITY ENGINES         │
│  ├─ LightGBM (Leaf-wise structural optimization)      │
│  ├─ XGBoost  (Histogram categorical matching layer)    │
│  └─ CatBoost (Symmetric depth + Ordered Target Enc.)   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│      4. META-ENSEMBLE WEIGHT OPTIMIZATION              │
│      Hill Climbing Coordinate Descent (Max Local AUC)  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
              [Optimized master_submission.csv]
```

---

## 🧮 Mathematical Engine Foundations

### 1. Shannon Entropy Profile (Cross-Service Emotional Variance)
Passengers evaluate 14 specific functional service categories (Wi-Fi, Cleanliness, Seat Comfort, Legroom, etc.) on a discrete scale of $0$ to $5$. Traditional gradient boosted trees examine these columns sequentially, completely missing the "transverse" relationship across a single row.

To quantify the emotional consistency or erratic nature of a passenger's flight experience, we calculate the **Shannon Information Entropy** ($\mathcal{H}$) across the discrete probability distribution $P(x)$ of their service scores within that specific row:

$$\mathcal{H} = -\sum_{i=0}^{5} P(x_i) \log_2 P(x_i)$$

* **Low Entropy ($\mathcal{H} 	o 0$):** High consistency in responses (e.g., a passenger giving straight $4$s or $5$s), indicating a highly stable customer experience profile.
* **High Entropy ($\mathcal{H} \gg 0$):** Highly erratic responses (e.g., extreme $1$ score for Wi-Fi but a $5$ for Food), exposing critical friction points that standard averages wash out.

### 2. Psychological Time Dilation (Non-Linear Delay Scale)
Flight departure and arrival delays contain extreme, long-tailed outliers that insert severe noise into tree splittings. From a human behavioral standpoint, a delay progression from $0$ minutes to $30$ minutes feels drastically different, whereas a delay difference between $180$ and $210$ minutes is psychologically identical.

To match model thresholds with human emotional realities, we project raw delays onto a compact, bounded $[0, 1]$ space using a **Relativistic Damping Factor** ($\gamma$):

$$\gamma = rac{1.0}{\sqrt{1.0 + rac{	ext{Delay}_{	ext{Departure}} + 	ext{Delay}_{	ext{Arrival}}}{10.0}}}$$

This forces the optimization algorithms to prioritize fine-grained splits in the low-delay zone while completely insulating the tree nodes from training noise caused by massive multi-hour flight delays.

---

## 🏆 Model Level-1 Layer Configuration

To establish maximal structural diversity and shield the final submission from public leaderboard over-fitting (*Shake-up Protection*), three separate boosting architectures are trained concurrently inside a strict **5-Fold Stratified Cross-Validation пайплайн**:

| Model Architecture | Growth Strategy | Special Feature Processing | Validation Peak (OOF AUC) |
| :--- | :--- | :--- | :--- |
| **LightGBM** | Leaf-wise (Best-first) | High responsiveness to $\mathcal{H}$ profile | `0.95940` |
| **XGBoost** | Tree-wise (Depth-first) | Strict category synchronization across splits | `0.95931` |
| **CatBoost** | Symmetric Trees | Ordered Target Encoding for `route_id` | **`0.96037`** |

---

## 🏎️ Meta-Ensemble Optimization (Hill Climbing)

Rather than blending model matrices with static, guessed ratios (e.g., $0.33$ each), the meta-ensemble layer applies a **Coordinate Descent Hill Climbing** search algorithm. 

The optimizer monitors the combined validation space, executing localized shifts in model weight arrays ($ec{W}$) to maximize the global area under the ROC curve:

$$rg\max_{ec{W}} 	ext{ROC-AUC} \left( Y_{	ext{true}}, \sum_{m \in M} W_m \cdot 	ext{OOF}_{m} ight) \quad 	ext{subject to } \sum W_m = 1.0, \, W_m \ge 0$$

This yields an optimized, robust blend where the superior structural traits of the leader (CatBoost) are perfectly reinforced and insulated by the secondary boosting frameworks.

---

## 🛠️ Repository File Structure

* `Masaya_Kawamata_Master_Framework.ipynb` — The production-ready framework architecture template containing pre-built classes for features, category sync layers, and weight optimizers.
* `airline_satisfaction_triple_stack.ipynb` — The active training session running the raw data, cross-validation runs, and logs.
* `README.md` — Complete engineering documentation and algorithmic breakdowns.

---
*Disclaimer: Developed independently as an author-driven machine learning methodology.*