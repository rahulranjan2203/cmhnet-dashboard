# CMH-Net: Cascading Multi-Hazard Early Warning System
## Senior Design Project (SDP) — Project Defense & Review Guide

---

## 1. 30-Second Elevator Pitch
> *"Most disaster prediction systems treat floods and landslides as isolated, independent events. In reality, they are **cascading hazards**: continuous heavy rainfall over several days saturates the soil and triggers riverine flooding, which in turn severely destabilizes mountain slopes and triggers landslides.*
>
> *Our project, **CMH-Net (Cascading Multi-Hazard Network)**, is a two-stage deep-learning and machine-learning framework. **Stage 1 (BiLSTM)** analyzes 7-day weather trends to predict flood risk and compresses the soil's hydrological memory into a 64-dimensional latent bridge vector. **Stage 2 (XGBoost)** takes that latent hydrological state, combines it with terrain topography (elevation, slope, wetness index), and computes the conditional landslide risk. Finally, we integrate **TreeSHAP** to provide transparent, explainable feature attributions for every prediction."*

---

## 2. The Problem We Are Solving

Traditional warning systems use **siloed / uncoupled models**:
1. A flood model looks only at rainfall.
2. A landslide model looks only at static slope angles and today's rain.

**Why this fails in real life:**
* Landslides rarely occur just because of today's rain. They happen because of **antecedent rainfall** (the rainfall accumulated over the past 3 to 7 days) that gradually weakens soil shear strength.
* If a model ignores the time sequence or ignores the flood state, it dramatically underpredicts compound disasters like debris flows.

---

## 3. How CMH-Net Works (Step-by-Step Architecture)

```
[Day 1 to Day 7 Weather: Rain, Temp, Humidity]
                      │
                      ▼
            [Log1p & Scaler Preprocessing]
                      │
                      ▼
         [Stage 1: PyTorch FloodBiLSTM]
         ┌────────────┴────────────┐
         ▼                         ▼
   Flood Probability      64-Dimensional Latent
   (Riverine Overflow)     Bridge Vector (z)
                                   │
                                   ▼
                   [Concatenate with Topography]
                   [Elevation, Slope, TWI] (3-D)
                                   │
                                   ▼
                    Compound 67-D Feature Vector
                                   │
                                   ▼
                     [Stage 2: XGBoost Booster]
                     ┌─────────────┴─────────────┐
                     ▼                           ▼
          Conditional Landslide            TreeSHAP
               Probability               Waterfall Plot
```

### Step 1: Hydrometeorological Preprocessing
* Input: 7 consecutive days of weather data `[Precipitation, Temperature, Relative Humidity]` (shape: `1 × 7 × 3`).
* **Log-Transform (`np.log1p`)**: Applied to rainfall to handle extreme skewness (rain can range from 0 mm to 300+ mm).
* **Z-Score Standardization**: Scaled using the pre-computed `scaler_flood.npz` (`mean` and `std`) so the neural network converges stably.

### Step 2: Stage 1 — Deep Bidirectional LSTM (`FloodBiLSTM`)
* **What model is used?** A 2-layer Bidirectional Long Short-Term Memory network (hidden size: 32).
* **Why Bidirectional?** 
  - Standard LSTMs only read forward in time.
  - A BiLSTM reads both forward (Day 1 $\rightarrow$ Day 7) and backward (Day 7 $\rightarrow$ Day 1), capturing both how rainfall accumulated and how recent intensity spikes compare to the weekly baseline.
* **Outputs of Stage 1**:
  1. **Flood Probability**: Extracted by passing the latent state through a linear head with a Sigmoid activation ($\in [0, 1]$).
  2. **The Latent Bridge ($\mathbf{z} \in \mathbb{R}^{64}$)**: Extracted by concatenating the final forward hidden state ($h_{-2}$, 32-dim) and backward hidden state ($h_{-1}$, 32-dim). This 64-number vector acts as a compressed numerical summary of the ground's moisture and hydrological saturation.

### Step 3: The "Cascading Bridge" (Information Transfer)
* Instead of throwing away the BiLSTM's internal representations, we pass the 64-dimensional latent vector to Stage 2.
* We append the 3 static topographic features:
  - **Elevation (m)**: Altitude above sea level.
  - **Slope Angle (°)**: Steepness of the hillside.
  - **Topographic Wetness Index (TWI)**: Physical metric $\ln(a / \tan \beta)$ measuring the propensity of a catchment to pool water.
* This forms a single **67-dimensional composite vector** ($64 + 3 = 67$).

### Step 4: Stage 2 — Gradient Boosted Trees (`XGBoost`)
* **What model is used?** An optimized XGBoost Classifier loaded from `xgb_cmhnet.json`.
* **Why XGBoost for Stage 2?**
  - Tabular data combining physical terrain indices with latent vectors is best handled by tree-based ensembles.
  - XGBoost excels at capturing non-linear interactions (e.g., *"If latent soil moisture is high AND slope angle $> 30^\circ$, landslide probability spikes exponentially"*).
* **Output**: **Conditional Landslide Probability** — the likelihood of a slope failure conditioned upon the prevailing hydrological state.

### Step 5: Model Interpretability via TreeSHAP
* Deep neural networks and tree ensembles are often criticized as "black boxes".
* We integrated **TreeSHAP (SHapley Additive exPlanations)** based on cooperative game theory.
* The **Waterfall Plot** visually displays:
  - The baseline hazard expectation ($E[f(x)]$).
  - Exactly which features (e.g., Slope, Elevation, or specific hydrological latent dimensions) pushed the risk **up** (red/positive contribution) or **down** (blue/negative contribution).
  - This allows disaster management authorities to trust and verify *why* an alert was issued.

---

## 4. Key Questions Reviewers Will Ask & How to Answer

### Q1: "Why use two different models (BiLSTM + XGBoost) instead of one single end-to-end model?"
> **Answer:** 
> *"Floods and landslides have fundamentally different data modalities. Floods are predominantly **temporal dynamic time-series** (best learned by recurrent networks like BiLSTMs), whereas landslides depend heavily on **static geographic/spatial tabular features** (slope, elevation, TWI) intersecting with soil moisture (best learned by tree-based ensembles like XGBoost). 
> By decoupling them via a 64-dimensional latent bridge, each model operates in its ideal regime while maintaining end-to-end cascading dependency."*

### Q2: "What does the 64-dimensional latent embedding actually represent?"
> **Answer:**
> *"In hydrology, directly using raw rainfall numbers ignores soil evaporation, drainage, and saturation history. The BiLSTM’s 64-d hidden state acts as a learned 'hydrological memory'—it captures cumulative antecedent moisture, precipitation persistence, and rate-of-change over the 7 days. It transfers this temporal context directly to the landslide classifier."*

### Q3: "What does 'Conditional Landslide Probability' mean?"
> **Answer:**
> *"A hillside's risk of collapse is not static. It is conditional on the antecedent environmental state: $P(\text{Landslide} \mid \text{Hydrology, Topography})$. Our model estimates the probability of landslide failure given the specific 7-day moisture history captured by Stage 1."*

### Q4: "Why did you use SHAP?"
> **Answer:**
> *"In emergency and disaster response, black-box predictions are unacceptable to civil authorities. SHAP provides local, mathematically rigorous feature attributions based on Shapley values. The waterfall plot tells the operator exactly which factor—whether steep terrain or prolonged ground saturation—is the primary driver behind the warning."*

### Q5: "How does the log1p transform help precipitation?"
> **Answer:**
> *"Daily precipitation data is zero-inflated and highly right-skewed (many dry days with 0 mm, few extreme storm days with $> 150\text{ mm}$). Applying $\log(1 + x)$ compresses the dynamic range, prevents extreme outliers from destabilizing the LSTM gradients, and ensures normal numerical behavior during scaling."*

---

## 5. Live Demonstration Script (Step-by-Step for Presentation)

1. **Open the Live Website**:
   Show the URL: `https://cmhnet-dashboard-qpwdmkvv39efkmuhmsjcqg.streamlit.app/`
2. **Showcase Scenario 1: Dry / Normal Conditions**:
   - In the sidebar, select **"Dry Mild Weather (Baseline Low Risk)"**.
   - Click **"Run CMH-Net Risk Analysis"**.
   - Point to both cards: Flood Probability ($\approx 0-5\%$) and Landslide Probability ($\approx 5-15\%$). Both chips show **Low Risk**.
3. **Showcase Scenario 2: Severe Monsoonal Deluge (Cascading Compound Event)**:
   - In the sidebar, select **"Severe Monsoonal Deluge (Cascading High Risk)"**.
   - Show the 7-day hyetograph where rainfall climbs to $280\text{ mm}$ on Day 7 on a $38.5^\circ$ slope.
   - Click **"Run CMH-Net Risk Analysis"**.
   - Point out the red warning cards: **Critical Compound Hazard Alert**.
4. **Showcase the SHAP Waterfall Plot**:
   - Scroll down to the waterfall plot.
   - Explain: *"Here the reviewer can see that the steep slope and the deep latent moisture vectors contributed positively (red arrows) to elevate the risk from baseline to severe."*
5. **Showcase the Latent Feature Inspector**:
   - Expand the bottom drawer to show the actual 64-dimensional latent vector and the 67-feature vector table.

---

## 6. Summary of Technologies Used
* **Frontend / Deployment**: Streamlit, Streamlit Community Cloud
* **Stage 1 (Deep Learning)**: PyTorch (`nn.LSTM`, Bidirectional, Dropout, Linear)
* **Stage 2 (Machine Learning)**: XGBoost (`xgb.Booster`, `xgb.DMatrix`, Tree ensemble)
* **Explainability (XAI)**: SHAP (`shap.TreeExplainer`, `shap.Explanation`, Waterfall plots)
* **Data Processing & Scientific Computing**: NumPy, Pandas, Matplotlib
