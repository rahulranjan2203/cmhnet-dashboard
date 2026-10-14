import os
import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib

# Set matplotlib backend to non-interactive
matplotlib.use('Agg')

# ==============================================================================
# XGBOOST & SHAP COMPATIBILITY PATCH
# ==============================================================================
# In newer XGBoost versions, base_score is serialized as a vector string '[5E-1]'.
# This monkey-patch sanitizes the config string so TreeExplainer does not raise ValueError.
try:
    _orig_save_config = xgb.Booster.save_config
    def _patched_save_config(self, *args, **kwargs):
        raw_cfg = _orig_save_config(self, *args, **kwargs)
        try:
            import json
            cfg_dict = json.loads(raw_cfg)
            if "learner" in cfg_dict and "learner_model_param" in cfg_dict["learner"]:
                bs = str(cfg_dict["learner"]["learner_model_param"].get("base_score", ""))
                if bs.startswith("[") and bs.endswith("]"):
                    cfg_dict["learner"]["learner_model_param"]["base_score"] = bs.strip("[]")
                    return json.dumps(cfg_dict)
        except Exception:
            pass
        return raw_cfg
    xgb.Booster.save_config = _patched_save_config
except Exception:
    pass

# ==============================================================================
# 1. STREAMLIT PAGE CONFIGURATION & CUSTOM STYLING
# ==============================================================================
st.set_page_config(
    page_title="CMH-Net | Cascading Multi-Hazard Early Warning System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI design, badges, and metric cards
st.markdown("""
<style>
    /* Global styling */
    .main {
        background-color: #0d1117;
    }
    
    /* Header hero styling */
    .hero-container {
        padding: 1.8rem 2.2rem;
        background: linear-gradient(135deg, rgba(14, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.85) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        margin-bottom: 2rem;
    }
    
    .badge-sdp {
        background: linear-gradient(90deg, #2563eb, #06b6d4);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 0.8rem;
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 0.5rem;
        line-height: 1.5;
    }
    
    /* Metric Cards */
    .metric-card {
        padding: 1.5rem 1.8rem;
        border-radius: 14px;
        background: #1e293b;
        border: 1px solid #334155;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    .metric-title {
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 0.6rem;
    }
    
    .hazard-chip {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    
    /* Risk levels */
    .chip-low { background: rgba(34, 197, 94, 0.18); color: #4ade80; border: 1px solid #22c55e; }
    .chip-moderate { background: rgba(234, 179, 8, 0.18); color: #facc15; border: 1px solid #eab308; }
    .chip-high { background: rgba(249, 115, 22, 0.18); color: #fb923c; border: 1px solid #f97316; }
    .chip-severe { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
    
    /* Info Callouts */
    .model-info-box {
        background: #1e293b;
        border-left: 4px solid #38bdf8;
        padding: 1rem 1.3rem;
        border-radius: 0 10px 10px 0;
        margin: 1.2rem 0;
        color: #cbd5e1;
        font-size: 0.92rem;
    }
    
    /* Sidebar Section Divider */
    .sidebar-section-hdr {
        font-weight: 700;
        color: #38bdf8;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 1.2rem;
        margin-bottom: 0.4rem;
        border-bottom: 1px solid #334155;
        padding-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. MODEL ARCHITECTURE (REQUIRED PYTORCH CLASS)
# ==============================================================================
class FloodBiLSTM(nn.Module):
    """
    Stage 1: Deep Bidirectional LSTM for Antecedent Hydrological Modeling.
    Extracts temporal dependencies over 7-day hydrometeorological sequences
    and exposes a 64-dimensional latent bridge embedding.
    """
    def __init__(self, n_in=3, hid=32):
        super().__init__()
        self.lstm = nn.LSTM(
            n_in, hid, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2
        )
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(2 * hid, 1)

    def embed(self, x):
        _, (h, _) = self.lstm(x)
        # Concatenate forward and backward final hidden states of the top LSTM layer
        return torch.cat([h[-2], h[-1]], dim=1)

    def forward(self, x):
        z = self.embed(x)
        return self.head(self.drop(z)).squeeze(1)


# ==============================================================================
# 3. RESOURCE CACHING & ARTIFACT LOADING
# ==============================================================================
@st.cache_resource(show_spinner="Loading CMH-Net PyTorch & XGBoost models...")
def load_all_artifacts():
    """
    Loads and caches:
    1. PyTorch FloodBiLSTM state_dict ('bilstm.pt')
    2. Scaler parameters ('scaler_flood.npz')
    3. XGBoost Booster ('xgb_cmhnet.json')
    """
    errors = []
    
    # Check required files
    required_files = ['bilstm.pt', 'scaler_flood.npz', 'xgb_cmhnet.json']
    for rf in required_files:
        if not os.path.exists(rf):
            errors.append(f"Missing required artifact: `{rf}` in app directory.")
    
    if errors:
        return None, None, None, errors

    try:
        # 1. Load Scaler
        scaler_data = np.load('scaler_flood.npz')
        mean = scaler_data['mean']
        std = scaler_data['std']

        # 2. Load PyTorch BiLSTM
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = FloodBiLSTM(n_in=3, hid=32)
        state_dict = torch.load('bilstm.pt', map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # 3. Load XGBoost Booster
        booster = xgb.Booster()
        booster.load_model('xgb_cmhnet.json')

        return model, (mean, std), booster, []
    except Exception as exc:
        return None, None, None, [str(exc)]


# ==============================================================================
# 4. PREPROCESSING & INFERENCE PIPELINE
# ==============================================================================
def preprocess_weather(weather_data_7d, scaler_tuple):
    """
    Preprocesses 7-day weather inputs:
    - Shape: (1, 7, 3) where features are [Precipitation, Temperature, Humidity]
    - Applies np.log1p ONLY to Precipitation (index 0)
    - Applies standardization using scaler_flood.npz (mean and std)
    - Converts to torch.float32 tensor
    """
    mean, std = scaler_tuple
    
    # Format into (1, 7, 3) array
    weather_arr = np.array(weather_data_7d, dtype=np.float32).reshape(1, 7, 3)
    
    # Apply np.log1p ONLY to Precipitation (index 0)
    weather_arr[:, :, 0] = np.log1p(weather_arr[:, :, 0])
    
    # Safe division to prevent division by zero
    safe_std = np.where(std == 0, 1e-7, std)
    
    # Robust broadcasting: handles (3,), (1, 3), (7, 3), (1, 7, 3), or flattened (21,)
    if mean.size == 21 and mean.ndim == 1:
        flattened = weather_arr.reshape(1, 21)
        standardized = (flattened - mean) / safe_std
        standardized = standardized.reshape(1, 7, 3)
    else:
        standardized = (weather_arr - mean) / safe_std
    
    # Convert to PyTorch float32 tensor
    tensor_x = torch.tensor(standardized, dtype=torch.float32)
    return tensor_x


def run_stage1_inference(model, tensor_x):
    """
    Stage 1 Inference:
    - Pass tensor_x to FloodBiLSTM
    - Return Flood Probability = sigmoid(model(x))
    - Extract 64-dimensional latent bridge embedding = model.embed(x)
    """
    device = next(model.parameters()).device
    tensor_x = tensor_x.to(device)
    
    with torch.no_grad():
        raw_logit = model(tensor_x)
        flood_prob = torch.sigmoid(raw_logit).item()
        latent_embed = model.embed(tensor_x).detach().cpu().numpy() # shape (1, 64)
        
    return float(flood_prob), latent_embed


def run_stage2_inference(booster, latent_embed, terrain_features):
    """
    Stage 2 Inference:
    - Concatenates 64-d bridge embedding and 3 terrain features: Elevation, Slope, TWI
    - Creates a 67-dimensional numpy array of shape (1, 67)
    - Converts to xgboost.DMatrix and passes through Booster
    - Returns Landslide Probability
    """
    # Terrain array of shape (1, 3)
    terrain_arr = np.array(terrain_features, dtype=np.float32).reshape(1, 3)
    
    # Concatenate into 67-dimensional feature vector
    features_67 = np.concatenate([latent_embed, terrain_arr], axis=1) # shape (1, 67)
    
    # Build feature names
    feature_names = [f"Latent_h{i:02d}" for i in range(64)] + [
        "Elevation (m)", "Slope (°)", "TWI"
    ]
    
    dmatrix = xgb.DMatrix(features_67, feature_names=feature_names)
    landslide_prob = float(booster.predict(dmatrix)[0])
    
    return landslide_prob, features_67, feature_names


def explain_with_shap(booster, features_67, feature_names):
    """
    Computes SHAP feature attributions on the XGBoost Booster
    and generates a high-resolution waterfall plot.
    Uses native XGBoost pred_contribs with SHAP Explanation fallback
    to prevent base_score parsing incompatibilities across versions.
    """
    fig, ax = plt.subplots(figsize=(10, 6.2))
    dmatrix = xgb.DMatrix(features_67, feature_names=feature_names)
    
    # 1. Primary Method: Native XGBoost TreeSHAP (100% robust & fast)
    try:
        contribs = booster.predict(dmatrix, pred_contribs=True)
        # contribs shape: (1, 68), last element is the expected base value
        shap_vals = contribs[0, :-1]
        base_val = float(contribs[0, -1])
        
        explanation = shap.Explanation(
            values=shap_vals,
            base_values=base_val,
            data=features_67[0],
            feature_names=feature_names
        )
        shap.plots.waterfall(explanation, max_display=12, show=False)
        fig = plt.gcf()
    except Exception:
        # 2. Secondary Method: SHAP TreeExplainer
        try:
            explainer = shap.TreeExplainer(booster)
            df_features = pd.DataFrame(features_67, columns=feature_names)
            explanation = explainer(df_features)
            shap.plots.waterfall(explanation[0], max_display=12, show=False)
            fig = plt.gcf()
        except Exception:
            # 3. Tertiary Method: Legacy shap_values waterfall
            explainer = shap.TreeExplainer(booster)
            shap_values = explainer.shap_values(features_67)
            expected_val = explainer.expected_value
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            if isinstance(expected_val, (list, np.ndarray)):
                expected_val = float(expected_val[0])
            shap.waterfall_plot(
                expected_val,
                shap_values[0],
                features=features_67[0],
                feature_names=feature_names,
                max_display=12,
                show=False
            )
            fig = plt.gcf()

    # Style Matplotlib figure for modern presentation
    plt.title("Stage 2: XGBoost SHAP Waterfall Plot (Feature Attributions)", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    return fig


def get_risk_meta(probability):
    """Returns color-coded hazard severity label and CSS class."""
    if probability < 0.25:
        return "Low Risk", "chip-low", "#22c55e"
    elif probability < 0.50:
        return "Moderate Risk (Watch)", "chip-moderate", "#eab308"
    elif probability < 0.75:
        return "High Risk (Warning)", "chip-high", "#f97316"
    else:
        return "Severe Risk (Evacuation)", "chip-severe", "#ef4444"


# ==============================================================================
# 5. SIDEBAR USER INTERACTION (WEATHER & TOPOGRAPHY)
# ==============================================================================
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 1.2rem;">
    <h2 style="color: #38bdf8; margin: 0; font-size: 1.5rem;">CMH-Net Controls</h2>
    <span style="color: #94a3b8; font-size: 0.85rem;">Hydrometeorological & Topographic Inputs</span>
</div>
""", unsafe_allow_html=True)

# Scenario Presets definition with high contrast in hydrological saturation
PRESETS = {
    "Severe Monsoonal Deluge (Cascading High Risk)": {
        "weather": [
            (35.0, 24.0, 85), (65.0, 23.0, 88), (110.0, 22.0, 92),
            (160.0, 21.5, 95), (220.0, 21.0, 98), (290.0, 20.5, 99), (340.0, 20.0, 100)
        ],
        "elevation": 1850.0,
        "slope": 42.0,
        "twi": 11.5
    },
    "Moderate Rain on Steep Mountain Slope": {
        "weather": [
            (8.0, 24.0, 65), (15.0, 23.5, 70), (28.0, 23.0, 75),
            (45.0, 22.0, 82), (65.0, 21.5, 86), (80.0, 21.0, 90), (105.0, 20.5, 94)
        ],
        "elevation": 2100.0,
        "slope": 34.0,
        "twi": 7.5
    },
    "Dry Mild Weather (Baseline Low Risk)": {
        "weather": [
            (0.0, 28.0, 42), (0.0, 29.0, 40), (0.0, 27.5, 45),
            (0.5, 26.0, 50), (0.0, 27.0, 46), (1.0, 26.5, 48), (0.0, 28.0, 44)
        ],
        "elevation": 550.0,
        "slope": 10.0,
        "twi": 4.8
    }
}

def load_preset_into_state(preset_name):
    if preset_name in PRESETS:
        p_data = PRESETS[preset_name]
        for d_i, (p, t, h) in enumerate(p_data["weather"]):
            st.session_state[f"precip_d{d_i+1}"] = float(p)
            st.session_state[f"temp_d{d_i+1}"] = float(t)
            st.session_state[f"rh_d{d_i+1}"] = int(h)
        st.session_state["elev_state"] = float(p_data["elevation"])
        st.session_state["slope_state"] = float(p_data["slope"])
        st.session_state["twi_state"] = float(p_data["twi"])

# Initialize session state on initial run
if "precip_d1" not in st.session_state:
    st.session_state["preset_selector"] = "Moderate Rain on Steep Mountain Slope"
    load_preset_into_state("Moderate Rain on Steep Mountain Slope")
    st.session_state["has_evaluated"] = False

def on_preset_change():
    selected = st.session_state.get("preset_selector")
    if selected in PRESETS:
        load_preset_into_state(selected)
        st.session_state["has_evaluated"] = True

st.sidebar.selectbox(
    "Load Scenario Preset:",
    [
        "Moderate Rain on Steep Mountain Slope",
        "Severe Monsoonal Deluge (Cascading High Risk)",
        "Dry Mild Weather (Baseline Low Risk)",
        "Custom Configuration"
    ],
    key="preset_selector",
    on_change=on_preset_change
)

st.sidebar.markdown('<div class="sidebar-section-hdr">7-Day Weather Sequence</div>', unsafe_allow_html=True)
st.sidebar.caption("Sequence order: Precipitation (mm), Temp (°C), Humidity (%)")

weather_inputs = []
day_tabs = st.sidebar.tabs([f"D{d}" for d in range(1, 8)])

for idx, tab in enumerate(day_tabs):
    with tab:
        st.markdown(f"**Day {idx + 1} Conditions**")
        precip = st.number_input(
            f"Precipitation (mm) - Day {idx+1}",
            min_value=0.0, max_value=800.0, step=1.0,
            key=f"precip_d{idx+1}"
        )
        temp = st.number_input(
            f"Temperature (°C) - Day {idx+1}",
            min_value=-20.0, max_value=55.0, step=0.5,
            key=f"temp_d{idx+1}"
        )
        rh = st.slider(
            f"Relative Humidity (%) - Day {idx+1}",
            min_value=1, max_value=100, step=1,
            key=f"rh_d{idx+1}"
        )
        weather_inputs.append([precip, temp, rh])

# Static Topography inputs
st.sidebar.markdown('<div class="sidebar-section-hdr">Static Topography</div>', unsafe_allow_html=True)
elevation = st.sidebar.number_input(
    "Elevation (meters)",
    min_value=0.0, max_value=8848.0, step=25.0,
    key="elev_state",
    help="Altitude above sea level"
)
slope = st.sidebar.slider(
    "Slope Angle (degrees)",
    min_value=0.0, max_value=75.0, step=0.5,
    key="slope_state",
    help="Terrain inclination angle"
)
twi = st.sidebar.number_input(
    "Topographic Wetness Index (TWI)",
    min_value=0.0, max_value=30.0, step=0.1,
    key="twi_state",
    help="ln(a / tan(beta)): tendency of site to accumulate soil moisture"
)


# ==============================================================================
# 6. MAIN APPLICATION PAGE
# ==============================================================================
# Hero Header
st.markdown("""
<div class="hero-container">
    <span class="badge-sdp">Senior Design Project • Capstone Deployment</span>
    <h1 class="hero-title">Cascading Multi-Hazard Early Warning System (CMH-Net)</h1>
    <p class="hero-subtitle">
        Coupled deep-learning & gradient-boosting architecture for compound disaster prediction.
        <strong>Stage 1 (BiLSTM)</strong> processes temporal antecedent hydrometeorology into a latent bridge embedding,
        conditioning <strong>Stage 2 (XGBoost)</strong> for terrain-coupled landslide hazard forecasting.
    </p>
</div>
""", unsafe_allow_html=True)

# Architecture Summary Expander
with st.expander("System Architecture and Technical Methodology", expanded=False):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("##### 1. Hydrometeorological BiLSTM")
        st.caption(
            "2-layer Bidirectional LSTM (hidden dim: 32). Processes log-transformed precipitation, "
            "temperature, and relative humidity over 7 days. Extracts a 64-d forward-backward latent state."
        )
    with col_b:
        st.markdown("##### 2. Latent Bridge Vector")
        st.caption(
            "Latent bridge $\\mathbf{z} \\in \\mathbb{R}^{64}$ concatenates with Static Topography "
            "[Elevation, Slope, TWI] $\\in \\mathbb{R}^3$ forming a compound 67-dimensional representation."
        )
    with col_c:
        st.markdown("##### 3. Conditional XGBoost & SHAP")
        st.caption(
            "Predicts conditional landslide triggering probability given antecedent hydrological state. "
            "Interpreted via TreeSHAP feature attributions on the 67-dimensional feature space."
        )

# Load Models
model_bilstm, scaler_tuple, booster_xgb, load_errors = load_all_artifacts()

if load_errors:
    st.error("### Model Artifact Loading Error")
    for err in load_errors:
        st.warning(f"- {err}")
    st.info("Please verify that `bilstm.pt`, `scaler_flood.npz`, and `xgb_cmhnet.json` reside in the application directory.")
    st.stop()

# 7-Day Weather Trend Visualization
with st.container():
    df_chart = pd.DataFrame({
        "Day": [f"Day {i}" for i in range(1, 8)],
        "Precipitation (mm)": [w[0] for w in weather_inputs],
        "Temperature (°C)": [w[1] for w in weather_inputs],
        "Humidity (%)": [w[2] for w in weather_inputs]
    }).set_index("Day")
    
    with st.expander("7-Day Weather Profile & Hyetograph", expanded=False):
        c1, c2 = st.columns([2, 1])
        with c1:
            st.bar_chart(df_chart["Precipitation (mm)"])
        with c2:
            st.line_chart(df_chart[["Temperature (°C)", "Humidity (%)"]])

# Primary Execution Button
st.markdown("<br>", unsafe_allow_html=True)
button_col1, button_col2, button_col3 = st.columns([1, 2, 1])
with button_col2:
    run_analysis = st.button("Run CMH-Net Risk Analysis", type="primary", use_container_width=True)

# Main Inference Execution: runs on button click OR when a preset is selected
should_run = run_analysis or st.session_state.get("has_evaluated", False)

if run_analysis:
    st.session_state["has_evaluated"] = True

if should_run:
    with st.spinner("Executing CMH-Net Cascading Inference Pipeline..."):
        try:
            # 1. Preprocessing
            tensor_x = preprocess_weather(weather_inputs, scaler_tuple)

            # 2. Stage 1: Flood Inference
            flood_prob, latent_embed = run_stage1_inference(model_bilstm, tensor_x)

            # 3. Stage 2: Landslide Inference
            terrain_features = [elevation, slope, twi]
            landslide_prob, features_67, feature_names = run_stage2_inference(
                booster_xgb, latent_embed, terrain_features
            )

            # Risk metadata
            flood_label, flood_chip, flood_color = get_risk_meta(flood_prob)
            ls_label, ls_chip, ls_color = get_risk_meta(landslide_prob)

            st.markdown("<hr style='border-color: #334155; margin: 1.8rem 0;'>", unsafe_allow_html=True)
            st.markdown("### Multi-Hazard Prediction Results")

            # Dual Metric Output Columns Side-by-Side
            col_metric1, col_metric2 = st.columns(2)

            with col_metric1:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 5px solid {flood_color};">
                    <div class="metric-title">Stage 1: Flood Probability</div>
                    <div class="metric-value" style="color: {flood_color};">{flood_prob * 100:.2f}%</div>
                    <span class="hazard-chip {flood_chip}">{flood_label}</span>
                    <p style="margin-top: 0.8rem; font-size: 0.86rem; color: #94a3b8; line-height: 1.4;">
                        Estimated by <strong>FloodBiLSTM</strong> based on 7-day antecedent rainfall accumulation and moisture trends.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            with col_metric2:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 5px solid {ls_color};">
                    <div class="metric-title">Stage 2: Conditional Landslide Probability</div>
                    <div class="metric-value" style="color: {ls_color};">{landslide_prob * 100:.2f}%</div>
                    <span class="hazard-chip {ls_chip}">{ls_label}</span>
                    <p style="margin-top: 0.8rem; font-size: 0.86rem; color: #94a3b8; line-height: 1.4;">
                        Estimated by <strong>XGBoost</strong> conditioned on the 64-d hydrological bridge embedding and terrain susceptibility.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Cascading Compound Interaction Alert
            if flood_prob > 0.5 and landslide_prob > 0.5:
                st.error(
                    "**CRITICAL COMPOUND HAZARD ALERT:** Simultaneous flood and landslide trigger thresholds exceeded. "
                    "Saturated hillslopes combined with severe riverine discharge indicate severe cascading debris flow risk."
                )
            elif flood_prob > 0.5 and landslide_prob <= 0.5:
                st.warning(
                    "**FLOOD WARNING:** Riverine overflow likely. Mountain slopes currently remain below failure threshold, "
                    "but low-lying drainage zones must prepare for inundation."
                )
            elif flood_prob <= 0.5 and landslide_prob > 0.5:
                st.warning(
                    "**LANDSLIDE WARNING:** High terrain susceptibility on steep topography. Localized slope destabilization "
                    "triggered even with moderate riverine flood levels."
                )
            else:
                st.success(
                    "**NORMAL CONDITIONS:** Compound hazard parameters are within standard baseline thresholds."
                )

            # ==================================================================
            # 7. SHAP EXPLAINABILITY INTEGRATION
            # ==================================================================
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### Model Interpretability: TreeSHAP Feature Attributions")
            st.markdown(
                "The waterfall plot below reveals how the topographic parameters and deep hydrological latent bridge "
                "features pushed the model prediction from the baseline expected log-odds to the final landslide output."
            )

            with st.spinner("Generating TreeSHAP waterfall explanation..."):
                fig_shap = explain_with_shap(booster_xgb, features_67, feature_names)
                st.pyplot(fig_shap, use_container_width=True)
                plt.close(fig_shap)

            # Latent Bridge Feature Breakdown
            with st.expander("Latent Bridge Embedding (64-D) and Terrain Feature Vector", expanded=False):
                tab_vec, tab_table = st.tabs(["Latent State Representation", "Feature Vector Table"])
                with tab_vec:
                    st.caption("64-dimensional vector extracted from FloodBiLSTM embed(x) representing compressed antecedent hydrology:")
                    st.line_chart(latent_embed.flatten(), height=180)
                with tab_table:
                    df_vec = pd.DataFrame({
                        "Feature Identifier": feature_names,
                        "Value": features_67.flatten()
                    })
                    st.dataframe(df_vec, use_container_width=True, height=280)

        except Exception as e:
            st.error(f"Execution Error during Risk Analysis: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
else:
    st.info("Configure 7-day weather conditions and topographic parameters in the sidebar, then click **Run CMH-Net Risk Analysis** above.")

# Footer
st.markdown("<br><hr style='border-color: #1e293b;'>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>"
    "CMH-Net Early Warning System • Senior Design Project • Built with Streamlit, PyTorch & XGBoost"
    "</div>",
    unsafe_allow_html=True
)
