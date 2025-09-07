import streamlit as st
import os, sys, pickle, pandas as pd, numpy as np
from datetime import datetime
from difflib import get_close_matches
from pathlib import Path

st.set_page_config(page_title="Car Price (Kenya) - Year Aware", page_icon="🚗", layout="centered")

# Hide Streamlit branding on mobile
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display:none;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────
# Path setup & artifact loading
# ────────────────────────────────────────────────────────────────────────────────

def setup_project_path():
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    return current_dir

PROJECT_ROOT = setup_project_path()

MODEL_FILE_MAPPING = {
    "model": "gradientboosting_tuned.pkl",
    "y_tf": "target_transformer.pkl",
    "feat_info": "feature_info.pkl",
    "priors": "priors.pkl"
}

ABS_PATHS = {
    key: os.path.join(PROJECT_ROOT, "models", filename)
    for key, filename in MODEL_FILE_MAPPING.items()
}

RELATIVE_FALLBACKS = {
    key: [f"models/{filename}", f"notebooks/models/{filename}"]
    for key, filename in MODEL_FILE_MAPPING.items()
}

def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

@st.cache_resource
def load_artifacts():
    from src.artifact_loader import load_all_artifacts
    return load_all_artifacts(PROJECT_ROOT, MODEL_FILE_MAPPING)

# ────────────────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────────────────

def canon(s): 
    return str(s).strip().casefold()

def resolve_name(user_input, canon_map, all_values, cutoff=0.7):
    c = canon(user_input)
    if c in canon_map:
        return canon_map[c]
    close = get_close_matches(user_input.strip(), all_values, n=1, cutoff=cutoff)
    return close[0] if close else user_input.strip()

def sanitize_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        return df
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    if not obj_cols:
        return df
    df = df.copy()
    for c in obj_cols:
        df[c] = df[c].astype(str)
    return df

def resolve_priors_with_year(make_name, model_name, year, priors):
    make_name = resolve_name(make_name, priors['make_canon_map'], priors['all_makes'])
    model_name = resolve_name(model_name, priors['model_canon_map'], priors['all_models'])
    year = int(year)
    year_bin = (year // 5) * 5

    lvl1 = priors.get('level1_exact', pd.DataFrame())
    if isinstance(lvl1, pd.DataFrame) and not lvl1.empty:
        exact = lvl1[(lvl1['make_name'] == make_name) &
                     (lvl1['model_name'] == model_name) &
                     (lvl1['year_of_manufacture'] == year)]
        if len(exact) > 0:
            return exact.iloc[0].to_dict(), f" Exact match ({make_name} {model_name} {year})", make_name, model_name

    lvl2 = priors.get('level2_year_bin', pd.DataFrame())
    if isinstance(lvl2, pd.DataFrame) and not lvl2.empty:
        m2 = lvl2[(lvl2['make_name'] == make_name) &
                  (lvl2['model_name'] == model_name) &
                  (lvl2['year_bin'] == year_bin)]
        if len(m2) > 0:
            return m2.iloc[0].to_dict(), f" Year range match ({make_name} {model_name} {year_bin}s)", make_name, model_name

    lvl3 = priors.get('level3_make_model', pd.DataFrame())
    if isinstance(lvl3, pd.DataFrame) and not lvl3.empty:
        m3 = lvl3[(lvl3['make_name'] == make_name) & (lvl3['model_name'] == model_name)]
        if len(m3) > 0:
            return m3.iloc[0].to_dict(), f" Make/model match ({make_name} {model_name})", make_name, model_name

    lvl4 = priors.get('level4_make_year', pd.DataFrame())
    if isinstance(lvl4, pd.DataFrame) and not lvl4.empty:
        m4 = lvl4[(lvl4['make_name'] == make_name) & (lvl4['year_bin'] == year_bin)]
        if len(m4) > 0:
            return m4.iloc[0].to_dict(), f" Make/year match ({make_name}, {year_bin}s)", make_name, model_name

    lvl5 = priors.get('level5_make', pd.DataFrame())
    if isinstance(lvl5, pd.DataFrame) and not lvl5.empty:
        m5 = lvl5[lvl5['make_name'] == make_name]
        if len(m5) > 0:
            return m5.iloc[0].to_dict(), f"🏷️ Make only ({make_name})", make_name, model_name

    lvl6 = priors.get('level6_year', pd.DataFrame())
    if isinstance(lvl6, pd.DataFrame) and not lvl6.empty:
        m6 = lvl6[lvl6['year_bin'] == year_bin]
        if len(m6) > 0:
            return m6.iloc[0].to_dict(), f" Year range only ({year_bin}s)", make_name, model_name

    global_dict = priors.get('level7_global', {})
    return dict(global_dict), " Global defaults", make_name, model_name

def create_feature_dict(
    make_name, 
    model_name, 
    year, 
    prior_data, 
    user_inputs=None
):
    """
    Build feature dict. Uses user_inputs if provided, else falls back to prior_data.
    """
    current_year = datetime.now().year
    year = int(year)
    car_age = max(0, current_year - year)

    # Start with priors
    features = {
        "model_name": model_name,
        "make_name": make_name,
        "name": f"{make_name} {model_name}",
        "year_of_manufacture": year,
        "car_age": car_age,

        "mileage": float(prior_data.get("mileage", 80000)),
        "annual_insurance": float(prior_data.get("annual_insurance", 120000)),
        "torque": float(prior_data.get("torque", 300)),
        "engine_size": float(prior_data.get("engine_size", 2000)),
        "horse_power": float(prior_data.get("horse_power", 170)),
        "acceleration": float(prior_data.get("acceleration", 10.0)),

        "body_type": prior_data.get("body_type", "SUV"),
        "drive": prior_data.get("drive", "2WD"),
        "fuel_type": prior_data.get("fuel_type", "Petrol"),
        "transmission": prior_data.get("transmission", "Automatic"),
        "usage_type": prior_data.get("usage_type", "Kenyan Used"),
        "condition": prior_data.get("condition", "Very Good"),
    }

    # Override with user inputs if provided
    if user_inputs:
        for key, value in user_inputs.items():
            if value is not None:
                features[key] = value

    # Back-compat
    features["usage type"] = features["usage_type"]

    # Clamp numeric bounds
    bounds = {
        "mileage": (0, 1_000_000),
        "annual_insurance": (1_000, 5_000_000),
        "torque": (50, 1000),
        "engine_size": (500, 8000),
        "horse_power": (30, 2000),
        "acceleration": (2.0, 50.0),
    }
    for k, (lo, hi) in bounds.items():
        if k in features:
            features[k] = min(max(features[k], lo), hi)

    return features

def predict_price(features_dict, artifacts):
    try:
        from importlib import reload
        import src.pipeline
        reload(src.pipeline)
        from src.pipeline import prepare_dataset

        input_df = pd.DataFrame([features_dict])
        X_processed, _, _, _ = prepare_dataset(input_df, is_train=False)

        model = artifacts["model"]
        y_tf = artifacts["y_tf"]
        pred_log = model.predict(X_processed)[0]
        price = y_tf.inverse_transform([[pred_log]])[0][0]
        return price, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

# ────────────────────────────────────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────────────────────────────────────

st.markdown("""
#  Ensemble Machine Learning Model for Predicting Prices of Used Cars in Kenya
### _Powered by Machine Learning + Year-Aware Market Intelligence_
""", unsafe_allow_html=True)

with st.spinner("📦 Loading models..."):
    artifacts, load_err = load_artifacts()

if load_err:
    st.error("⚠️ Model loading failed. Please check that all model files are valid in the GitHub release.")
    with st.expander("Error Details"):
        st.code(load_err)
    st.stop()

st.success("✅ Models ready")

# Main required inputs
col1, col2 = st.columns(2)
with col1:
    make_input = st.text_input("Make", value="Toyota", help="e.g., Toyota, Nissan, BMW")
with col2:
    model_input = st.text_input("Model", value="Fortuner", help="e.g., Fortuner, X-Trail, X5")

year_input = st.number_input(
    "Year of Manufacture",
    min_value=1990,
    max_value=datetime.now().year + 1,
    value=2015,
)

# Advanced inputs in expanders
user_inputs = {}

with st.expander(" Advanced: Numeric Features (Optional)"):
    st.markdown("###  Financial & Performance")
    c1, c2 = st.columns(2)
    with c1:
        user_inputs["mileage"] = st.number_input(
            "Mileage (km)", 
            min_value=0, 
            max_value=1000000, 
            value=None, 
            help="Leave blank to use typical value for this car"
        )
        user_inputs["annual_insurance"] = st.number_input(
            "Annual Insurance (KES)", 
            min_value=1000, 
            max_value=5000000, 
            value=None,
            help="Major price driver — override for accurate prediction!"
        )
        user_inputs["torque"] = st.number_input(
            "Torque (Nm)", 
            min_value=50, 
            max_value=1000, 
            value=None
        )
    with c2:
        user_inputs["engine_size"] = st.number_input(
            "Engine Size (cc)", 
            min_value=500, 
            max_value=8000, 
            value=None
        )
        user_inputs["horse_power"] = st.number_input(
            "Horse Power (hp)", 
            min_value=30, 
            max_value=2000, 
            value=None
        )
        user_inputs["acceleration"] = st.number_input(
            "Acceleration (0-100 km/h in seconds)", 
            min_value=2.0, 
            max_value=50.0, 
            value=None,
            step=0.1
        )

with st.expander(" Advanced: Categorical Features (Optional)"):
    st.markdown("### 🏷️ Specifications")
    c1, c2 = st.columns(2)
    with c1:
        user_inputs["body_type"] = st.selectbox(
            "Body Type", 
            ["Auto-detect", "SUV", "Sedan", "Hatchback", "Pickup", "Van", "Coupe", "Convertible"],
            help="Auto-detect uses typical value for this model"
        )
        if user_inputs["body_type"] == "Auto-detect":
            user_inputs["body_type"] = None

        user_inputs["drive"] = st.selectbox(
            "Drive Type", 
            ["Auto-detect", "2WD", "4WD", "AWD", "FWD", "RWD"],
            help="Auto-detect uses typical value"
        )
        if user_inputs["drive"] == "Auto-detect":
            user_inputs["drive"] = None

        user_inputs["fuel_type"] = st.selectbox(
            "Fuel Type", 
            ["Auto-detect", "Petrol", "Diesel", "Hybrid", "Electric"],
            help="Auto-detect uses typical value"
        )
        if user_inputs["fuel_type"] == "Auto-detect":
            user_inputs["fuel_type"] = None

    with c2:
        user_inputs["transmission"] = st.selectbox(
            "Transmission", 
            ["Auto-detect", "Automatic", "Manual", "Semi-Auto"],
            help="Auto-detect uses typical value"
        )
        if user_inputs["transmission"] == "Auto-detect":
            user_inputs["transmission"] = None

        user_inputs["usage_type"] = st.selectbox(
            "Usage Type", 
            ["Auto-detect", "Kenyan Used", "Foreign Used", "Brand New"],
            help="Auto-detect uses typical value"
        )
        if user_inputs["usage_type"] == "Auto-detect":
            user_inputs["usage_type"] = None

        user_inputs["condition"] = st.selectbox(
            "Condition", 
            ["Auto-detect", "Excellent", "Very Good", "Good", "Fair"],
            help="Auto-detect uses typical value"
        )
        if user_inputs["condition"] == "Auto-detect":
            user_inputs["condition"] = None

# Action buttons
b1, b2 = st.columns(2)
predict_clicked = b1.button(" Predict Price", type="primary")
preview_clicked = b2.button(" Preview Assumptions")

priors = artifacts["priors"]

if preview_clicked:
    prior_row, source_desc, r_make, r_model = resolve_priors_with_year(make_input, model_input, year_input, priors)
    st.info(f"**Data Source:** {source_desc}")
    
    # Show what would be used if no user overrides
    features_preview = create_feature_dict(r_make, r_model, year_input, prior_row)
    
    pretty = {
        "Make": r_make,
        "Model": r_model,
        "Year": int(year_input),
        "Mileage (km)": f"{features_preview.get('mileage', 0):,.0f}",
        "Annual Insurance (KES)": f"{features_preview.get('annual_insurance', 0):,.0f}",
        "Body Type": features_preview.get("body_type", "N/A"),
        "Drive": features_preview.get("drive", "N/A"),
        "Fuel": features_preview.get("fuel_type", "N/A"),
        "Transmission": features_preview.get("transmission", "N/A"),
        "Engine Size (cc)": f"{features_preview.get('engine_size', 0):,.0f}",
        "Horse Power (hp)": f"{features_preview.get('horse_power', 0):,.0f}",
        "Acceleration (0-100)": features_preview.get("acceleration", "N/A"),
        "Torque (Nm)": f"{features_preview.get('torque', 0):,.0f}",
        "Condition": features_preview.get("condition", "N/A"),
        "Usage Type": features_preview.get("usage_type", "N/A"),
    }
    perf_df = pd.DataFrame([pretty]).T.rename(columns={0: "Value"})
    st.table(sanitize_for_display(perf_df))

if predict_clicked:
    with st.spinner("Analyzing market data…"):
        prior_row, source_desc, r_make, r_model = resolve_priors_with_year(make_input, model_input, year_input, priors)
        features = create_feature_dict(r_make, r_model, year_input, prior_row, user_inputs)
        price, err = predict_price(features, artifacts)

    if err:
        st.error(f"Prediction failed: {err}")
    else:
        st.success(f"##  Predicted Price: KES {price:,.0f}")
        st.caption(source_desc)

        with st.expander(" Why did I get this price?"):
            st.markdown("""
            Your price is calculated using:
            -  **Your inputs** for insurance, mileage, power, etc. (if provided)
            -  **Year-aware market trends** for this make/model
            -  **ML model** trained on insurance-driven patterns, mileage, car_age, Usage_type (Kenyan or Foreign) + performance specs
            -  **Dynamic recalculation** — change any input to see real-time effect
            """)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Annual Insurance", f"KES {features['annual_insurance']:,.0f}")
        with c2:
            st.metric("Engine Size", f"{features['engine_size']:,.0f} cc")
        with c3:
            st.metric("Horse Power", f"{features['horse_power']:,.0f} hp")

        with st.expander(" Features used in this prediction"):
            show = features.copy()
            for k in ["name", "usage type"]:
                show.pop(k, None)
            df_show = pd.DataFrame([show]).T.rename(columns={0: "value"})
            st.dataframe(sanitize_for_display(df_show))

        if st.checkbox(" Debug: Sensitivity to insurance & mileage"):
            variants = []
            for scale in [0.5, 1.0, 2.0]:
                f = features.copy()
                f["annual_insurance"] = max(1_000, f["annual_insurance"] * scale)
                p, _ = predict_price(f, artifacts)
                variants.append(
                    {"Annual Insurance (KES)": f"{f['annual_insurance']:,.0f}",
                     "Predicted Price (KES)": f"{p:,.0f}" if p else "ERR"}
                )
            for scale in [0.5, 1.0, 2.0]:
                f = features.copy()
                f["mileage"] = min(max(0, f["mileage"] * scale), 1_000_000)
                p, _ = predict_price(f, artifacts)
                variants.append(
                    {"Mileage (km)": f"{f['mileage']:,.0f}",
                     "Predicted Price (KES)": f"{p:,.0f}" if p else "ERR"}
                )
            st.write("**Insurance sweeps:**")
            st.table(sanitize_for_display(pd.DataFrame(variants[:3])))
            st.write("**Mileage sweeps:**")
            st.table(sanitize_for_display(pd.DataFrame(variants[3:])))

with st.expander(" Inspect Priors Data (Debug)"):
    for lvl in ["level1_exact", "level2_year_bin", "level3_make_model", "level4_make_year", "level5_make", "level6_year"]:
        if lvl in priors:
            df = priors[lvl]
            if isinstance(df, pd.DataFrame) and not df.empty:
                st.write(f"### {lvl}")
                st.dataframe(sanitize_for_display(df.head(10)))
            else:
                st.caption(f"{lvl} is empty or not a DataFrame.")

st.markdown("---")
st.markdown("""
###  Quick Guide
-  Use **Preview Assumptions** to see default values
-  Override **any feature** in expanders for personalized results
-  Try **Sensitivity Analysis** to see how insurance/mileage and other features affect price (shows well the importance of these features in the model)
-  Use **Inspect Priors** only for debugging
""")

st.markdown("---")
st.caption(" Kenya Car Price Predictor —   ML model + Year-aware priors. Override any feature for personalized prediction!")