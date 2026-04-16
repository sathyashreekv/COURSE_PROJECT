import streamlit as st
import numpy as np
from PIL import Image
import torch
import pandas as pd
from dataclasses import dataclass
from tensorflow.keras.models import load_model
import torchvision.transforms as transforms
import random
from diffusers import StableDiffusionImg2ImgPipeline, AutoencoderKL

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="SEMEND — Hull Fouling AI", layout="wide", page_icon="🚢")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CLASS_NAMES = ["Clean", "Micro", "Medium", "Macro"]

SD_MODEL_ID = "./my_saved_sd_model"
SD_DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

FOULING_MAP = {
    0: {
        "folder": "0_clean",
        "prompt": (
            "close-up photograph of clean ship hull steel plate submerged underwater, "
            "smooth flat metal surface with antifouling red or grey paint, "
            "hull plate fills entire frame, no marine growth, no organisms, "
            "uniform surface texture, underwater photography"
        ),
        "strength": 0.30,
    },
    1: {
        "folder": "1_micro",
        "prompt": (
            "close-up photograph of ship hull steel plate underwater, "
            "thin greenish-brown biofilm and slime layer adhered directly onto metal surface, "
            "hull plate fills entire frame, faint microbial film coating the steel, "
            "early stage biofouling attached to hull, flat metal substrate visible beneath"
        ),
        "strength": 0.45,
    },
    2: {
        "folder": "2_medium",
        "prompt": (
            "close-up photograph of ship hull steel plate underwater, "
            "patchy barnacles and green algae physically attached and encrusted on metal surface, "
            "hull plate fills entire frame, steel substrate visible between fouling patches, "
            "medium density marine growth stuck to flat metal hull"
        ),
        "strength": 0.58,
    },
    3: {
        "folder": "3_macro",
        "prompt": (
            "close-up photograph of severely fouled ship hull steel plate underwater, "
            "dense barnacles mussels and tubeworms completely encrusted on metal surface, "
            "hull plate fills entire frame, heavy marine growth physically attached to steel, "
            "thick layer of calcareous organisms covering flat metal hull surface"
        ),
        "strength": 0.68,
    },
}

NEGATIVE_PROMPT = (
    "text, watermark, label, writing, numbers, caption, annotation, overlay, timestamp, "
    "open ocean, floating organisms, fish, swimming creatures, coral reef, sea floor, sand, "
    "water column, jellyfish, wide angle, aerial view, above water, dry dock, "
    "abstract, gradient, foggy, haze, glow, bokeh, blurry, "
    "painting, illustration, cartoon, render, 3d, cgi, art"
)

# ─────────────────────────────────────────────
# MAINTENANCE OPTIMIZER CONSTANTS
# ─────────────────────────────────────────────
COST = {
    "cleaning_scheduled": 15_000,
    "cleaning_emergency": 45_000,
    "fuel_penalty_per_day": {0: 0, 1: 150, 2: 600, 3: 2_500},
    "drydock_per_day": 8_000,
}

FOULING_GROWTH_RATE = {
    "cold":      {0: 180, 1: 300, 2: 400},   # temp < 10C
    "temperate": {0: 90,  1: 180, 2: 240},   # 10–20C
    "warm":      {0: 45,  1: 90,  2: 120},   # >20C
}

URGENCY_COLORS = {
    "NONE":     ("✅", "#28a745"),
    "LOW":      ("🟡", "#ffc107"),
    "MEDIUM":   ("🟠", "#fd7e14"),
    "HIGH":     ("🔴", "#dc3545"),
    "CRITICAL": ("🚨", "#9b0000"),
}

# ─────────────────────────────────────────────
# MAINTENANCE OPTIMIZER DATACLASS & FUNCTIONS
# ─────────────────────────────────────────────
@dataclass
class VesselState:
    vessel_id:          str
    current_class:      int
    days_since_drydock: float
    temperature_C:      float
    salinity_psu:       float
    region:             str
    speed_knots:        float
    wetted_surface_m2:  float

def temp_band(temp_c: float) -> str:
    if temp_c < 10:   return "cold"
    elif temp_c < 20: return "temperate"
    else:             return "warm"

def predict_days_to_next_class(state: VesselState) -> float:
    """Estimate days until fouling progresses one class."""
    if state.current_class >= 3:
        return float("inf")
    band  = temp_band(state.temperature_C)
    rates = FOULING_GROWTH_RATE[band]
    return rates.get(state.current_class, 200)

def fuel_cost_over_days(state: VesselState, days: int) -> float:
    """Estimate extra fuel cost from fouling drag over N days."""
    daily = COST["fuel_penalty_per_day"][min(state.current_class, 3)]
    return daily * days

def recommend_action(state: VesselState, planning_horizon_days: int = 90) -> dict:
    """Full rule-based maintenance decision engine."""
    days_to_next      = predict_days_to_next_class(state)
    current_fuel_cost = fuel_cost_over_days(state, planning_horizon_days)

    # Decision rules (priority order)
    if state.current_class == 3:
        action   = "IMMEDIATE CLEANING"
        urgency  = "CRITICAL"
        reason   = "Macro fouling — maximum drag penalty. Schedule dry dock immediately."
        est_cost = COST["cleaning_emergency"]

    elif state.current_class == 2 and days_to_next < 60:
        action   = "SCHEDULE CLEANING"
        urgency  = "HIGH"
        reason   = (
            f"Medium fouling in {temp_band(state.temperature_C)} water — "
            f"will reach macro fouling in approximately {days_to_next:.0f} days."
        )
        est_cost = COST["cleaning_scheduled"]

    elif state.current_class == 2:
        action   = "MONITOR CLOSELY"
        urgency  = "MEDIUM"
        reason   = (
            f"Medium fouling detected — progression to macro expected in "
            f"approximately {days_to_next:.0f} days. Plan ahead."
        )
        est_cost = current_fuel_cost

    elif state.current_class == 1 and state.days_since_drydock > 700:
        action   = "PLAN INSPECTION"
        urgency  = "LOW"
        reason   = (
            f"Micro fouling with hull aged {state.days_since_drydock:.0f} days "
            f"since drydock — schedule next inspection."
        )
        est_cost = current_fuel_cost

    else:
        action   = "NO ACTION NEEDED"
        urgency  = "NONE"
        reason   = "Fouling is within acceptable limits. Continue regular monitoring."
        est_cost = 0

    # Fuel savings if cleaned now vs staying fouled
    clean_state      = VesselState(**{**state.__dict__, "current_class": 0})
    post_clean_fuel  = fuel_cost_over_days(clean_state, planning_horizon_days)
    fuel_saving      = current_fuel_cost - post_clean_fuel
    net_benefit      = fuel_saving - est_cost

    return {
        "action":               action,
        "urgency":              urgency,
        "reason":               reason,
        "estimated_cost_usd":   est_cost,
        "fuel_saving_usd":      round(fuel_saving, 2),
        "days_to_next_class":   round(days_to_next, 0) if days_to_next != float("inf") else "—",
        "net_benefit_usd":      round(net_benefit, 2),
        "current_fuel_cost":    round(current_fuel_cost, 2),
        "temp_band":            temp_band(state.temperature_C),
    }

# ─────────────────────────────────────────────
# HELPER: build fouling progression timeline
# ─────────────────────────────────────────────
def build_progression_timeline(state: VesselState) -> pd.DataFrame:
    band  = temp_band(state.temperature_C)
    rates = FOULING_GROWTH_RATE[band]
    rows  = []
    cumulative = 0
    for cls in range(4):
        rows.append({"Fouling Class": CLASS_NAMES[cls], "Starts at Day": cumulative})
        if cls < 3:
            cumulative += rates.get(cls, 200)
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource
def load_cnn():
    return load_model("models/final_cnn_model.keras")

@st.cache_resource
def load_sd_pipeline():
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float16 if SD_DEVICE == "cuda" else torch.float32,
    )
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        SD_MODEL_ID,
        vae=vae,
        torch_dtype=torch.float16 if SD_DEVICE == "cuda" else torch.float32,
        safety_checker=None,
    ).to(SD_DEVICE)
    pipe.enable_attention_slicing()
    return pipe

cnn_model   = load_cnn()
sd_pipeline = load_sd_pipeline()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# ─────────────────────────────────────────────
# SD HELPER FUNCTIONS
# ─────────────────────────────────────────────
def temperature_cue(temp_c):
    if temp_c < 0:    return "sub-zero Arctic seawater, polar conditions"
    elif temp_c < 5:  return "near-freezing cold Arctic water"
    elif temp_c < 10: return "cold Great Lakes or North Atlantic water"
    elif temp_c < 16: return "cool temperate coastal water"
    elif temp_c < 22: return "warm temperate Atlantic or Pacific coast"
    else:             return "warm subtropical coastal seawater"

def salinity_cue(sal, env="marine"):
    if env == "fresh":  return "freshwater Great Lakes environment"
    elif sal < 32:      return "low salinity coastal water"
    elif sal < 34.5:    return "moderate salinity open ocean"
    else:               return "high salinity open ocean"

def region_cue(region_code):
    return {
        "gl": "Great Lakes Canada",
        "ec": "East Canadian Atlantic coast",
        "wc": "West Canadian Pacific coast",
        "ar": "Canadian Arctic polar ocean",
    }.get(region_code, "Canadian coastal waters")

def build_prompt(predicted_label, temp_val, salinity_val, region_val):
    base = FOULING_MAP[predicted_label]["prompt"]
    env_context = (
        f"{temperature_cue(temp_val)}, "
        f"{salinity_cue(salinity_val, 'fresh' if salinity_val < 10 else 'marine')}, "
        f"{region_cue(region_val)}"
    )
    return f"{base}, {env_context}", NEGATIVE_PROMPT

def prepare_init_image_for_sd(img_pil):
    img = img_pil.convert("RGB")
    w, h = img.size
    img = img.crop((int(w*0.10), int(h*0.08), int(w*0.90), int(h*0.92)))
    return img.resize((512, 512), Image.LANCZOS)

# ─────────────────────────────────────────────
# PREDICTION FUNCTIONS
# ─────────────────────────────────────────────
def predict_image(img):
    img_t = transform(img).unsqueeze(0)
    img_t = np.transpose(img_t.numpy(), (0, 2, 3, 1))
    preds     = cnn_model.predict(img_t)
    class_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    return class_idx, confidence, preds

# ─────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────
st.title("🚢 SEMEND — AI-Driven Hull Biofouling System")
st.markdown(
    "**Generative · Predictive · Optimization Pipeline** — "
    "Upload a hull image and set environmental conditions to classify fouling "
    "and get a full maintenance decision."
)
st.divider()

# ─────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────
col_img, col_env = st.columns([1, 1], gap="large")

with col_img:
    st.subheader("📸 Hull Image")
    uploaded_file = st.file_uploader("Upload hull image", type=["jpg", "png"])
    image = None
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded hull image", use_column_width=True)

with col_env:
    st.subheader("🌊 Environmental Conditions")
    temp     = st.slider("Temperature (°C)",      -2.0, 30.0, 15.0, step=0.5)
    salinity = st.slider("Salinity (PSU)",         30.0, 40.0, 34.0, step=0.1)
    days     = st.slider("Days since drydock",      0,   1500,  300, step=10)
    speed    = st.slider("Vessel speed (knots)",   5.0,  25.0,  14.0, step=0.5)
    surface  = st.slider("Wetted surface (m²)",  1000, 20000, 8000,  step=100)
    region   = st.selectbox(
        "Region",
        ["gl", "ec", "wc", "ar"],
        format_func=lambda x: {
            "gl": "Great Lakes (freshwater)",
            "ec": "East Canada (Atlantic)",
            "wc": "West Canada (Pacific)",
            "ar": "Arctic (polar)",
        }[x],
    )

st.divider()

# ─────────────────────────────────────────────
# RUN ANALYSIS BUTTON
# ─────────────────────────────────────────────
run_analysis = st.button("🔍 Run Full Analysis", type="primary", use_container_width=True)

if run_analysis:
    if image is None:
        st.error("Please upload a hull image before running analysis.")
        st.stop()

    # ── CNN Prediction ────────────────────────
    with st.spinner("Running CNN classification..."):
        cls, conf, probs = predict_image(image)

    # Store in session state for SD generation later
    st.session_state["final_pred"] = cls
    st.session_state["conf"]       = conf
    st.session_state["probs"]      = probs

    # ── Classification Results ────────────────
    st.subheader("📊 Classification Results")
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Fouling Class", CLASS_NAMES[cls])
    m2.metric("Model Confidence",        f"{conf:.1%}")
    m3.metric("Temperature Band",        temp_band(temp).capitalize())

    # Probability bar chart
    prob_df = pd.DataFrame({
        "Class":       CLASS_NAMES,
        "Probability": [float(p) for p in probs[0]],
    })
    st.bar_chart(prob_df.set_index("Class"), color="#1D9E75")

    st.divider()

    # ─────────────────────────────────────────────
    # MAINTENANCE SECTION — full optimizer engine
    # ─────────────────────────────────────────────
    st.subheader("🛠️ Maintenance Recommendation")

    # Build VesselState from current inputs
    vessel = VesselState(
        vessel_id          = "UPLOADED_VESSEL",
        current_class      = cls,
        days_since_drydock = float(days),
        temperature_C      = float(temp),
        salinity_psu       = float(salinity),
        region             = region,
        speed_knots        = float(speed),
        wetted_surface_m2  = float(surface),
    )

    rec = recommend_action(vessel, planning_horizon_days=90)

    urgency_icon, urgency_color = URGENCY_COLORS[rec["urgency"]]

    # ── Urgency + Action banner ───────────────
    st.markdown(
        f"""
        <div style="
            background-color:{urgency_color}22;
            border-left: 5px solid {urgency_color};
            border-radius: 8px;
            padding: 1rem 1.25rem;
            margin-bottom: 1rem;
        ">
            <div style="font-size:22px; font-weight:700; color:{urgency_color};">
                {urgency_icon} {rec['action']}
            </div>
            <div style="font-size:15px; color:#ccc; margin-top:6px;">
                {rec['reason']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Cost & savings metrics ────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Urgency Level",
        rec["urgency"],
    )
    c2.metric(
        "Estimated cleaning cost",
        f"${rec['estimated_cost_usd']:,}" if rec["estimated_cost_usd"] > 0 else "—",
    )
    c3.metric(
        "Fuel saving (90 days)",
        f"${rec['fuel_saving_usd']:,.0f}",
        delta="vs staying fouled",
        delta_color="normal",
    )
    c4.metric(
        "Net benefit",
        f"${rec['net_benefit_usd']:,.0f}",
        delta="clean now vs wait",
        delta_color="normal" if rec["net_benefit_usd"] >= 0 else "inverse",
    )

    # ── Fuel drag breakdown ───────────────────
    st.markdown("#### Fuel drag penalty breakdown")
    fuel_df = pd.DataFrame({
        "Fouling class": CLASS_NAMES,
        "Extra fuel cost / day (USD)": [
            COST["fuel_penalty_per_day"][i] for i in range(4)
        ],
    })
    # Highlight current class
    st.dataframe(
        fuel_df.style.apply(
            lambda row: [
                "background-color:#1e3a1e" if row["Fouling class"] == CLASS_NAMES[cls]
                else ""
                for _ in row
            ],
            axis=1,
        ),
        use_container_width=True,
        hide_index=True,
    )

    # ── Fouling progression timeline ──────────
    st.markdown("#### Fouling progression timeline (current conditions)")
    timeline_df = build_progression_timeline(vessel)

    prog_chart = {}
    for _, row in timeline_df.iterrows():
        prog_chart[row["Fouling Class"]] = row["Starts at Day"]

    st.markdown(
        f"In **{rec['temp_band']}** water conditions ({temp}°C), fouling progresses as:"
    )

    bar_data = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Days to reach this class": [
            timeline_df.loc[i, "Starts at Day"] for i in range(4)
        ],
    })
    st.bar_chart(bar_data.set_index("Class"))

    days_label = (
        f"~{int(rec['days_to_next_class'])} days"
        if rec["days_to_next_class"] != "—"
        else "Already at maximum"
    )
    st.info(
        f"⏱️ At current fouling class **{CLASS_NAMES[cls]}** — "
        f"estimated time to next class: **{days_label}**"
    )

    # ── 90-day cost projection ────────────────
    st.markdown("#### 90-day cost projection")
    proj_cols = st.columns(2)
    with proj_cols[0]:
        st.markdown("**If no action taken:**")
        st.markdown(f"- Current fuel penalty: **${rec['current_fuel_cost']:,.0f}**")
        st.markdown(f"- Fouling class: **{CLASS_NAMES[cls]}**")
        st.markdown(
            f"- Risk of escalation: "
            f"**{'High' if rec['days_to_next_class'] != '—' and rec['days_to_next_class'] < 90 else 'Low'}**"
        )
    with proj_cols[1]:
        st.markdown("**If cleaned now:**")
        st.markdown(f"- Cleaning cost: **${rec['estimated_cost_usd']:,}**")
        st.markdown(f"- Fuel saving over 90 days: **${rec['fuel_saving_usd']:,.0f}**")
        net = rec["net_benefit_usd"]
        color = "green" if net >= 0 else "red"
        st.markdown(f"- Net benefit: **:{color}[${net:,.0f}]**")

    st.success("✅ Analysis complete")

st.divider()

# ─────────────────────────────────────────────
# STABLE DIFFUSION GENERATION SECTION
# ─────────────────────────────────────────────
st.subheader("✨ Generate Simulated Fouling Image")
st.caption(
    "Uses Stable Diffusion img2img conditioned on the predicted fouling class "
    "and environmental parameters to synthesise a realistic hull image."
)

if st.button("🖼️ Generate Simulated Fouling Image", use_container_width=True):
    if image is None:
        st.error("Please upload an image first.")
        st.stop()

    if "final_pred" not in st.session_state:
        st.warning("Run the analysis first so the predicted class is available.")
        st.stop()

    predicted_label = st.session_state["final_pred"]

    with st.spinner("Generating with Stable Diffusion — this may take a minute..."):
        init_image_sd      = prepare_init_image_for_sd(image)
        prompt, neg_prompt = build_prompt(predicted_label, temp, salinity, region)
        strength_val       = FOULING_MAP[predicted_label]["strength"]

        output = sd_pipeline(
            prompt              = prompt,
            negative_prompt     = neg_prompt,
            image               = init_image_sd,
            strength            = strength_val,
            guidance_scale      = 8.5,
            num_inference_steps = 45,
            generator           = torch.Generator(SD_DEVICE).manual_seed(
                                    random.randint(0, 99999)
                                  ),
        )
        generated_image = output.images[0]

    gen_col1, gen_col2 = st.columns(2)
    with gen_col1:
        st.image(image,           caption="Original uploaded image", use_column_width=True)
    with gen_col2:
        st.image(generated_image, caption=f"Simulated — {CLASS_NAMES[predicted_label]} fouling", use_column_width=True)

    st.success("Image generation complete!")