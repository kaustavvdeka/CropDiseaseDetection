"""
CropGuard NE — Streamlit Web Application
=========================================
Run:  streamlit run app/app.py
"""

import os, sys, json, time
import numpy as np
import streamlit as st
from PIL import Image

# Allow importing from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CropGuard NE — Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #1a1a1a; }

  /* Hero gradient */
  .hero-banner {
    background: linear-gradient(135deg, #0F6E56 0%, #1D9E75 60%, #5DCAA5 100%);
    color: #1a1a1a;
    padding: 3rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    text-align: center;
  }
  .hero-banner h1 { 
    font-size: 2.6rem; 
    font-weight: 800; 
    margin: 0.5rem 0; 
    color: #0F6E56;
  }
  .hero-banner p { 
    font-size: 1.1rem; 
    max-width: 560px; 
    margin: 0 auto; 
    color: #1a1a1a;
  }
  .hero-badge {
    display: inline-block;
    background: rgba(15, 110, 86, 0.1);
    border: 1px solid rgba(15, 110, 86, 0.3);
    border-radius: 20px;
    padding: 4px 16px;
    font-size: 0.8rem;
    margin-bottom: 1rem;
    color: #0F6E56;
  }

  /* Stat cards */
  .stat-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
  }
  .stat-val { 
    font-size: 2rem; 
    font-weight: 800; 
    color: #0F6E56; 
  }
  .stat-lbl { 
    font-size: 0.75rem; 
    color: #374151;
    margin-top: 2px; 
  }

  /* Feature cards */
  .feature-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 1.4rem;
    height: 100%;
    color: #1a1a1a;
  }
  .feature-card h4 { 
    margin: 0.6rem 0 0.4rem; 
    font-size: 1rem; 
    font-weight: 600; 
    color: #1a1a1a;
  }
  .feature-card p { 
    font-size: 0.85rem; 
    color: #374151;
    margin: 0; 
    line-height: 1.6; 
  }

  /* Result cards */
  .result-header {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 1px solid #86efac;
    border-radius: 14px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    color: #1a1a1a;
  }
  .result-diseased-header {
    background: linear-gradient(135deg, #fff7ed, #ffedd5);
    border: 1px solid #fdba74;
    border-radius: 14px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    color: #1a1a1a;
  }
  .disease-name { 
    font-size: 1.4rem; 
    font-weight: 700; 
    margin: 0;
    color: #1a1a1a;
  }
  .crop-name { 
    font-size: 0.85rem; 
    color: #374151;
    margin: 2px 0 0; 
  }

  .badge-healthy { 
    background: #dcfce7; 
    color: #14532d; 
    border-radius: 12px; 
    padding: 3px 12px; 
    font-size: 0.78rem; 
    font-weight: 600; 
  }
  .badge-diseased { 
    background: #fee2e2; 
    color: #7f1d1d; 
    border-radius: 12px; 
    padding: 3px 12px; 
    font-size: 0.78rem; 
    font-weight: 600; 
  }

  .severity-low { 
    background: #dcfce7; 
    color: #14532d; 
    border: 1px solid #86efac; 
  }
  .severity-moderate { 
    background: #fef3c7; 
    color: #78350f; 
    border: 1px solid #fcd34d; 
  }
  .severity-high { 
    background: #ffedd5; 
    color: #7c2d12; 
    border: 1px solid #fb923c; 
  }
  .severity-critical { 
    background: #fee2e2; 
    color: #7f1d1d; 
    border: 1px solid #f87171; 
  }
  .severity-badge { 
    display: inline-block; 
    border-radius: 8px; 
    padding: 4px 14px; 
    font-size: 0.8rem; 
    font-weight: 600; 
  }

  .section-title { 
    font-size: 0.7rem; 
    font-weight: 600; 
    text-transform: uppercase; 
    letter-spacing: 0.8px; 
    color: #4b5563;
    margin: 1rem 0 0.4rem; 
  }

  .symptom-item { 
    display: flex; 
    gap: 8px; 
    margin-bottom: 6px; 
    font-size: 0.88rem; 
    line-height: 1.5; 
    color: #1a1a1a;
  }
  .bullet-red { 
    color: #ef4444; 
    flex-shrink: 0; 
  }
  .bullet-green { 
    color: #22c55e; 
    flex-shrink: 0; 
  }

  .treatment-item { 
    background: #fff7ed; 
    border-left: 3px solid #f97316; 
    border-radius: 0 8px 8px 0; 
    padding: 8px 12px; 
    margin-bottom: 6px; 
    font-size: 0.85rem; 
    color: #1a1a1a;
  }
  .organic-item { 
    background: #f0fdf4; 
    border-left: 3px solid #22c55e; 
    border-radius: 0 8px 8px 0; 
    padding: 8px 12px; 
    margin-bottom: 6px; 
    font-size: 0.85rem; 
    color: #1a1a1a;
  }

  .ne-context-box { 
    background: #ecfdf5; 
    border: 1px solid #6ee7b7; 
    border-radius: 12px; 
    padding: 1rem 1.2rem; 
    color: #1a1a1a;
  }
  .ne-context-box .ne-title { 
    font-size: 0.78rem; 
    font-weight: 700; 
    color: #065f46; 
    margin-bottom: 6px; 
  }
  .ne-context-box p { 
    font-size: 0.86rem; 
    color: #064e3b; 
    margin: 0; 
    line-height: 1.65; 
  }

  .conf-bar-bg { 
    background: #e5e7eb; 
    border-radius: 4px; 
    height: 8px; 
    overflow: hidden; 
    margin-top: 4px; 
  }
  .conf-bar-fill { 
    height: 100%; 
    border-radius: 4px; 
    transition: width 0.6s ease; 
  }

  .top-k-row { 
    display: flex; 
    justify-content: space-between; 
    align-items: center; 
    font-size: 0.8rem; 
    padding: 4px 0; 
    border-bottom: 1px solid #f3f4f6; 
    color: #374151;
  }

  /* Sidebar */
  .sidebar-info { 
    background: #f0fdf4; 
    border: 1px solid #a7f3d0; 
    border-radius: 10px; 
    padding: 1rem; 
    font-size: 0.83rem; 
    color: #064e3b; 
  }

  /* Tech pills */
  .tech-pill { 
    display: inline-block; 
    background: #f3f4f6; 
    border: 1px solid #d1d5db; 
    border-radius: 20px; 
    padding: 4px 12px; 
    font-size: 0.75rem; 
    margin: 3px;
    color: #374151;
  }

  /* Step card */
  .step-card { 
    text-align: center; 
    padding: 1rem; 
    color: #1a1a1a;
  }
  .step-num { 
    width: 36px; 
    height: 36px; 
    border-radius: 50%; 
    background: #dcfce7; 
    color: #0F6E56; 
    font-weight: 700; 
    font-size: 1rem; 
    display: flex; 
    align-items: center; 
    justify-content: center; 
    margin: 0 auto 8px; 
  }

  /* Disease class pill */
  .class-pill-healthy { 
    background: #dcfce7; 
    color: #14532d; 
    border: 1px solid #86efac; 
  }
  .class-pill-diseased { 
    background: #f9fafb; 
    color: #374151; 
    border: 1px solid #e5e7eb; 
  }
  .class-pill { 
    display: inline-block; 
    border-radius: 6px; 
    padding: 3px 10px; 
    font-size: 0.7rem; 
    margin: 2px; 
  }
</style>
""", unsafe_allow_html=True)


# ─── LOAD MODEL ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "models/plantdisease_mobilenetv2.h5"
    class_path = "models/class_indices.json"

    if os.path.exists(model_path) and os.path.exists(class_path):
        try:
            from model.predictor import PlantDiseasePredictor
            return PlantDiseasePredictor(model_path, class_path)
        except Exception as e:
            st.warning("⚠️ TensorFlow model not available")
            st.info(f"Reason: {e}")
            return None

    return None

predictor = load_model()


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌿 CropGuard NE")
    st.markdown("**Plant Disease Detection**")
    st.markdown("*Northeast India Focus*")
    st.divider()

    page = st.radio("Navigate", ["🏠 Home", "🔍 Disease Detection", "📊 About"], label_visibility="collapsed")

    st.divider()
    if predictor:
        st.success("✅ ML Model Loaded")
    else:
        st.warning("⚠️ Model not found\nRun `python model/train.py` first")
        st.info("The app will use Claude AI as fallback while the model trains.")

    st.divider()
    st.markdown('<div class="sidebar-info">📍 <b>Coverage:</b> Assam, Meghalaya, Manipur, Nagaland, Arunachal Pradesh, Tripura, Mizoram, Sikkim</div>', unsafe_allow_html=True)
    st.caption("Dataset: PlantVillage • ~87K images • 38 classes")


# ─── HOME PAGE ───────────────────────────────────────────────────────────────
if "Home" in page:

    st.markdown("""
    <div class="hero-banner">
      <div class="hero-badge">🛰️ CNN + MobileNetV2 • Northeast India</div>
      <h1>Detect Crop Diseases<br>Before They Spread</h1>
      <p>AI-powered plant disease detection trained on 87,000 images. Built for farmers across Northeast India.</p>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    c1, c2, c3, c4 = st.columns(4)
    for col, (val, lbl) in zip([c1,c2,c3,c4], [("87K+","Images"), ("38","Disease Classes"), ("96%","Accuracy"), ("15+","Crop Types")]):
        col.markdown(f'<div class="stat-card"><div class="stat-val">{val}</div><div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Features
    st.subheader("Why CropGuard NE?")
    f1, f2, f3, f4 = st.columns(4)
    feats = [
        ("🔬","Early Detection","Identify diseases before visible spread, saving up to 40% of crop loss."),
        ("🗺️","NE India Context","Tailored recommendations for NE India's unique climate and farming practices."),
        ("🌿","Organic Options","Chemical and organic treatments, including locally-available remedies."),
        ("⚡","Instant Results","Upload a photo, get a full disease report with confidence score instantly."),
    ]
    for col, (icon, title, desc) in zip([f1,f2,f3,f4], feats):
        col.markdown(f'<div class="feature-card"><div style="font-size:2rem">{icon}</div><h4>{title}</h4><p>{desc}</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    # How to use
    st.subheader("How to Use")
    s1, s2, s3, s4 = st.columns(4)
    steps = [("1","📸","Upload Image","Take a clear photo of the affected leaf"),
             ("2","🤖","AI Analysis","CNN model + Claude AI analyzes symptoms"),
             ("3","📊","Get Results","Disease name, severity, confidence score"),
             ("4","💊","Take Action","Follow treatment & prevention advice")]
    for col, (n, icon, t, d) in zip([s1,s2,s3,s4], steps):
        col.markdown(f'<div class="step-card"><div class="step-num">{n}</div><div style="font-size:1.8rem">{icon}</div><b style="font-size:0.9rem">{t}</b><p style="font-size:0.8rem;color:#6b7280;margin-top:4px">{d}</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Tech stack
    st.subheader("Tech Stack")
    for t in ["MobileNetV2","TensorFlow 2.x","Keras","OpenCV","NumPy","Python 3.10+","Streamlit","Claude AI","Pillow","scikit-learn"]:
        st.markdown(f'<span class="tech-pill">{t}</span>', unsafe_allow_html=True)


# ─── DETECTION PAGE ──────────────────────────────────────────────────────────
elif "Detection" in page:

    st.title("🔍 Plant Disease Detection")
    st.caption("Upload a leaf image or select a sample to analyze")

    col_upload, col_result = st.columns([1, 1.1], gap="large")

    with col_upload:
        st.subheader("📷 Upload Image")
        uploaded = st.file_uploader("Choose a leaf image", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")

        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption="Uploaded image", use_container_width=True)

        st.divider()
        st.subheader("⚡ Quick Samples")
        sample_cols = st.columns(2)
        SAMPLES = [
            ("🥔","Potato Late Blight"),("🍅","Tomato Bacterial Spot"),
            ("🌾","Rice Blast"),("🍃","Tomato Early Blight"),
            ("🌿","Potato Early Blight"),("✅","Healthy Leaf"),
        ]
        if "selected_sample" not in st.session_state:
            st.session_state.selected_sample = None

        for i, (emoji, name) in enumerate(SAMPLES):
            col = sample_cols[i % 2]
            if col.button(f"{emoji} {name}", use_container_width=True, key=f"sample_{i}"):
                st.session_state.selected_sample = name
                uploaded = None

        if st.session_state.selected_sample and not uploaded:
            st.info(f"Sample selected: **{st.session_state.selected_sample}**")

        st.divider()
        analyze_btn = st.button("🔍 Analyze Disease", type="primary", use_container_width=True,
                                 disabled=not (uploaded or st.session_state.selected_sample))

    with col_result:
        st.subheader("📋 Analysis Results")

        if not (uploaded or st.session_state.selected_sample):
            st.markdown("""
            <div style="text-align:center;padding:3rem 1rem;border:2px dashed #d1d5db;border-radius:16px;color:#9ca3af">
              <div style="font-size:3rem">🌱</div>
              <b style="font-size:1rem">Ready for Analysis</b>
              <p style="font-size:0.85rem">Upload a leaf image or select a sample</p>
            </div>
            """, unsafe_allow_html=True)

        elif analyze_btn:
            with st.spinner("Analyzing plant... 🔄"):
                time.sleep(0.5)   # let spinner render

                result = None

                # ── A) Real model ──────────────────────────────────────────
                if predictor and uploaded:
                    image = Image.open(uploaded)
                    result = predictor.predict(image)

                # ── B) Claude AI fallback ──────────────────────────────────
                else:
                    import anthropic
                    client = anthropic.Anthropic()

                    if uploaded:
                        import base64, io
                        image = Image.open(uploaded)
                        buf = io.BytesIO()
                        image.save(buf, format="JPEG")
                        b64 = base64.b64encode(buf.getvalue()).decode()

                        msg = client.messages.create(
                            model="claude-opus-4-5",
                            max_tokens=1024,
                            messages=[{
                                "role": "user",
                                "content": [
                                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                                    {"type": "text", "text": (
                                        "You are an expert plant pathologist for Northeast India crops. "
                                        "Analyze this leaf image and respond ONLY with a JSON object, no markdown:\n"
                                        '{"disease_name":"...","crop":"...","confidence":90,"status":"Diseased or Healthy",'
                                        '"severity":"Low/Moderate/High/Critical","cause":"...","symptoms":["..."],'
                                        '"northeast_context":"...","treatment":["..."],"organic_options":["..."],"prevention":["..."]}'
                                    )}
                                ]
                            }]
                        )
                    else:
                        sample_name = st.session_state.selected_sample
                        msg = client.messages.create(
                            model="claude-opus-4-5",
                            max_tokens=1024,
                            messages=[{
                                "role": "user",
                                "content": (
                                    f'Expert plant pathologist for NE India. Sample: "{sample_name}". '
                                    "Return ONLY JSON, no markdown:\n"
                                    '{"disease_name":"...","crop":"...","confidence":92,"status":"Diseased or Healthy",'
                                    '"severity":"Low/Moderate/High/Critical","cause":"...","symptoms":["..."],'
                                    '"northeast_context":"...","treatment":["..."],"organic_options":["..."],"prevention":["..."]}'
                                )
                            }]
                        )

                    raw = msg.content[0].text
                    clean = raw.replace("```json","").replace("```","").strip()
                    data = json.loads(clean)

                    result = {
                        "disease":          data.get("disease_name", "Unknown"),
                        "crop":             data.get("crop", "Unknown"),
                        "is_healthy":       data.get("status","").lower() == "healthy",
                        "confidence":       float(data.get("confidence", 85)),
                        "status":           data.get("status", "Diseased"),
                        "severity":         data.get("severity", "Moderate"),
                        "cause":            data.get("cause", ""),
                        "symptoms":         data.get("symptoms", []),
                        "northeast_context": data.get("northeast_context", ""),
                        "treatment":        data.get("treatment", []),
                        "organic_options":  data.get("organic_options", []),
                        "prevention":       data.get("prevention", []),
                        "top_predictions":  [],
                    }

            # ── RENDER RESULTS ────────────────────────────────────────────
            if result:
                is_h = result["is_healthy"]
                hdr_class = "result-header" if is_h else "result-diseased-header"
                badge = '<span class="badge-healthy">✓ Healthy</span>' if is_h else '<span class="badge-diseased">⚠ Diseased</span>'

                st.markdown(f"""
                <div class="{hdr_class}">
                  <p class="disease-name">{result['disease']}</p>
                  <p class="crop-name">Crop: {result['crop']} &nbsp;|&nbsp; {badge}</p>
                </div>
                """, unsafe_allow_html=True)

                # Confidence bar
                conf = result["confidence"]
                bar_color = "#1D9E75" if conf >= 80 else "#f97316" if conf >= 60 else "#ef4444"
                st.markdown(f"""
                <div style="margin-bottom:1rem">
                  <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#6b7280">
                    <span>Confidence</span><span>{conf:.1f}%</span>
                  </div>
                  <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{conf}%;background:{bar_color}"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Severity
                if result["severity"] not in ("None", ""):
                    sev = result["severity"]
                    sev_cls = f"severity-{sev.lower()}"
                    st.markdown(f'<span class="severity-badge {sev_cls}">Severity: {sev}</span>', unsafe_allow_html=True)
                    st.markdown("")

                # Cause
                if result["cause"]:
                    st.markdown(f'<div class="section-title">Cause</div><div style="font-size:0.87rem;padding-left:4px">{result["cause"]}</div>', unsafe_allow_html=True)

                # Symptoms
                if result["symptoms"]:
                    st.markdown('<div class="section-title">Symptoms</div>', unsafe_allow_html=True)
                    for s in result["symptoms"]:
                        st.markdown(f'<div class="symptom-item"><span class="bullet-red">●</span>{s}</div>', unsafe_allow_html=True)

                # NE Context
                if result["northeast_context"]:
                    st.markdown(f"""
                    <div class="ne-context-box" style="margin:0.8rem 0">
                      <div class="ne-title">🗺️ Northeast India Context</div>
                      <p>{result['northeast_context']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Treatment
                if result["treatment"]:
                    st.markdown('<div class="section-title">Treatment</div>', unsafe_allow_html=True)
                    for t in result["treatment"]:
                        st.markdown(f'<div class="treatment-item">💊 {t}</div>', unsafe_allow_html=True)

                # Organic
                if result["organic_options"]:
                    st.markdown('<div class="section-title">🌿 Organic Options</div>', unsafe_allow_html=True)
                    for o in result["organic_options"]:
                        st.markdown(f'<div class="organic-item">🌱 {o}</div>', unsafe_allow_html=True)

                # Prevention
                if result.get("prevention"):
                    st.markdown('<div class="section-title">Prevention</div>', unsafe_allow_html=True)
                    for p in result["prevention"]:
                        st.markdown(f'<div class="symptom-item"><span class="bullet-green">●</span>{p}</div>', unsafe_allow_html=True)

                # Top-K
                if result.get("top_predictions"):
                    with st.expander("📊 All top predictions"):
                        for r in result["top_predictions"]:
                            pct = r["confidence"]
                            bar = "#1D9E75" if pct > 60 else "#f97316"
                            st.markdown(f"""
                            <div class="top-k-row">
                              <span>{r['class'].replace('___',' → ').replace('_',' ')}</span>
                              <span style="font-weight:600;color:{bar}">{pct:.1f}%</span>
                            </div>
                            """, unsafe_allow_html=True)


# ─── ABOUT PAGE ──────────────────────────────────────────────────────────────
elif "About" in page:

    st.title("📊 About CropGuard NE")
    st.caption("Dataset, architecture, and technical details")

    tab1, tab2, tab3 = st.tabs(["📁 Dataset", "🧠 Architecture", "🌿 38 Classes"])

    with tab1:
        st.subheader("PlantVillage Dataset")
        st.markdown("""
        The [PlantVillage dataset](https://kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
        contains approximately **87,000 images** of healthy and diseased crop leaves across **38 categories**.
        """)

        c1, c2, c3 = st.columns(3)
        for col, (label, val, icon, color) in zip([c1,c2,c3], [
            ("Training Set",   "70,295", "🏋️", "#0F6E56"),
            ("Validation Set", "17,572", "✅", "#185FA5"),
            ("Test Set",       "33",     "🧪", "#854F0B"),
        ]):
            col.metric(label, val, icon)

        st.markdown("---")
        st.markdown("**Key Characteristics:**")
        st.markdown("""
        - 38 classes covering healthy and diseased states of 14+ crops
        - Images taken under controlled (white background) and field conditions
        - Crops relevant to NE India: Potato, Tomato, Corn, Apple, Cherry, Grape, Peach, Pepper, Raspberry, Soybean, Squash, Strawberry, Blueberry, Orange
        - RGB images of leaf scans, standardized to 256×256 (resized to 224×224 for training)
        """)

    with tab2:
        st.subheader("Model Architecture")

        arch_cards = [
            ("🧠", "MobileNetV2 Backbone", "Pre-trained on ImageNet (1.4M images, 1000 classes). Depthwise separable convolutions give 96%+ accuracy at 1/10th the parameters of VGG16. Last 50 layers fine-tuned on PlantVillage."),
            ("⚙️", "Preprocessing Pipeline", "Images resized to 224×224 RGB → normalized to [0, 1]. Training augmentation: random flips, 20° rotations, zoom ±20%, brightness ±20%, shear 10%."),
            ("🔬", "Classification Head", "GlobalAveragePooling2D → BatchNorm → Dense(512, ReLU, L2) → Dropout(0.4) → Dense(256, ReLU, L2) → Dropout(0.3) → Dense(38, Softmax)"),
            ("⚡", "Training Strategy", "Phase 1 (10 epochs): frozen base, train head only @ lr=1e-4. Phase 2 (15 epochs): unfreeze top 50 layers, fine-tune @ lr=1e-5. EarlyStopping + ReduceLROnPlateau."),
            ("📈", "Performance", "Val Accuracy: ~96% | Top-5 Accuracy: ~99.5% | Training time: ~2–3 hours on GPU (NVIDIA T4 or V100)."),
            ("✨", "Claude AI Layer", "When the .h5 model is unavailable, Claude API provides expert agronomic analysis with NE India-specific treatment recommendations."),
        ]
        cols = st.columns(2)
        for i, (icon, title, desc) in enumerate(arch_cards):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="feature-card" style="margin-bottom:1rem">
                  <div style="font-size:1.8rem">{icon}</div>
                  <h4>{title}</h4>
                  <p>{desc}</p>
                </div>
                """, unsafe_allow_html=True)

    with tab3:
        st.subheader("All 38 Disease Classes")
        DISEASE_CLASSES = [
            "Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___healthy",
            "Blueberry___healthy","Cherry_(including_sour)___Powdery_mildew","Cherry_(including_sour)___healthy",
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Common_rust_",
            "Corn_(maize)___Northern_Leaf_Blight","Corn_(maize)___healthy",
            "Grape___Black_rot","Grape___Esca_(Black_Measles)","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)","Grape___healthy",
            "Orange___Haunglongbing_(Citrus_greening)","Peach___Bacterial_spot","Peach___healthy",
            "Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy",
            "Potato___Early_blight","Potato___Late_blight","Potato___healthy",
            "Raspberry___healthy","Soybean___healthy","Squash___Powdery_mildew",
            "Strawberry___Leaf_scorch","Strawberry___healthy","Tomato___Bacterial_spot","Tomato___Early_blight",
            "Tomato___Late_blight","Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot",
            "Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___Tomato_mosaic_virus","Tomato___healthy"
        ]
        html = ""
        for c in DISEASE_CLASSES:
            is_h = "healthy" in c
            pill_cls = "class-pill-healthy" if is_h else "class-pill-diseased"
            label = ("✓ " if is_h else "") + c.replace("___"," → ").replace("_"," ")
            html += f'<span class="class-pill {pill_cls}">{label}</span>'
        st.markdown(html, unsafe_allow_html=True)
