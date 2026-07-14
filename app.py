"""
AgroFarmer - Streamlit Frontend
Connects to FastAPI backend at http://127.0.0.1:8000
"""

import streamlit as st
import requests
from PIL import Image
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgroFarmer",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Backend URL ───────────────────────────────────────────────────────────────
API_URL = "http://127.0.0.1:8000"

# ── Design System ─────────────────────────────────────────────────────────────
# Theme: Deep forest green + gold accent + dark neutral background
# Professional, agricultural, trustworthy

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Playfair+Display:wght@600;700&display=swap');

    /* ── Base ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0f1710;
        color: #d4e6d4;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #0a1209 !important;
        border-right: 1px solid #1e3320;
    }
    [data-testid="stSidebar"] * {
        color: #a8c8a8 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #1e3320 !important;
    }

    /* ── Main content background ── */
    .stApp {
        background-color: #0f1710;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* ── Typography ── */
    h1, h2, h3 {
        font-family: 'Playfair Display', serif !important;
        color: #f5c518 !important;
        letter-spacing: -0.02em;
    }
    h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        color: #c8dcc8 !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.78rem !important;
    }
    p, li, span {
        color: #a8c8a8;
        line-height: 1.7;
    }

    /* ── Divider ── */
    hr {
        border-color: #1e3320 !important;
        margin: 1.5rem 0 !important;
    }

    /* ── Cards ── */
    .card {
        background: #131f14;
        border: 1px solid #1e3320;
        border-radius: 6px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .card-accent-green {
        border-left: 3px solid #4a9e4a;
    }
    .card-accent-gold {
        border-left: 3px solid #f5c518;
    }
    .card-accent-blue {
        border-left: 3px solid #4a90c4;
    }

    /* ── Disease banner ── */
    .banner {
        border-radius: 6px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .banner-disease {
        background: linear-gradient(135deg, #1a0a00, #3d1500);
        border: 1px solid #8b3a00;
    }
    .banner-healthy {
        background: linear-gradient(135deg, #001a0a, #003d1a);
        border: 1px solid #006b2e;
    }
    .banner h2 {
        font-family: 'Playfair Display', serif !important;
        font-size: 2rem !important;
        margin: 0 0 0.4rem !important;
        color: #ffffff !important;
    }
    .banner p {
        color: #cccccc !important;
        font-size: 0.95rem;
        margin: 0;
        letter-spacing: 0.04em;
    }
    .confidence-text {
        color: #f5c518 !important;
        font-weight: 600;
    }

    /* ── Feature cards on home ── */
    .feature-card {
        background: #131f14;
        border: 1px solid #1e3320;
        border-top: 3px solid #4a9e4a;
        border-radius: 6px;
        padding: 1.5rem;
        height: 100%;
    }
    .feature-title {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #4a9e4a !important;
        margin-bottom: 0.8rem;
    }
    .feature-text {
        color: #8aaa8a !important;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* ── Stats row ── */
    .stat-box {
        background: #131f14;
        border: 1px solid #1e3320;
        border-radius: 6px;
        padding: 1.2rem;
        text-align: center;
    }
    .stat-number {
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        color: #f5c518 !important;
        display: block;
        line-height: 1;
        margin-bottom: 0.3rem;
    }
    .stat-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6a8a6a !important;
    }

    /* ── Upload area ── */
    .upload-placeholder {
        background: #131f14;
        border: 2px dashed #1e3320;
        border-radius: 6px;
        padding: 3rem 1rem;
        text-align: center;
        color: #4a6a4a;
    }

    /* ── Tags ── */
    .tag {
        display: inline-block;
        border-radius: 3px;
        padding: 0.25rem 0.75rem;
        font-size: 0.78rem;
        font-weight: 500;
        margin: 0.2rem;
        letter-spacing: 0.02em;
    }
    .tag-advisory {
        background: #2a1f00;
        border: 1px solid #6b4f00;
        color: #d4a800 !important;
    }
    .tag-source {
        background: #0a1a0a;
        border: 1px solid #2a4a2a;
        color: #6aaa6a !important;
    }

    /* ── Section label ── */
    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #4a9e4a !important;
        margin-bottom: 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #1e3320;
    }

    /* ── Recommendation text ── */
    .rec-content {
        color: #c8dcc8 !important;
        font-size: 0.92rem;
        line-height: 1.8;
    }
    .rec-content * {
        color: #c8dcc8 !important;
    }
    .rec-content strong, .rec-content b {
        color: #f5c518 !important;
        font-weight: 600;
    }
    .rec-content h3 {
        color: #f5c518 !important;
        font-size: 1rem !important;
        margin-top: 1rem;
    }

    /* ── Weather values ── */
    .weather-value {
        color: #ffffff !important;
        font-weight: 600;
    }
    .weather-label {
        color: #6a8a6a !important;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* ── Table ── */
    table {
        border-collapse: collapse;
        width: 100%;
    }
    th {
        background: #1e3320 !important;
        color: #f5c518 !important;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 0.7rem 1rem !important;
        border: 1px solid #2a4a2a !important;
    }
    td {
        color: #a8c8a8 !important;
        padding: 0.6rem 1rem !important;
        border: 1px solid #1e3320 !important;
        font-size: 0.88rem;
    }
    tr:nth-child(even) td {
        background: #131f14 !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: #f5c518 !important;
        color: #0a1209 !important;
        border: none !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        background: #d4a800 !important;
        transform: translateY(-1px);
    }

    /* ── Input fields ── */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        background: #131f14 !important;
        border: 1px solid #2a4a2a !important;
        color: #c8dcc8 !important;
        border-radius: 4px !important;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: #131f14 !important;
        border: 1px dashed #2a4a2a !important;
        border-radius: 6px !important;
    }

    /* ── Radio ── */
    .stRadio > div {
        gap: 1rem;
    }

    /* ── Footer ── */
    footer { visibility: hidden; }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: #f5c518 !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 0.5rem;'>
        <span style='font-size:1.5rem;'>🌾</span>
        <span style='font-family: Playfair Display, serif; font-size: 1.4rem;
                     color: #f5c518; font-weight: 700; margin-left: 0.4rem;'>
            AgroFarmer
        </span>
    </div>
    <p style='color: #4a6a4a; font-size: 0.78rem; text-transform: uppercase;
              letter-spacing: 0.1em; margin: 0 0 1rem;'>
        Rice Disease Intelligence
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Home", "Disease Advisor", "About"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown("""
    <div style='font-size: 0.75rem; color: #4a6a4a; text-transform: uppercase;
                letter-spacing: 0.08em; margin-bottom: 0.8rem;'>
        Model Info
    </div>
    """, unsafe_allow_html=True)

    info_items = [
        ("Model", "YOLOv8n-cls"),
        ("Accuracy", "99.4%"),
        ("Crop", "Rice"),
        ("Classes", "10 diseases"),
    ]
    for label, value in info_items:
        st.markdown(f"""
        <div style='display:flex; justify-content:space-between;
                    padding: 0.3rem 0; border-bottom: 1px solid #1e3320;'>
            <span style='color: #4a6a4a; font-size: 0.8rem;'>{label}</span>
            <span style='color: #c8dcc8; font-size: 0.8rem; font-weight: 500;'>{value}</span>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — HOME
# ─────────────────────────────────────────────────────────────────────────────
if page == "Home":

    # Hero
    st.markdown("""
    <div style='padding: 3rem 0 2rem; border-bottom: 1px solid #1e3320; margin-bottom: 2rem;'>
        <div style='font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
                    letter-spacing: 0.15em; color: #4a9e4a; margin-bottom: 0.8rem;'>
            AI-Powered Agricultural Intelligence
        </div>
        <h1 style='font-size: 3.5rem; margin: 0 0 1rem; line-height: 1.1;'>
            🌾 AgroFarmer
        </h1>
        <p style='font-size: 1.05rem; color: #6a8a6a; max-width: 560px;
                  line-height: 1.7; margin: 0;'>
            Assam is an agricultural state where a significant portion of people's
            livelihood depends on farming. AgroFarmer brings AI to the fields —
            helping farmers identify rice diseases and get expert guidance instantly.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    stats = [
        ("99.4%", "Test Accuracy"),
        ("15,023", "Training Images"),
        ("10", "Disease Classes"),
        ("9", "Knowledge Documents"),
    ]
    cols = st.columns(4)
    for col, (number, label) in zip(cols, stats):
        with col:
            st.markdown(f"""
            <div class='stat-box'>
                <span class='stat-number'>{number}</span>
                <span class='stat-label'>{label}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Features
    features = [
        ("Disease Detection",
         "Upload a rice leaf photo and the AI identifies the disease with 99.4% accuracy across 10 disease classes using YOLOv8."),
        ("Weather-Aware Advice",
         "Real-time weather from your location is factored into every recommendation — humidity, rainfall, and temperature all considered."),
        ("Knowledge-Grounded",
         "Recommendations are generated from ICAR, IRRI, and FAO agricultural documents via RAG — not LLM guesswork."),
        ("Multilingual",
         "Get recommendations in English, Hindi, or Assamese — designed for farmers across Assam and Northeast India."),
    ]
    col1, col2 = st.columns(2)
    for i, (title, text) in enumerate(features):
        col = col1 if i % 2 == 0 else col2
        with col:
            st.markdown(f"""
            <div class='feature-card' style='margin-bottom: 1rem;'>
                <div class='feature-title'>{title}</div>
                <div class='feature-text'>{text}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Disease classes
    st.markdown("""
    <div class='section-label'>Supported Rice Diseases</div>
    """, unsafe_allow_html=True)

    diseases = [
        "Bacterial Leaf Blight", "Brown Spot", "Leaf Blast", "Leaf Scald",
        "Narrow Brown Spot", "Neck Blast", "Rice Hispa", "Sheath Blight",
        "Tungro", "Healthy"
    ]
    cols = st.columns(5)
    for i, disease in enumerate(diseases):
        with cols[i % 5]:
            color = "#003d1a" if disease == "Healthy" else "#1a0a00"
            border = "#006b2e" if disease == "Healthy" else "#4a2000"
            text_color = "#4aaa6a" if disease == "Healthy" else "#c87a3a"
            st.markdown(f"""
            <div style='background:{color}; border:1px solid {border};
                        border-radius:4px; padding:0.5rem 0.8rem;
                        margin-bottom:0.5rem; text-align:center;'>
                <span style='color:{text_color}; font-size:0.82rem; font-weight:500;'>
                    {disease}
                </span>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — DISEASE ADVISOR
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Disease Advisor":

    st.markdown("""
    <div style='padding-bottom: 1.5rem; border-bottom: 1px solid #1e3320; margin-bottom: 2rem;'>
        <div style='font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
                    letter-spacing: 0.15em; color: #4a9e4a; margin-bottom: 0.4rem;'>
            Diagnosis Tool
        </div>
        <h2 style='margin: 0; font-size: 2rem;'>Disease Advisor</h2>
        <p style='color: #4a6a4a; margin: 0.4rem 0 0; font-size: 0.88rem;'>
            Upload a rice leaf image and enter your location to receive a complete
            diagnosis and treatment plan.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_gap, col_preview = st.columns([2, 0.15, 1])

    with col_form:
        # Image upload
        st.markdown("<div class='section-label'>Leaf Image</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload rice leaf image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Location
        st.markdown("<div class='section-label'>Location</div>", unsafe_allow_html=True)
        location_type = st.radio(
            "Location type",
            ["City Name", "Coordinates"],
            horizontal=True,
            label_visibility="collapsed",
        )

        city = lat = lon = None
        if location_type == "City Name":
            city = st.text_input(
                "City",
                placeholder="e.g. Tezpur, Guwahati, Jorhat",
                label_visibility="collapsed",
            )
        else:
            c1, c2 = st.columns(2)
            with c1:
                lat = st.number_input("Latitude", value=26.63, format="%.4f")
            with c2:
                lon = st.number_input("Longitude", value=92.80, format="%.4f")

        st.markdown("<br>", unsafe_allow_html=True)

        # Language
        st.markdown("<div class='section-label'>Response Language</div>", unsafe_allow_html=True)
        language = st.selectbox(
            "Language",
            ["english", "hindi", "assamese"],
            format_func=lambda x: {
                "english": "English",
                "hindi": "Hindi",
                "assamese": "Assamese (Experimental)"
            }.get(x, x),
            label_visibility="collapsed",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("Run Analysis", type="primary", use_container_width=True)

    with col_preview:
        st.markdown("<div class='section-label'>Image Preview</div>", unsafe_allow_html=True)
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        else:
            st.markdown("""
            <div class='upload-placeholder'>
                <div style='font-size: 2rem; margin-bottom: 0.5rem; opacity: 0.3;'>
                    [ ]
                </div>
                <div style='font-size: 0.8rem; color: #3a5a3a;'>
                    No image uploaded
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Analysis ──────────────────────────────────────────────────────────────
    if analyze_btn:
        if not uploaded_file:
            st.error("Please upload a rice leaf image before running analysis.")
            st.stop()

        if location_type == "City Name" and not city:
            st.warning("No location entered — weather context will be excluded from the recommendation.")

        with st.spinner("Running analysis..."):
            try:
                files  = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                params = {"language": language}

                if location_type == "City Name" and city:
                    params["city"] = city
                elif location_type == "Coordinates" and lat and lon:
                    params["lat"] = lat
                    params["lon"] = lon

                response = requests.post(
                    f"{API_URL}/advisor",
                    files=files,
                    params=params,
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the backend. Make sure FastAPI is running on port 8000.")
                st.stop()
            except requests.exceptions.Timeout:
                st.error("Request timed out. Please try again.")
                st.stop()
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.stop()

        st.markdown("---")

        # Disease banner
        detection  = data.get("detection", {})
        disease    = detection.get("disease", "Unknown").replace("_", " ").title()
        confidence = detection.get("confidence", 0) * 100
        is_healthy = detection.get("is_healthy", False)

        banner_class  = "banner-healthy" if is_healthy else "banner-disease"
        status_text   = "No disease detected" if is_healthy else "Disease detected"

        st.markdown(f"""
        <div class='banner {banner_class}'>
            <h2>{disease}</h2>
            <p>
                {status_text} &nbsp;&nbsp;|&nbsp;&nbsp;
                <span class='confidence-text'>Confidence: {confidence:.1f}%</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Results columns
        col_w, col_r = st.columns([1, 2])

        with col_w:
            weather = data.get("weather")
            if weather:
                st.markdown("<div class='section-label'>Weather Conditions</div>", unsafe_allow_html=True)

                weather_items = [
                    ("Location", weather.get("location", "—")),
                    ("Temperature", f"{weather.get('temperature')}°C"),
                    ("Humidity", f"{weather.get('humidity')}%"),
                    ("Rainfall", f"{weather.get('rainfall_1h')} mm/hr"),
                    ("Wind Speed", f"{weather.get('wind_speed')} m/s"),
                    ("Conditions", weather.get("description", "—").capitalize()),
                ]

                st.markdown("<div class='card card-accent-blue'>", unsafe_allow_html=True)
                for label, value in weather_items:
                    st.markdown(f"""
                    <div style='display:flex; justify-content:space-between;
                                padding: 0.35rem 0; border-bottom: 1px solid #1e2a3a;'>
                        <span class='weather-label'>{label}</span>
                        <span class='weather-value'>{value}</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                advisories = weather.get("advisory", [])
                if advisories:
                    st.markdown("<div class='section-label' style='margin-top:1rem;'>Farming Advisories</div>",
                                unsafe_allow_html=True)
                    for advisory in advisories:
                        st.markdown(f"<span class='tag tag-advisory'>{advisory}</span>",
                                    unsafe_allow_html=True)
            else:
                st.info("No weather data — location was not provided.")

        with col_r:
            st.markdown("<div class='section-label'>Treatment Recommendation</div>", unsafe_allow_html=True)
            recommendation = data.get("recommendation", "No recommendation available.")

            st.markdown(f"""
            <div class='card card-accent-green'>
                <div class='rec-content'>
                    {recommendation.replace(chr(10), '<br>')}
                </div>
            </div>
            """, unsafe_allow_html=True)

            rag_sources = data.get("rag_sources", [])
            if rag_sources:
                st.markdown("<div class='section-label' style='margin-top:1rem;'>Knowledge Sources</div>",
                            unsafe_allow_html=True)
                for source in list(set(rag_sources)):
                    st.markdown(f"<span class='tag tag-source'>{source}</span>",
                                unsafe_allow_html=True)

        if is_healthy:
            st.success("The rice plant appears healthy. Continue with regular crop management practices.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "About":

    st.markdown("""
    <div style='padding-bottom: 1.5rem; border-bottom: 1px solid #1e3320; margin-bottom: 2rem;'>
        <div style='font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
                    letter-spacing: 0.15em; color: #4a9e4a; margin-bottom: 0.4rem;'>
            Project Overview
        </div>
        <h2 style='margin: 0; font-size: 2rem;'>About AgroFarmer</h2>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("<div class='section-label'>Motivation</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='card card-accent-gold'>
            <p style='color: #c8dcc8; line-height: 1.8; margin: 0;'>
                Assam is an agricultural state where a significant portion of people's
                livelihood depends on farming. Living in a modern era, I asked myself —
                why can't we use technology for the betterment of our farmers?
                <br><br>
                Thinking of that, I started AgroFarmer — a project that helps farmers
                identify crop diseases, gather information about those diseases, and get
                guidance on the precautions, pesticides, and fertilizers they should use.
                For this version, I selected Rice as the crop and built a rice disease
                identification and information query system as the foundation.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>System Architecture</div>", unsafe_allow_html=True)
        st.code("""
Farmer Input  →  Image + Location + Language
                        |
          ┌─────────────┼─────────────┐
          |             |             |
    YOLOv8n-cls   OpenWeatherMap  ChromaDB RAG
    (Disease)      (Weather)      (Documents)
          |             |             |
          └─────────────┼─────────────┘
                        |
              Groq Llama 3.3-70B
                        |
             Structured Recommendation
        """, language=None)

    with col2:
        st.markdown("<div class='section-label'>Model Performance</div>", unsafe_allow_html=True)
        metrics = [
            ("Architecture", "YOLOv8n-cls"),
            ("Total Images", "15,023"),
            ("Training Set", "10,516"),
            ("Validation Set", "3,005"),
            ("Test Set", "1,502"),
            ("Validation Accuracy", "99.6%"),
            ("Test Accuracy", "99.4%"),
            ("Inference Speed", "0.3 ms/image"),
            ("Disease Classes", "10"),
        ]
        for label, value in metrics:
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between;
                        padding: 0.5rem 0; border-bottom: 1px solid #1e3320;'>
                <span style='color: #4a6a4a; font-size: 0.82rem;'>{label}</span>
                <span style='color: #c8dcc8; font-size: 0.82rem; font-weight: 500;'>{value}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Tech Stack</div>", unsafe_allow_html=True)
        stack = {
            "Detection": "YOLOv8n-cls (Ultralytics)",
            "LLM": "Groq Llama 3.3-70B",
            "Vector DB": "ChromaDB",
            "Embeddings": "sentence-transformers",
            "Backend": "FastAPI + Uvicorn",
            "Frontend": "Streamlit",
            "Weather": "OpenWeatherMap API",
            "Training": "Google Colab (T4 GPU)",
        }
        for label, value in stack.items():
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between;
                        padding: 0.4rem 0; border-bottom: 1px solid #1e3320;'>
                <span style='color: #4a6a4a; font-size: 0.8rem;'>{label}</span>
                <span style='color: #c8dcc8; font-size: 0.8rem;'>{value}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style='display:flex; justify-content:space-between; align-items:center;
                padding: 1rem 0;'>
        <div>
            <div style='font-weight: 600; color: #c8dcc8; font-size: 0.95rem;'>
                Rajesh Sarma Bordoloi
            </div>
            <div style='color: #4a6a4a; font-size: 0.8rem; margin-top: 0.2rem;'>
                AI Engineer &nbsp;|&nbsp; GenAI &nbsp;|&nbsp; Computer Vision &nbsp;|&nbsp; FastAPI
            </div>
        </div>
        <div>
            <a href='https://github.com/rajesh-snippet/AgroFarmer'
               style='color: #4a9e4a; font-size: 0.85rem; text-decoration: none;
                      border: 1px solid #2a4a2a; padding: 0.4rem 1rem; border-radius: 4px;'>
                View on GitHub
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)