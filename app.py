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

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Sora:wght@600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #1a2e1a;
    }
    [data-testid="stSidebar"] * {
        color: #e8f5e8 !important;
    }

    /* ── Result cards ── */
    .result-card {
        background: #1e3a1e;
        border-left: 4px solid #4caf50;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        color: #e8f5e8 !important;
    }
    .result-card * {
        color: #e8f5e8 !important;
    }
    .disease-banner {
        background: linear-gradient(135deg, #1a2e1a, #2d6a2d);
        color: white;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .disease-banner h2 {
        font-family: 'Sora', sans-serif;
        font-size: 1.8rem;
        margin: 0;
        color: white !important;
    }
    .disease-banner p {
        margin: 0.3rem 0 0;
        font-size: 1rem;
        opacity: 0.85;
        color: #c8e6c8 !important;
    }
    .healthy-banner {
        background: linear-gradient(135deg, #1b4332, #40916c);
    }
    .disease-detected-banner {
        background: linear-gradient(135deg, #7b2d00, #d4651a);
    }
    .weather-card {
        background: #1a2a3a;
        border-left: 4px solid #42a5f5;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        color: #e3f2fd !important;
    }
    .weather-card * {
        color: #e3f2fd !important;
    }
    .advisory-tag {
        display: inline-block;
        background: #fff3cd;
        border: 1px solid #ffc107;
        color: #856404;
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.85rem;
        margin: 0.2rem;
    }
    .source-tag {
        display: inline-block;
        background: #e8f5e8;
        border: 1px solid #2d6a2d;
        color: #1a3d1a;
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.82rem;
        margin: 0.2rem;
    }
    .section-title {
        font-family: 'Sora', sans-serif;
        color: #81c784 !important;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #2d6a2d;
        padding-bottom: 0.3rem;
    }
    /* ── Hide Streamlit default footer ── */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='color: #f5c518;'>🌾 AgroFarmer</h2>", unsafe_allow_html=True)
    st.markdown("*AI-powered rice disease advisor*")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔬 Disease Advisor", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Model:** YOLOv8n-cls")
    st.markdown("**Accuracy:** 99.4%")
    st.markdown("**Crops:** Rice")
    st.markdown("**Diseases:** 10 classes")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — HOME
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown("""
    <div style='text-align:center; padding: 2rem 0 1rem;'>
        <h1 style='font-family: Sora, sans-serif; font-size: 3rem; color: #f5c518;'>
            🌾 AgroFarmer
        </h1>
        <p style='font-size: 1.2rem; color: #4a7a4a; max-width: 600px; margin: 0 auto;'>
            AI-powered rice disease detection and treatment advisor for farmers.
            Upload a leaf image and get an instant diagnosis with actionable advice.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🔬 Disease Detection")
        st.markdown("Upload a rice leaf photo and the AI identifies the disease with 99.4% accuracy across 10 disease classes.")
    with col2:
        st.markdown("### 🌤️ Weather-Aware Advice")
        st.markdown("Real-time weather data from your location is factored into every recommendation.")
    with col3:
        st.markdown("### 📚 Knowledge-Grounded")
        st.markdown("Recommendations are generated from ICAR, IRRI, and FAO agricultural documents — not guesswork.")

    st.markdown("---")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("### Supported Diseases")
        diseases = [
            " Bacterial Leaf Blight",
            " Brown Spot",
            " Leaf Blast",
            " Leaf Scald",
            " Narrow Brown Spot",
            " Neck Blast",
            " Rice Hispa",
            " Sheath Blight",
            " Tungro",
            " Healthy",
        ]
        for d in diseases:
            st.markdown(f"- {d}")

    with col_b:
        st.markdown("### Tech Stack")
        st.markdown("""
        -  **YOLOv8n-cls** — Disease detection
        -  **FastAPI** — Backend API
        -  **ChromaDB** — Vector knowledge base
        -  **Groq Llama 3.3-70B** — LLM recommendations
        -  **OpenWeatherMap** — Weather data
        -  **Streamlit** — Frontend
        """)

    with col_c:
        st.markdown("### Languages Supported")
        st.markdown("""
        - 🇬🇧 English
        - 🇮🇳 Hindi
        - 🏔️ Assamese *(experimental)*
        """)

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;'>
        <p style='color: #4a7a4a;'>
            Select <strong>Disease Advisor</strong> from the sidebar to get started.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — DISEASE ADVISOR
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔬 Disease Advisor":
    st.markdown("## Rice Disease Advisor")
    st.markdown("Upload a rice leaf image and enter your location to get a complete diagnosis and treatment plan.")
    st.markdown("---")

    # ── Input Section ─────────────────────────────────────────────────────────
    col_input, col_gap, col_preview = st.columns([2, 0.2, 1])

    with col_input:
        st.markdown("#### 📷 Upload Leaf Image")
        uploaded_file = st.file_uploader(
            "Choose a rice leaf image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        st.markdown("#### 📍 Location")
        location_type = st.radio(
            "Location input type",
            ["City Name", "Coordinates"],
            horizontal=True,
            label_visibility="collapsed",
        )

        city = None
        lat = lon = None

        if location_type == "City Name":
            city = st.text_input("City name", placeholder="e.g. Tezpur, Guwahati, Jorhat")
        else:
            c1, c2 = st.columns(2)
            with c1:
                lat = st.number_input("Latitude", value=26.63, format="%.4f")
            with c2:
                lon = st.number_input("Longitude", value=92.80, format="%.4f")

        st.markdown("#### 🌐 Language")
        language = st.selectbox(
            "Response language",
            ["english", "hindi", "assamese"],
            format_func=lambda x: {"english": "🇬🇧 English", "hindi": "🇮🇳 Hindi", "assamese": "🏔️ Assamese"}.get(x, x),
            label_visibility="collapsed",
        )

        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

    with col_preview:
        if uploaded_file:
            st.markdown("#### Preview")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        else:
            st.markdown("""
            <div style='background:#f0f7f0; border:2px dashed #2d6a2d; border-radius:10px;
                        padding:3rem 1rem; text-align:center; color:#4a7a4a; margin-top:2rem;'>
                <p style='font-size:2rem;margin:0;'>🍃</p>
                <p style='margin:0.5rem 0 0;'>Image preview appears here</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Analysis ──────────────────────────────────────────────────────────────
    if analyze_btn:
        if not uploaded_file:
            st.error("Please upload a rice leaf image first.")
            st.stop()

        if location_type == "City Name" and not city:
            st.warning("No city entered — weather context will be skipped.")

        with st.spinner("Analyzing image and generating recommendations..."):
            try:
                # Build multipart form data
                files   = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                params  = {"language": language}

                if location_type == "City Name" and city:
                    params["city"] = city
                elif location_type == "Coordinates":
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
                st.error("Request timed out. The server might be overloaded.")
                st.stop()
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()

        st.markdown("---")
        st.markdown("## 📊 Results")

        # ── Disease Banner ────────────────────────────────────────────────────
        detection   = data.get("detection", {})
        disease     = detection.get("disease", "Unknown").replace("_", " ").title()
        confidence  = detection.get("confidence", 0) * 100
        is_healthy  = detection.get("is_healthy", False)

        banner_class = "healthy-banner" if is_healthy else "disease-detected-banner"
        status_icon  = "✅" if is_healthy else "⚠️"
        status_text  = "Plant appears healthy" if is_healthy else "Disease detected"

        st.markdown(f"""
        <div class='disease-banner {banner_class}'>
            <h2>{status_icon} {disease}</h2>
            <p>{status_text} &nbsp;|&nbsp; Confidence: {confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

        # ── Weather + Recommendation side by side ─────────────────────────────
        col_weather, col_rec = st.columns([1, 2])

        with col_weather:
            weather = data.get("weather")
            if weather:
                st.markdown("<p class='section-title'>🌤️ Weather Conditions</p>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='weather-card'>
                    <b>📍 {weather.get('location', 'Unknown')}</b><br><br>
                    🌡️ Temperature: <b>{weather.get('temperature')}°C</b><br>
                    💧 Humidity: <b>{weather.get('humidity')}%</b><br>
                    🌧️ Rainfall: <b>{weather.get('rainfall_1h')} mm/hr</b><br>
                    💨 Wind: <b>{weather.get('wind_speed')} m/s</b><br>
                    ☁️ {weather.get('description', '').capitalize()}
                </div>
                """, unsafe_allow_html=True)

                advisories = weather.get("advisory", [])
                if advisories:
                    st.markdown("**⚠️ Farming Advisories:**")
                    advisory_html = "".join([f"<span class='advisory-tag'>⚡ {a}</span>" for a in advisories])
                    st.markdown(advisory_html, unsafe_allow_html=True)
            else:
                st.info("No weather data — location was not provided.")

        with col_rec:
            st.markdown("<p class='section-title'>💊 Treatment Recommendation</p>", unsafe_allow_html=True)
            recommendation = data.get("recommendation", "No recommendation available.")
            st.markdown(f"""
            <div class='result-card'>
                {recommendation.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

            # RAG sources
            rag_sources = data.get("rag_sources", [])
            if rag_sources:
                st.markdown("**📚 Knowledge Sources Used:**")
                unique_sources = list(set(rag_sources))
                sources_html = "".join([f"<span class='source-tag'>📄 {s}</span>" for s in unique_sources])
                st.markdown(sources_html, unsafe_allow_html=True)

        # ── Healthy plant message ─────────────────────────────────────────────
        if is_healthy:
            st.success("Your rice plant looks healthy! Continue with regular crop management practices.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "ℹ️ About":
    st.markdown("## ℹ️ About AgroFarmer")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Project Goal")
        st.markdown("""
        AgroFarmer is an AI-powered agricultural advisory system designed to help
        rice farmers in Assam and across India identify crop diseases early and
        get actionable treatment recommendations in their local language.

        The system combines computer vision, retrieval-augmented generation (RAG),
        and large language models to provide grounded, accurate advice based on
        real agricultural documents from ICAR, IRRI, and FAO.
        """)

        st.markdown("### 🏗️ Architecture")
        st.markdown("""
        ```
        Farmer Input (Image + Location + Language)
                    ↓
        YOLOv8n-cls  →  Disease + Confidence
        OpenWeatherMap →  Weather Context
        ChromaDB RAG  →  Relevant Document Chunks
                    ↓
        Groq Llama 3.3-70B
                    ↓
        Structured Recommendation
        ```
        """)

    with col2:
        st.markdown("### 🧪 Model Performance")
        st.markdown("""
        | Metric | Value |
        |--------|-------|
        | Model | YOLOv8n-cls |
        | Training samples | 10,516 |
        | Validation accuracy | 99.6% |
        | Test accuracy | 99.4% |
        | Inference speed | 0.3ms/image |
        | Disease classes | 10 |
        """)

        st.markdown("### 🦠 Supported Disease Classes")
        st.markdown("""
        1. Bacterial Leaf Blight
        2. Brown Spot
        3. Leaf Blast
        4. Leaf Scald
        5. Narrow Brown Spot
        6. Neck Blast
        7. Rice Hispa
        8. Sheath Blight
        9. Tungro
        10. Healthy
        """)

    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        **AI / ML**
        - YOLOv8n-cls (Ultralytics)
        - Groq Llama 3.3-70B
        - sentence-transformers
        - ChromaDB
        - LangChain
        """)
    with c2:
        st.markdown("""
        **Backend**
        - FastAPI
        - Uvicorn
        - Python 3.12
        - OpenWeatherMap API
        - Pydantic
        """)
    with c3:
        st.markdown("""
        **Frontend & Tools**
        - Streamlit
        - Roboflow (annotation)
        - Google Colab (training)
        - Git / GitHub
        """)

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color:#4a7a4a; padding:1rem;'>
        Built by <strong>Rajesh Sarma Bordoloi</strong> · AI Engineer <br>
        <a href='https://github.com' style='color:#2d6a2d;'>GitHub</a>
    </div>
    """, unsafe_allow_html=True)