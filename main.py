"""
AI Smart Agriculture Advisor "AgroFarmer" - FastAPI Backend
Endpoints:
  GET  /health
  POST /detect-disease
  GET  /weather
  POST /recommend
  POST /advisor  (full pipeline)

Memory note: torch/YOLO and the RAG stack (sentence-transformers + ChromaDB)
are imported and loaded on first use, not at import/startup, so the server
opens its port quickly and /health stays cheap. Set PRELOAD_MODELS=1 on a
host with plenty of RAM (e.g. Hugging Face Spaces) to warm everything at boot.
"""

import io
import os
import threading

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from PIL import Image
from pydantic import BaseModel

# ── Environment ──────────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "0") == "1"

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set (use .env locally or a host secret)")
if not OPENWEATHER_API_KEY:
    raise RuntimeError("OPENWEATHER_API_KEY not set (use .env locally or a host secret)")

MODEL_PATH = "models/best.pt"

# ── Low-confidence guard (see notes: this is a threshold, not real OOD detection)
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.70"))
LOW_CONFIDENCE_MESSAGE = (
    "The image could not be classified with enough confidence. Please upload a "
    "clear, well-lit, close-up photo of a single rice leaf and try again."
)

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AgroFarmer",
    description="Multimodal AI assistant for rice farmers",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=GROQ_API_KEY)


# ── Lazy loaders ─────────────────────────────────────────────────────────────
_model = None
_model_lock = threading.Lock()


def get_model():
    """Load the YOLO classifier on first use (imports torch lazily too)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                if not os.path.exists(MODEL_PATH):
                    raise RuntimeError(f"Model file not found: {MODEL_PATH}")
                # A real best.pt is ~3 MB; a Git LFS pointer file is ~130 bytes.
                if os.path.getsize(MODEL_PATH) < 100_000:
                    raise RuntimeError(
                        f"{MODEL_PATH} looks like a Git LFS pointer, not the real "
                        "weights. Make sure the host pulls LFS files."
                    )
                from ultralytics import YOLO

                _model = YOLO(MODEL_PATH)
                print(f"Model loaded from {MODEL_PATH}")
                print(f"Classes: {list(_model.names.values())}")
    return _model


def _retrieve(query: str, k: int = 4):
    """Import the RAG stack on first use."""
    from rag_service import retrieve

    return retrieve(query, k=k)


@app.on_event("startup")
async def startup_event():
    if PRELOAD_MODELS:
        print("PRELOAD_MODELS=1: warming model and vectorstore...")
        get_model()
        from rag_service import get_vectorstore

        get_vectorstore()
        print("Preload complete")
    else:
        print("Lazy mode: model and vectorstore load on first request")


# ── Schemas ──────────────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    disease: str
    confidence: float
    language: str = "english"  # "english" | "hindi" | "assamese"


class AdvisorRequest(BaseModel):
    language: str = "english"
    city: str | None = None
    lat: float | None = None
    lon: float | None = None


# ── Shared helpers ───────────────────────────────────────────────────────────
def _predict(image: Image.Image) -> tuple[str, float, list[dict]]:
    """Run the classifier. Returns (disease, top1_confidence, top5)."""
    model = get_model()
    result = model.predict(image, verbose=False)[0]
    top1_idx = result.probs.top1
    top1_conf = float(result.probs.top1conf)
    top5 = [
        {"disease": model.names[idx], "confidence": round(float(conf), 4)}
        for idx, conf in zip(result.probs.top5, result.probs.top5conf.tolist())
    ]
    return model.names[top1_idx], top1_conf, top5


async def _read_image(file: UploadFile) -> Image.Image:
    try:
        image_bytes = await file.read()
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not process uploaded image")


def _weather_advisory(w: dict) -> list[str]:
    """Rule-based weather advisory flags fed into the LLM prompt."""
    tips = []
    if w["rainfall_1h"] > 2.0:
        tips.append("Heavy rainfall detected — avoid pesticide spraying.")
    if w["humidity"] > 85:
        tips.append("High humidity — increased risk of fungal diseases.")
    if w["temperature"] > 35:
        tips.append("High temperature — ensure adequate irrigation.")
    if w["wind_speed"] > 7:
        tips.append("Strong winds — delay foliar spray application.")
    if not tips:
        tips.append("Weather conditions are suitable for field operations.")
    return tips


def fetch_weather(city: str | None, lat: float | None, lon: float | None) -> dict:
    """Plain function (no FastAPI Query defaults) so it is safe to call directly."""
    if not city and (lat is None or lon is None):
        raise HTTPException(
            status_code=400,
            detail="Provide either 'city' or both 'lat' and 'lon'",
        )

    base_url = "https://api.openweathermap.org/data/2.5/weather"
    if city:
        params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    else:
        params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"}

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.HTTPError:
        raise HTTPException(
            status_code=404, detail="Location not found. Check city name or coordinates."
        )
    except requests.exceptions.RequestException:
        raise HTTPException(status_code=503, detail="Could not reach weather service.")

    weather = {
        "location": data["name"],
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "description": data["weather"][0]["description"],
        "wind_speed": data["wind"]["speed"],
        "rainfall_1h": data.get("rain", {}).get("1h", 0.0),
    }
    weather["advisory"] = _weather_advisory(weather)
    return weather


def _weather_context(w: dict | None) -> str:
    if not w:
        return ""
    return (
        f"Current weather in {w['location']}: "
        f"{w['temperature']}°C, humidity {w['humidity']}%, "
        f"rainfall {w['rainfall_1h']}mm/hr. "
        f"Advisory: {'; '.join(w['advisory'])}"
    )


def generate_recommendation(
    disease: str, confidence: float, language: str, weather: dict | None
) -> tuple[str, list[str]]:
    """RAG retrieval + Groq call. Returns (recommendation_text, rag_sources)."""
    rag_results = _retrieve(f"How to treat and manage {disease} in rice?", k=4)
    rag_context = "\n\n".join(
        f"[Source: {r['source']}]\n{r['content']}" for r in rag_results
    )

    language_instruction = {
        "english": "Respond in English.",
        "hindi": "Respond in Hindi (Devanagari script).",
        "assamese": (
            "Respond in Assamese language (অসমীয়া). This is Assamese, NOT Bengali. "
            "Use Assamese vocabulary and grammar, not Bengali. For example, use "
            "'কৰক' not 'করুন', 'হয়' not 'হয়', 'আপোনাৰ' not 'আপনার'."
        ),
    }.get(language.lower(), "Respond in English.")

    weather_context = _weather_context(weather)
    weather_line = f"Weather Context: {weather_context}" if weather_context else ""

    prompt = f"""You are an expert agricultural advisor specializing in rice crop diseases.
Use the retrieved knowledge base documents below to give accurate, grounded recommendations.

Disease Detected: {disease}
Confidence: {confidence * 100:.1f}%
{weather_line}

Retrieved Knowledge Base Documents:
{rag_context}

Based on the above documents, provide a structured recommendation covering:
1. Disease explanation (what it is, how it spreads)
2. Immediate actions the farmer should take
3. Treatment (organic and chemical options with dosage if mentioned in documents)
4. Preventive measures for the future
5. Fertilizer advice if relevant

Keep advice practical and suitable for small-scale rice farmers.
Prioritize information from the retrieved documents. If the documents lack specific
treatment details, supplement with your general agricultural knowledge but clearly
indicate when doing so.
{language_instruction}"""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        text = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM service error: {e}")

    return text, [r["source"] for r in rag_results]


# ─────────────────────────────────────────────────────────────────────────────
# 1. HEALTH CHECK  (cheap: does not load the model)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "model_loaded": _model is not None,
        "classes": list(_model.names.values()) if _model is not None else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. DISEASE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/detect-disease")
async def detect_disease(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image = await _read_image(file)
    disease, conf, top5 = await run_in_threadpool(_predict, image)

    return {
        "disease": disease,
        "confidence": round(conf, 4),
        "is_healthy": disease.lower() == "healthy",
        "low_confidence": conf < MIN_CONFIDENCE,
        "top5": top5,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. WEATHER
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/weather")
def get_weather(
    city: str | None = Query(default=None, description="City name e.g. Tezpur"),
    lat: float | None = Query(default=None, description="Latitude e.g. 26.63"),
    lon: float | None = Query(default=None, description="Longitude e.g. 92.80"),
):
    return fetch_weather(city, lat, lon)


# ─────────────────────────────────────────────────────────────────────────────
# 4. LLM RECOMMENDATION
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/recommend")
def recommend(req: RecommendRequest, city: str | None = None):
    weather = None
    if city:
        try:
            weather = fetch_weather(city, None, None)
        except Exception:
            weather = None

    text, sources = generate_recommendation(
        req.disease, req.confidence, req.language, weather
    )
    return {
        "disease": req.disease,
        "confidence": req.confidence,
        "language": req.language,
        "rag_sources": sources,
        "recommendation": text,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. FULL ADVISOR PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/advisor")
async def advisor(
    file: UploadFile = File(...),
    language: str = Query(default="english"),
    city: str | None = Query(default=None),
    lat: float | None = Query(default=None),
    lon: float | None = Query(default=None),
):
    # Step 1: disease detection
    image = await _read_image(file)
    disease, conf, _ = await run_in_threadpool(_predict, image)

    detection = {
        "disease": disease,
        "confidence": round(conf, 4),
        "is_healthy": disease.lower() == "healthy",
    }

    # Step 2: weather (optional; fetched once, reused for the LLM context)
    weather = None
    if city or (lat is not None and lon is not None):
        try:
            weather = await run_in_threadpool(fetch_weather, city, lat, lon)
        except Exception:
            weather = None

    # ── LOW-CONFIDENCE GUARD (drop this block if you don't want it yet) ──────
    # Returns a plain message in the same response shape, so the frontend
    # does not need to change. No LLM call is made.
    if conf < MIN_CONFIDENCE:
        detection["low_confidence"] = True
        return {
            "detection": detection,
            "weather": weather,
            "recommendation": LOW_CONFIDENCE_MESSAGE,
            "rag_sources": [],
            "language": language,
        }
    # ── END GUARD ────────────────────────────────────────────────────────────

    # Step 3: LLM recommendation with RAG
    text, sources = await run_in_threadpool(
        generate_recommendation, disease, conf, language, weather
    )

    return {
        "detection": detection,
        "weather": weather,
        "recommendation": text,
        "rag_sources": sources,
        "language": language,
    }