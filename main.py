"""
AI Smart Agriculture Advisor "AgroFarmer" - FastAPI Backend
Endpoints:
  GET  /health
  POST /detect-disease
  GET  /weather
  POST /recommend
  POST /advisor  (full pipeline)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
from groq import Groq
from dotenv import load_dotenv
import requests
import os
import io
from rag_service import retrieve, get_vectorstore

# ── Load environment variables ───────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY       = os.getenv("GROQ_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in .env")
if not OPENWEATHER_API_KEY:
    raise RuntimeError("OPENWEATHER_API_KEY not found in .env")

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AgroFarmer",
    description="Multimodal AI assistant for rice farmers",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    print("Loading ChromaDB vectorstore...")
    get_vectorstore()
    print("✅ RAG vectorstore ready")


# ── Load YOLOv8 model once at startup ────────────────────────────────────────
MODEL_PATH = "models/best.pt"
try:
    model = YOLO(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
    print(f"   Classes: {list(model.names.values())}")
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

# ── Groq client ──────────────────────────────────────────────────────────────
groq_client = Groq(api_key=GROQ_API_KEY)

# ── Pydantic schemas ─────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    disease: str
    confidence: float
    language: str = "english"   # "english" | "hindi" | "assamese"

class AdvisorRequest(BaseModel):
    language: str = "english"
    city: str | None = None
    lat: float | None = None
    lon: float | None = None


# ─────────────────────────────────────────────────────────────────────────────
# 1. HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "classes": list(model.names.values()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. DISEASE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/detect-disease")
async def detect_disease(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read and open image
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not process uploaded image")

    # Run inference
    results = model.predict(image, verbose=False)
    result  = results[0]

    top1_idx  = result.probs.top1
    top1_conf = float(result.probs.top1conf)
    disease   = model.names[top1_idx]

    # Top-5 predictions
    top5 = [
        {"disease": model.names[idx], "confidence": round(float(conf), 4)}
        for idx, conf in zip(result.probs.top5, result.probs.top5conf.tolist())
    ]

    return {
        "disease":    disease,
        "confidence": round(top1_conf, 4),
        "is_healthy": disease.lower() == "healthy",
        "top5":       top5,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. WEATHER
# Supports both city name and lat/lon coordinates
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/weather")
def get_weather(
    city: str | None = Query(default=None, description="City name e.g. Tezpur"),
    lat:  float | None = Query(default=None, description="Latitude e.g. 26.63"),
    lon:  float | None = Query(default=None, description="Longitude e.g. 92.80"),
):
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
        raise HTTPException(status_code=404, detail="Location not found. Check city name or coordinates.")
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Could not reach weather service.")

    weather = {
        "location":    data["name"],
        "temperature": data["main"]["temp"],        # °C
        "humidity":    data["main"]["humidity"],    # %
        "description": data["weather"][0]["description"],
        "wind_speed":  data["wind"]["speed"],       # m/s
        "rainfall_1h": data.get("rain", {}).get("1h", 0.0),  # mm, 0 if no rain
    }

    # Simple farming advisory flags derived from weather
    weather["advisory"] = _weather_advisory(weather)

    return weather


def _weather_advisory(w: dict) -> list[str]:
    """Rule-based weather advisory flags fed into the LLM prompt later."""
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


# ─────────────────────────────────────────────────────────────────────────────
# 4. LLM RECOMMENDATION  (disease + weather context → Groq Llama 3.3-70B)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/recommend")
def recommend(req: RecommendRequest, city: str | None = None):
    # Weather context (same as before)
    weather_context = ""
    if city:
        try:
            w = get_weather(city=city)
            weather_context = (
                f"Current weather in {w['location']}: "
                f"{w['temperature']}°C, humidity {w['humidity']}%, "
                f"rainfall {w['rainfall_1h']}mm/hr. "
                f"Advisory: {'; '.join(w['advisory'])}"
            )
        except Exception:
            weather_context = "Weather data unavailable."

    # RAG retrieval — get relevant chunks from knowledge base
    rag_query = f"How to treat and manage {req.disease} in rice?"
    rag_results = retrieve(rag_query, k=4)
    rag_context = "\n\n".join([
        f"[Source: {r['source']}]\n{r['content']}"
        for r in rag_results
    ])

    language_instruction = {
        "english":  "Respond in English.",
        "hindi":    "Respond in Hindi (Devanagari script).",
        "assamese": "Respond in Assamese language (অসমীয়া). This is Assamese, NOT Bengali. Use Assamese vocabulary and grammar, not Bengali. For example, use 'কৰক' not 'করুন', 'হয়' not 'হয়', 'আপোনাৰ' not 'আপনার'.",
    }.get(req.language.lower(), "Respond in English.")

    # Updated prompt now includes RAG context
    prompt = f"""You are an expert agricultural advisor specializing in rice crop diseases.
Use the retrieved knowledge base documents below to give accurate, grounded recommendations.

Disease Detected: {req.disease}
Confidence: {req.confidence * 100:.1f}%
{f"Weather Context: {weather_context}" if weather_context else ""}

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
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        recommendation = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM service error: {e}")

    return {
        "disease":        req.disease,
        "confidence":     req.confidence,
        "language":       req.language,
        "rag_sources":    [r["source"] for r in rag_results],
        "recommendation": recommendation,
    }
# ─────────────────────────────────────────────────────────────────────────────
# 5. FULL ADVISOR PIPELINE
#    Image + location + language → disease detection + weather + LLM advice
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/advisor")
async def advisor(
    file:     UploadFile = File(...),
    language: str        = Query(default="english"),
    city:     str | None = Query(default=None),
    lat:      float | None = Query(default=None),
    lon:      float | None = Query(default=None),
):
    # Step 1 — Disease detection
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not process uploaded image")

    results   = model.predict(image, verbose=False)
    result    = results[0]
    top1_idx  = result.probs.top1
    top1_conf = float(result.probs.top1conf)
    disease   = model.names[top1_idx]

    # Step 2 — Weather (optional, doesn't fail the whole request)
    weather = None
    if city or (lat is not None and lon is not None):
        try:
            weather = get_weather(city=city, lat=lat, lon=lon)
        except Exception:
            weather = None

    # Step 3 — LLM recommendation
    rec_req = RecommendRequest(
        disease=disease,
        confidence=top1_conf,
        language=language,
    )
    recommendation_response = recommend(rec_req, city=city)

    return {
        "detection": {
            "disease":    disease,
            "confidence": round(top1_conf, 4),
            "is_healthy": disease.lower() == "healthy",
        },
        "weather":        weather,
        "recommendation": recommendation_response["recommendation"],
        "language":       language,
    }