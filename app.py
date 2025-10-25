from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
import joblib
import pandas as pd

# 🌊 FastAPI App Initialization
app = FastAPI(
    title="🌊 Tsunami Prediction API",
    description="Predicts tsunami possibility using trained ML model",
    version="1.2"
)

# 🔓 Enable CORS (for Flutter frontend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # sab origins allowed (for testing)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🧠 Load Trained Model
try:
    model = joblib.load("tsunami_prediction_model.pkl")
except Exception as e:
    model = None
    print(f"⚠️ Model load error: {e}")

# ✅ Input Validation Schema
class TsunamiInput(BaseModel):
    magnitude: float = Field(..., gt=0, description="Earthquake magnitude (e.g., 6.5)")
    depth: float = Field(..., ge=0, description="Depth of earthquake in km")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude (-90 to 90)")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude (-180 to 180)")
    dmin: float = Field(..., ge=0, description="Minimum distance from the epicenter")

# 🌐 Home Route
@app.get("/")
def home():
    return {
        "message": "Welcome to 🌊 Tsunami Prediction API",
        "docs": "Visit /docs to test the API",
    }

# 🔮 Prediction Route
@app.post("/predict")
async def predict_tsunami(request: Request):
    """
    Predicts whether a tsunami is likely based on earthquake parameters.
    """

    try:
        # Parse JSON manually (so we can print it if invalid)
        body = await request.json()
        print("📩 Received Data:", body)

        # Validate data
        input_data = TsunamiInput(**body)

        # Convert validated input into DataFrame
        new_data = pd.DataFrame([input_data.dict()])

        # Ensure model loaded
        if model is None:
            return {"error": "Model not loaded on server."}

        # Model prediction
        prediction = model.predict(new_data)[0]
        result = "🌊 Tsunami Warning!" if prediction == 1 else "✅ No Tsunami Expected"

        return {
            "input": input_data.dict(),
            "prediction": result
        }

    except ValidationError as ve:
        print("❌ Validation Error:", ve.errors())
        return {"error": "Invalid input format", "details": ve.errors()}

    except Exception as e:
        print("⚠️ Prediction Error:", str(e))
        return {"error": "Internal server error", "details": str(e)}
