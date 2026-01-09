from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from app.detector import detect_ui_elements
from app.utils import load_image_from_input
from app.schemas import DetectionResponse

app = FastAPI(
    title="UI Component Detector",
    description="Detects UI components from screenshots using a vision-capable LLM",
    version="1.0.0"
)
app.mount("/ui", StaticFiles(directory="static", html=True), name="static")
class DetectionRequest(BaseModel):
    image: str  # base64 string or image URL

@app.post("/detect-ui-elements", response_model=DetectionResponse)
def detect_ui_elements_endpoint(request: DetectionRequest):
    try:
        # Load image bytes
        image_bytes = load_image_from_input(request.image)

        # Run detection
        result = detect_ui_elements(image_bytes)

        # Validate output structure
        validated = DetectionResponse(**result)

        return validated

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
