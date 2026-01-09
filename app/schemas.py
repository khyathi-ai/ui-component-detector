from pydantic import BaseModel, Field
from typing import List

class BoundingBox(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0, description="Relative x coordinate")
    y: float = Field(..., ge=0.0, le=1.0, description="Relative y coordinate")
    w: float = Field(..., ge=0.0, le=1.0, description="Relative width")
    h: float = Field(..., ge=0.0, le=1.0, description="Relative height")

class UIElement(BaseModel):
    type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    description: str
    bounds: BoundingBox

class DetectionResponse(BaseModel):
    elements: List[UIElement]
