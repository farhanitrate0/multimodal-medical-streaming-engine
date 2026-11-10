"""Typed contracts for the WebSocket stream and the REST endpoints."""
from typing import Literal, Optional

from pydantic import BaseModel, Field

MAX_PAYLOAD_BYTES = 20 * 1024 * 1024  # 20 MB cap


RequestId = Optional[str]


class StreamRequest(BaseModel):
    mode: Literal["medical", "spatial"]
    image: str = Field(min_length=16, max_length=MAX_PAYLOAD_BYTES, description="base64 image, data URL allowed")
    id: RequestId = Field(None, max_length=64, description="echoed back so replies can arrive out of order")


class MedicalResult(BaseModel):
    id: RequestId = None
    mode: Literal["medical"] = "medical"
    status: str
    confidence: float = Field(description="confidence in the predicted class, %")
    probability: float = Field(ge=0, le=1, description="P(pneumonia)")
    logit: float
    threshold: float
    prediction: Literal[0, 1]
    overlay: str = Field(description="Grad-CAM overlay, PNG data URL")
    latency_ms: float


class Landmarks(BaseModel):
    face: list[list[float]] = Field(description="468/478 normalised x, y, z")
    hands: list[list[list[float]]] = Field(description="21 points per hand")


class SpatialResult(BaseModel):
    id: RequestId = None
    mode: Literal["spatial"] = "spatial"
    status: str
    confidence: float = 0.0
    landmarks: Landmarks
    overlay: str = Field(description="tracking overlay, JPEG data URL")
    latency_ms: float


class ErrorReply(BaseModel):
    id: RequestId = None
    error: str
    status: str


class ModelCard(BaseModel):
    architecture: str
    dataset: str
    test_auroc: Optional[float] = None
    test_sensitivity: Optional[float] = None
    test_specificity: Optional[float] = None


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    device: str
    weights_loaded: bool
    decision_threshold: float
    model: ModelCard
    mean_batch_size: Optional[float] = Field(None, description="avg requests per gpu pass so far")
