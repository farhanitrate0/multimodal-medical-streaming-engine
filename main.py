import asyncio
import base64
import binascii
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError
from torchvision import transforms

from batcher import MicroBatcher
from gradcam import GradCAMEngine
from model import get_medical_model
from tracking import SpatialTracker, draw
from schemas import (MAX_PAYLOAD_BYTES, ErrorReply, Health, Landmarks, MedicalResult, ModelCard,
                     SpatialResult, StreamRequest)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("engine")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", os.path.join(BASE_DIR, "fine_tuned_medical_resnet.pth"))
METRICS_PATH = os.path.join(BASE_DIR, "results", "metrics.json")
MAX_BATCH = int(os.environ.get("MAX_BATCH", 16))
MAX_WAIT_MS = float(os.environ.get("MAX_WAIT_MS", 2))
MAX_IN_FLIGHT = int(os.environ.get("MAX_IN_FLIGHT", 8))  # per socket

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
model = get_medical_model(pretrained=False).to(device)  # weights loaded below
weights_loaded = False
try:
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
    weights_loaded = True
    log.info("Loaded fine-tuned weights from %s", WEIGHTS_PATH)
except (FileNotFoundError, RuntimeError) as exc:
    log.warning("Could not load fine-tuned weights (%s); predictions are NOT meaningful.", exc)
model.eval()

grad_cam = GradCAMEngine(model, model.layer4[-1])
# warm-up so the first request isn't slow
grad_cam.generate(torch.zeros(1, 3, 224, 224, device=device))


def _load_metrics() -> dict:
    """Metrics + threshold from evaluate.py."""
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return {}


METRICS = _load_metrics()
DECISION_THRESHOLD = float(METRICS.get("selected_threshold_from_val", 0.5))
MODEL_CARD = ModelCard(
    architecture="ResNet50 (ImageNet init, fine-tuned)",
    dataset="PneumoniaMNIST 224x224",
    test_auroc=METRICS.get("test_auroc"),
    test_sensitivity=METRICS.get("test_at_selected", {}).get("sensitivity"),
    test_specificity=METRICS.get("test_at_selected", {}).get("specificity"),
)
log.info("Decision threshold: %.3f", DECISION_THRESHOLD)

# keep identical to the val transform (no train/serve skew)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

tracker = SpatialTracker(video=True)  # camera stream


class PayloadError(ValueError):
    def __init__(self, msg, req_id=None):
        super().__init__(msg)
        self.req_id = req_id


def _salvage_id(payload_str: str):
    """Best-effort id for error replies on invalid requests."""
    try:
        rid = json.loads(payload_str).get("id")
        return rid if isinstance(rid, str) and len(rid) <= 64 else None
    except (ValueError, AttributeError):
        return None


def _decode_image(payload_str: str):
    if len(payload_str) > MAX_PAYLOAD_BYTES:
        raise PayloadError("Payload too large")
    try:
        req = StreamRequest.model_validate_json(payload_str)
    except ValidationError as exc:
        err = exc.errors()[0]
        field = ".".join(str(p) for p in err["loc"]) or "body"
        raise PayloadError(f"Invalid request ({field}): {err['msg']}", _salvage_id(payload_str))
    b64_data = req.image
    if "," in b64_data:
        b64_data = b64_data.split(",", 1)[1]
    try:
        pil = Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("RGB")
    except (binascii.Error, UnidentifiedImageError, OSError):
        raise PayloadError("Could not decode image", req.id)
    return pil, req.mode, req.id


def _encode(img_bgr, fmt):
    ok, buf = cv2.imencode(f".{fmt}", img_bgr)
    if not ok:
        raise RuntimeError("Overlay encoding failed")
    mime = "image/png" if fmt == "png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(buf).decode('utf-8')}"


def infer_batch(tensors: list) -> list:
    """Grad-CAM + logit for a micro-batch in one forward/backward."""
    cams, logits = grad_cam.generate_batch(torch.stack(tensors).to(device))
    return list(zip(cams, logits.tolist()))


def run_medical(pil: Image.Image, t0: float) -> MedicalResult:
    """Unbatched path (scripts, tests)."""
    heatmap, logit = infer_batch([transform(pil)])[0]
    return medical_result(pil, heatmap, logit, t0)


def medical_result(pil: Image.Image, heatmap: np.ndarray, logit: float, t0: float) -> MedicalResult:
    img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    prob = float(torch.sigmoid(torch.tensor(logit)))

    positive = prob >= DECISION_THRESHOLD
    diagnosis = "PNEUMONIA SUSPECTED" if positive else "NO PNEUMONIA DETECTED"

    # upscale cam to native res
    heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * np.clip(heatmap, 0, 1)), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, 0.4, img_bgr, 0.6, 0)

    return MedicalResult(
        status=f"{diagnosis} | P(pneumonia) = {prob:.3f} (threshold {DECISION_THRESHOLD:.2f})",
        confidence=round(100 * (prob if positive else 1 - prob), 1),
        probability=prob,
        logit=logit,
        threshold=DECISION_THRESHOLD,
        prediction=int(positive),
        overlay=_encode(overlay, "png"),
        latency_ms=_ms_since(t0),
    )


def run_spatial(pil: Image.Image, t0: float) -> SpatialResult:
    rgb = np.array(pil)
    face_pts, hands_pts = tracker.process(rgb)
    overlay = draw(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), face_pts, hands_pts, face_step=5)
    return SpatialResult(
        status=f"Spatial tracking | face: {'yes' if face_pts else 'no'} | hands: {len(hands_pts)}",
        landmarks=Landmarks(face=face_pts, hands=hands_pts),
        overlay=_encode(overlay, "jpg"),
        latency_ms=_ms_since(t0),
    )


def _ms_since(t0: float) -> float:
    return round(1000 * (time.perf_counter() - t0), 1)


batcher = MicroBatcher(infer_batch, max_batch=MAX_BATCH, max_wait_ms=MAX_WAIT_MS)


async def handle(payload_str: str) -> dict:
    t0 = time.perf_counter()
    try:
        pil, mode, req_id = await asyncio.to_thread(_decode_image, payload_str)
    except PayloadError as exc:
        return ErrorReply(id=exc.req_id, error=str(exc), status=f"Error: {exc}").model_dump()
    try:
        if mode == "spatial":
            result = await asyncio.to_thread(run_spatial, pil, t0)
        else:
            tensor = await asyncio.to_thread(transform, pil)
            heatmap, logit = await batcher.submit(tensor)  # shares a gpu pass with concurrent requests
            result = await asyncio.to_thread(medical_result, pil, heatmap, logit, t0)
    except Exception:
        log.exception("Pipeline failure")
        return ErrorReply(id=req_id, error="internal", status="Error: processing failed").model_dump()
    result.id = req_id
    return result.model_dump()


@asynccontextmanager
async def lifespan(_app):
    batcher.start()
    yield
    await batcher.stop()


app = FastAPI(title="Multimodal Medical Streaming Engine", lifespan=lifespan)


# routes
@app.get("/")
def index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"), headers={"Cache-Control": "no-cache"})


@app.get("/health", response_model=Health)
def health():
    return Health(device=str(device), weights_loaded=weights_loaded, decision_threshold=DECISION_THRESHOLD,
                  model=MODEL_CARD, mean_batch_size=batcher.mean_batch_size)


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """Handles a socket's messages concurrently so batches share gpu passes."""
    await websocket.accept()
    slots = asyncio.Semaphore(MAX_IN_FLIGHT)
    send_lock = asyncio.Lock()
    tasks = set()

    async def serve(raw):
        try:
            reply = await handle(raw)
        finally:
            slots.release()
        async with send_lock:
            try:
                await websocket.send_json(reply)
            except (WebSocketDisconnect, RuntimeError):
                pass  # client left mid-batch

    try:
        while True:
            raw = await websocket.receive_text()
            await slots.acquire()
            task = asyncio.create_task(serve(raw))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
    except WebSocketDisconnect:
        for task in tasks:
            task.cancel()
        log.info("WebSocket client disconnected.")
