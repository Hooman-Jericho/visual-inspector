"""FastAPI application for PatchCore industrial defect detection serving."""

from __future__ import annotations

import base64
import io
import logging
import os
import time
import uuid
from typing import Any

import torchvision.transforms as T
from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from visual_inspector.serve.registry import ModelRegistry
from visual_inspector.viz.heatmap import create_heatmap_overlay

# Maximum image pixel guard to prevent decompression bombs
Image.MAX_IMAGE_PIXELS = 20_000_000
MAX_UPLOAD_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB

logger = logging.getLogger("visual_inspector.serve")

app = FastAPI(
    title="Visual Inspector API",
    description="Unsupervised Industrial Defect Detection powered by PatchCore (Roth et al., CVPR 2022).",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Optional CORS
cors_env = os.environ.get("CORS_ORIGINS")
if cors_env:
    origins = [orig.strip() for orig in cors_env.split(",") if orig.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

registry = ModelRegistry(artifacts_dir=os.environ.get("ARTIFACTS_DIR", "artifacts"))


# Middleware for Request-ID & latency logging
@app.middleware("http")
async def request_id_and_logging_middleware(request: Request, call_next: Any) -> Response:
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start_time = time.perf_counter()

    response: Response = await call_next(request)

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Request-ID"] = req_id
    response.headers["X-Response-Time-Ms"] = f"{latency_ms:.2f}"
    return response


# Response Schemas
class HealthResponse(BaseModel):
    status: str = Field("ok", examples=["ok"])
    version: str = Field("0.1.0", examples=["0.1.0"])
    loaded_categories: list[str] = Field(default_factory=list, examples=[["bottle", "carpet"]])


class PredictResponse(BaseModel):
    category: str = Field(..., examples=["bottle"])
    score: float = Field(..., examples=[0.842])
    threshold: float = Field(..., examples=[0.750])
    is_anomalous: bool = Field(..., examples=[True])
    latency_ms: float = Field(..., examples=[18.4])
    image_size: list[int] = Field(..., examples=[[224, 224]])
    heatmap_png_base64: str = Field(..., description="Base64-encoded PNG heatmap overlay")
    model_version: str = Field("0.1.0", examples=["0.1.0"])


def _validate_and_read_image(content: bytes, content_type: str | None) -> Image.Image:
    """Validate file size, content type, and integrity."""
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file exceeds maximum size limit of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB.",
        )

    valid_types = {"image/png", "image/jpeg", "image/jpg"}
    if content_type and content_type.lower() not in valid_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{content_type}'. Supported formats: image/png, image/jpeg.",
        )

    try:
        raw_img = Image.open(io.BytesIO(content))
        raw_img.verify()  # verify integrity
        # Re-open after verify
        return Image.open(io.BytesIO(content)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Corrupted or invalid image file: {exc}",
        ) from exc


@app.get("/health", response_model=HealthResponse, tags=["System"])
def get_health() -> HealthResponse:
    """System health check and loaded models list."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        loaded_categories=registry.loaded_category_names(),
    )


@app.get("/categories", tags=["System"])
def get_categories() -> dict[str, Any]:
    """List supported defect detection categories and their status."""
    return {"categories": registry.list_categories()}


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict_defect(
    category: str = Query(..., description="MVTec category name, e.g. 'bottle'"),
    file: UploadFile = File(..., description="Inspection image file (PNG/JPEG)"),  # noqa: B008
) -> PredictResponse:
    """Predict defect score and generate heatmap overlay."""
    try:
        model = registry.get_model(category)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    content = await file.read()
    img = _validate_and_read_image(content, file.content_type)

    t0 = time.perf_counter()

    # Preprocess image
    crop_size = model.config.data.crop_size
    transform = T.Compose(
        [
            T.Resize(model.config.data.image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor_img = transform(img).unsqueeze(0)

    # Score
    scores, anomaly_maps = model.score(tensor_img)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    score = scores[0]
    threshold = model.threshold
    is_anom = bool(score > threshold)

    # Generate heatmap overlay
    overlay = create_heatmap_overlay(img, anomaly_maps[0])
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

    return PredictResponse(
        category=category,
        score=round(score, 4),
        threshold=round(threshold, 4),
        is_anomalous=is_anom,
        latency_ms=round(latency_ms, 2),
        image_size=[img.width, img.height],
        heatmap_png_base64=b64_str,
        model_version="0.1.0",
    )


@app.post("/predict/heatmap", tags=["Inference"])
async def predict_heatmap_image(
    category: str = Query(..., description="MVTec category name"),
    file: UploadFile = File(..., description="Inspection image file"),  # noqa: B008
) -> Response:
    """Returns pure image/png heatmap overlay response."""
    try:
        model = registry.get_model(category)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    content = await file.read()
    img = _validate_and_read_image(content, file.content_type)

    transform = T.Compose(
        [
            T.Resize(model.config.data.image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(model.config.data.crop_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor_img = transform(img).unsqueeze(0)
    _, anomaly_maps = model.score(tensor_img)

    overlay = create_heatmap_overlay(img, anomaly_maps[0])
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


# Single-file HTML drag-and-drop demo
DEMO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Visual Inspector — Industrial Defect Detection Demo</title>
<style>
  :root {
    --bg-primary: #0f172a;
    --bg-card: rgba(30, 41, 59, 0.7);
    --border: rgba(148, 163, 184, 0.15);
    --accent: #38bdf8;
    --accent-hover: #0284c7;
    --text: #f8fafc;
    --text-muted: #94a3b8;
    --success: #10b981;
    --danger: #ef4444;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg-primary);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 2rem 1rem;
  }
  header {
    text-align: center;
    max-width: 800px;
    margin-bottom: 2rem;
  }
  h1 { font-size: 2.2rem; font-weight: 700; color: #fff; margin-bottom: 0.5rem; letter-spacing: -0.5px; }
  .badge { display: inline-block; padding: 4px 12px; background: rgba(56, 189, 248, 0.15); color: var(--accent); border-radius: 9999px; font-size: 0.85rem; font-weight: 600; margin-bottom: 0.75rem; border: 1px solid rgba(56, 189, 248, 0.3); }
  p.subtitle { color: var(--text-muted); font-size: 1.05rem; }
  .main-card {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    width: 100%;
    max-width: 900px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.4);
  }
  .controls {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
    align-items: center;
  }
  label { font-size: 0.95rem; font-weight: 500; color: var(--text-muted); }
  select {
    background: #1e293b;
    border: 1px solid var(--border);
    color: #fff;
    padding: 8px 14px;
    border-radius: 8px;
    font-size: 0.95rem;
    outline: none;
    cursor: pointer;
  }
  .drop-zone {
    border: 2px dashed rgba(56, 189, 248, 0.4);
    background: rgba(15, 23, 42, 0.4);
    border-radius: 12px;
    padding: 2.5rem 1.5rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  .drop-zone:hover, .drop-zone.dragover {
    border-color: var(--accent);
    background: rgba(56, 189, 248, 0.05);
  }
  .drop-zone input { display: none; }
  .drop-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 0.25rem; }
  .drop-sub { font-size: 0.85rem; color: var(--text-muted); }
  .results-panel {
    display: none;
    margin-top: 2rem;
    padding-top: 2rem;
    border-top: 1px solid var(--border);
  }
  .verdict-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
  }
  .verdict-tag {
    font-size: 1.25rem;
    font-weight: 700;
    padding: 6px 18px;
    border-radius: 9999px;
    text-transform: uppercase;
  }
  .tag-pass { background: rgba(16, 185, 129, 0.2); color: var(--success); border: 1px solid var(--success); }
  .tag-fail { background: rgba(239, 68, 68, 0.2); color: var(--danger); border: 1px solid var(--danger); }
  .meta-stats { display: flex; gap: 1.5rem; font-size: 0.95rem; color: var(--text-muted); }
  .stat-val { color: #fff; font-weight: 600; }
  .image-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
  }
  .img-box {
    background: #0f172a;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem;
    text-align: center;
  }
  .img-box img { max-width: 100%; height: auto; border-radius: 6px; display: block; margin: 0 auto; }
  .img-box span { display: block; margin-top: 0.5rem; font-size: 0.85rem; color: var(--text-muted); }
  .loader { display: none; margin: 1.5rem auto 0; text-align: center; color: var(--accent); font-weight: 600; }
</style>
</head>
<body>
  <header>
    <div class="badge">PatchCore Inspection Engine</div>
    <h1>Industrial Defect Inspector</h1>
    <p class="subtitle">Unsupervised pixel-level defect localization on MVTec AD benchmark</p>
  </header>

  <div class="main-card">
    <div class="controls">
      <label for="categorySelect">Inspection Category:</label>
      <select id="categorySelect">
        <option value="bottle" selected>bottle</option>
        <option value="carpet">carpet</option>
        <option value="hazelnut">hazelnut</option>
        <option value="screw">screw</option>
        <option value="transistor">transistor</option>
        <option value="synthetic">synthetic (demo)</option>
      </select>
    </div>

    <div class="drop-zone" id="dropZone">
      <input type="file" id="fileInput" accept="image/png, image/jpeg">
      <div class="drop-title">Drop an inspection image here, or click to browse</div>
      <div class="drop-sub">PNG or JPEG format (max 5 MB)</div>
    </div>

    <div class="loader" id="loader">Running PatchCore embedding & kNN nearest neighbor search...</div>

    <div class="results-panel" id="resultsPanel">
      <div class="verdict-header">
        <div class="verdict-tag" id="verdictTag">NORMAL</div>
        <div class="meta-stats">
          <div>Score: <span class="stat-val" id="scoreVal">0.000</span></div>
          <div>Threshold: <span class="stat-val" id="threshVal">0.000</span></div>
          <div>Inference: <span class="stat-val" id="latencyVal">0 ms</span></div>
        </div>
      </div>
      <div class="image-grid">
        <div class="img-box">
          <img id="origImg" src="" alt="Original">
          <span>Input Image</span>
        </div>
        <div class="img-box">
          <img id="overlayImg" src="" alt="Overlay">
          <span>Defect Localization Heatmap</span>
        </div>
      </div>
    </div>
  </div>

<script>
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const catSelect = document.getElementById('categorySelect');
  const loader = document.getElementById('loader');
  const resultsPanel = document.getElementById('resultsPanel');
  const origImg = document.getElementById('origImg');
  const overlayImg = document.getElementById('overlayImg');
  const verdictTag = document.getElementById('verdictTag');
  const scoreVal = document.getElementById('scoreVal');
  const threshVal = document.getElementById('threshVal');
  const latencyVal = document.getElementById('latencyVal');

  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
  });

  async function handleFile(file) {
    origImg.src = URL.createObjectURL(file);
    loader.style.display = 'block';
    resultsPanel.style.display = 'none';

    const formData = new FormData();
    formData.append('file', file);
    const category = catSelect.value;

    try {
      const resp = await fetch(`/predict?category=${category}`, {
        method: 'POST',
        body: formData
      });
      if (!resp.ok) {
        const err = await resp.json();
        alert(`Error (${resp.status}): ${err.detail || 'Prediction failed'}`);
        return;
      }
      const data = await resp.json();
      scoreVal.textContent = data.score.toFixed(4);
      threshVal.textContent = data.threshold.toFixed(4);
      latencyVal.textContent = data.latency_ms.toFixed(1) + ' ms';

      if (data.is_anomalous) {
        verdictTag.textContent = 'DEFECT DETECTED';
        verdictTag.className = 'verdict-tag tag-fail';
      } else {
        verdictTag.textContent = 'NORMAL / PASS';
        verdictTag.className = 'verdict-tag tag-pass';
      }
      overlayImg.src = 'data:image/png;base64,' + data.heatmap_png_base64;
      resultsPanel.style.display = 'block';
    } catch (err) {
      alert('Network error connecting to API: ' + err.message);
    } finally {
      loader.style.display = 'none';
    }
  }
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, tags=["Demo"])
def get_demo_ui() -> HTMLResponse:
    """Single-file interactive defect detection demo UI."""
    return HTMLResponse(content=DEMO_HTML)
