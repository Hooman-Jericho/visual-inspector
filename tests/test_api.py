"""API endpoint tests using FastAPI TestClient."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from visual_inspector.serve.app import MAX_UPLOAD_SIZE_BYTES, app


@pytest.fixture
def client() -> TestClient:
    """Create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def dummy_png_bytes() -> bytes:
    """Generate valid small test PNG image bytes."""
    img = Image.new("RGB", (64, 64), color=(120, 150, 180))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestSystemEndpoints:
    def test_health_check(self, client: TestClient) -> None:
        """GET /health returns status ok, version, and loaded categories."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"
        assert isinstance(data["loaded_categories"], list)
        assert "X-Request-ID" in response.headers
        assert "X-Response-Time-Ms" in response.headers

    def test_categories_list(self, client: TestClient) -> None:
        """GET /categories returns list with known categories."""
        response = client.get("/categories")
        assert response.status_code == 200
        cats = response.json()["categories"]
        cat_names = [c["category"] for c in cats]
        assert "bottle" in cat_names
        assert "synthetic" in cat_names

    def test_demo_ui_page(self, client: TestClient) -> None:
        """GET / returns HTML demo interface."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Industrial Defect Inspector" in response.text


class TestPredictEndpoint:
    def test_predict_happy_path(self, client: TestClient, dummy_png_bytes: bytes) -> None:
        """POST /predict on trained category returns full verdict and heatmap."""
        files = {"file": ("test.png", dummy_png_bytes, "image/png")}
        response = client.post("/predict?category=synthetic", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["category"] == "synthetic"
        assert isinstance(data["score"], float)
        assert isinstance(data["threshold"], float)
        assert isinstance(data["is_anomalous"], bool)
        assert isinstance(data["latency_ms"], float)
        assert "heatmap_png_base64" in data
        assert len(data["heatmap_png_base64"]) > 50

    def test_predict_heatmap_raw_image(self, client: TestClient, dummy_png_bytes: bytes) -> None:
        """POST /predict/heatmap returns binary image/png."""
        files = {"file": ("test.png", dummy_png_bytes, "image/png")}
        response = client.post("/predict/heatmap?category=synthetic", files=files)
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert len(response.content) > 100

    def test_predict_wrong_media_type(self, client: TestClient) -> None:
        """POST /predict with non-image file type returns 415."""
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        response = client.post("/predict?category=synthetic", files=files)
        assert response.status_code == 415
        assert "Unsupported media type" in response.json()["detail"]

    def test_predict_corrupted_image(self, client: TestClient) -> None:
        """POST /predict with corrupted image data returns 422."""
        files = {
            "file": ("corrupted.png", b"\x89PNG\r\n\x1a\ncorrupted_garbage_bytes", "image/png")
        }
        response = client.post("/predict?category=synthetic", files=files)
        assert response.status_code == 422
        assert "Corrupted or invalid image" in response.json()["detail"]

    def test_predict_unknown_category(self, client: TestClient, dummy_png_bytes: bytes) -> None:
        """POST /predict with unknown category returns 404."""
        files = {"file": ("test.png", dummy_png_bytes, "image/png")}
        response = client.post("/predict?category=non_existent_part", files=files)
        assert response.status_code == 404
        assert "Unknown category" in response.json()["detail"]

    def test_predict_untrained_category(self, client: TestClient, dummy_png_bytes: bytes) -> None:
        """POST /predict with recognized but untrained category returns 409 with hint."""
        files = {"file": ("test.png", dummy_png_bytes, "image/png")}
        response = client.post("/predict?category=toothbrush", files=files)
        assert response.status_code == 409
        assert "not trained yet" in response.json()["detail"]
        assert "Hint:" in response.json()["detail"]

    def test_predict_oversized_file(self, client: TestClient) -> None:
        """POST /predict with file > 5 MB returns 413."""
        oversized = b"0" * (MAX_UPLOAD_SIZE_BYTES + 1024)
        files = {"file": ("huge.png", oversized, "image/png")}
        response = client.post("/predict?category=synthetic", files=files)
        assert response.status_code == 413
        assert "exceeds maximum size limit" in response.json()["detail"]
