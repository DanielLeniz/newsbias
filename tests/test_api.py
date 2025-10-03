# tests/test_api.py
from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json().get("ok") is True

def test_predict_minimal():
    r = client.post("/predict", json={"text": "Lawmakers reached a bipartisan deal after talks."})
    assert r.status_code == 200
    body = r.json()
    assert "bias" in body and body["bias"]["label"] in {"Left","Center","Right"}
    assert "summary" in body

def test_batch_predict_array():
    payload = [
        {"text":"War on business with heavy regulation."},
        {"text":"Bipartisan talks concluded."}
    ]
    r = client.post("/batch_predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list) and len(data) == 2
