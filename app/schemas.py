from pydantic import BaseModel
from typing import List, Optional, Dict

class PredictRequest(BaseModel):
    title: Optional[str] = None
    text: str

class BiasOut(BaseModel):
    label: str
    confidence: float
    probs: Optional[Dict[str, float]] = None

class RationaleSpan(BaseModel):
    start: int
    end: int
    text: str

class ExplainOut(BaseModel):
    spans: List[RationaleSpan] = []

class PredictResponse(BaseModel):
    summary: str
    bias: BiasOut
    explain: ExplainOut
    source_prior: Optional[Dict[str, str]] = None   # e.g., {"source":"cnn.com","rating":"Left","origin":"AllSides"}

class EnvResponse(BaseModel):
    device: str
    torch: str
    cuda_available: bool
    cuda_version: Optional[str] = None
