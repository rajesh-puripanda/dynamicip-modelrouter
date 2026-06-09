from pydantic import BaseModel
from typing import Optional, List

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "auto"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024

class ChatResponse(BaseModel):
    model: str
    provider: str
    classification: str
    cost_savings: float
    response: str
    routing_path: List[str]

class ContractInput(BaseModel):
    text: str
    customer_name: str

class CustomerProfile(BaseModel):
    customer: str
    region: str
    latency_target_ms: int
    blocked_providers: List[str]
    preferred_providers: List[str]
    cost_sensitivity: str
    routing_tiers: dict

class RouteDecision(BaseModel):
    classification: str
    confidence: float
    selected_model: str
    provider: str
    estimated_cost: float
    estimated_latency_ms: int
    reason: str
