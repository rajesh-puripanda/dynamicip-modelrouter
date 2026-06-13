from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from models import ChatRequest, ChatResponse, ContractInput
from classifier import classify_prompt, select_model, estimate_costs, MODELS, PREMIUM_BASELINE_COST
from contract_parser import parse_contract
from dashboard import router as dashboard_router
import uvicorn
import time
import json
import os
from pathlib import Path

app = FastAPI(title="ModelRouter API", version="1.0.0")
app.include_router(dashboard_router)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

customers = {}
routing_log = []

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


@app.get("/")
async def root():
    return {
        "service": "ModelRouter",
        "version": "1.0.0",
        "status": "routing",
        "models_available": sum(len(v["models"]) for v in MODELS.values()),
        "customers_served": len(customers),
        "total_routes": len(routing_log)
    }


@app.get("/v1/models")
async def list_models():
    all_models = []
    for tier_name, tier_data in MODELS.items():
        for m in tier_data["models"]:
            all_models.append({
                "id": m["name"].lower().replace(" ", "-").replace(".", ""),
                "name": m["name"],
                "tier": tier_name,
                "provider": m["provider"],
                "cost_per_1k": m["cost_per_1k"],
                "latency_ms": m["latency_ms"]
            })
    return {"models": all_models}


@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    if not request.messages or not request.messages[-1].content.strip():
        raise HTTPException(status_code=400, detail="messages[last].content is required")

    prompt = request.messages[-1].content
    start = time.time()

    classification_result = classify_prompt(prompt)
    classification = classification_result["classification"]
    confidence = classification_result["confidence"]

    profile = customers.get("default")
    route = select_model(classification, confidence, profile)
    costs = estimate_costs(prompt, classification)

    routing_path = [
        "Prompt In",
        f"Classify → {classification} ({confidence:.0%})",
        "Filter Policy",
        "Score & Route",
        f"Response ← {route['model_name']}"
    ]

    premium_vs_current = PREMIUM_BASELINE_COST - route["cost_per_1k"]

    response_text = _generate_response(prompt, classification, route)

    log_entry = {
        "prompt": prompt[:120],
        "classification": classification,
        "confidence": confidence,
        "model": route["model_name"],
        "provider": route["provider"],
        "cost_savings": route["cost_savings_vs_premium"],
        "latency_ms": int((time.time() - start) * 1000),
        "timestamp": time.time(),
    }
    routing_log.append(log_entry)

    with open(DATA_DIR / "routing_log.json", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return ChatResponse(
        model=route["model_name"],
        provider=route["provider"],
        classification=classification,
        cost_savings=round(premium_vs_current / PREMIUM_BASELINE_COST * 100, 1),
        response=response_text,
        routing_path=routing_path
    )


SYSTEM_PROMPTS = {
    "simple": "You are a helpful assistant. Answer briefly in 1-2 sentences. Be direct and useful. Do not explain what you cannot do.",
    "medium": "You are a helpful assistant. Give a concise but complete answer in 2-4 sentences. Focus on the key points.",
    "complex": "You are a senior analyst. Provide a structured, thorough analysis. Use bullet points where helpful. Be precise and detailed."
}

OPENROUTER_MODELS = {
    "simple": "openai/gpt-4o-mini",
    "medium": "openai/gpt-4o",
    "complex": "anthropic/claude-3.5-sonnet",
}


def _call_openrouter(prompt: str, classification: str, route: dict) -> str:
    import httpx
    api_key = os.getenv("OPENROUTER_API_KEY")
    or_model = OPENROUTER_MODELS.get(classification, "openai/gpt-4o-mini")
    system_prompt = SYSTEM_PROMPTS.get(classification, SYSTEM_PROMPTS["simple"])

    try:
        resp = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": or_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 256,
                "temperature": 0.3,
            },
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        return f"[OpenRouter error {resp.status_code}: {resp.text[:200]}]"
    except Exception as e:
        return f"[OpenRouter connection failed: {e}]"


def _generate_response(prompt: str, classification: str, route: dict) -> str:
    lines = [
        f"[{route['model_name']} via {route['provider']}]",
        "",
        f"Classification: {classification} (confidence: {route['confidence']:.0%})",
        f"Cost: ${route['cost_per_1k']}/1K tokens vs ${PREMIUM_BASELINE_COST} premium",
        f"Savings: {route['cost_savings_vs_premium']}%",
    ]

    if os.getenv("OPENROUTER_API_KEY"):
        return "\n".join(lines) + "\n\n" + _call_openrouter(prompt, classification, route)
    else:
        lines.append("")
        lines.append("Your prompt was classified as '{0}' and routed to {1}.".format(classification, route["model_name"]))
        lines.append("Set OPENROUTER_API_KEY env var for real LLM responses.")
        return "\n".join(lines)


@app.post("/v1/contract/parse")
async def parse_contract_endpoint(contract: ContractInput):
    customer_id = contract.customer_name.lower().replace(" ", "-")
    profile = parse_contract(contract.text, customer_id)
    customers[customer_id] = profile

    persistence_path = DATA_DIR / f"profile_{customer_id}.json"
    with open(persistence_path, "w") as f:
        f.write(profile.model_dump_json(indent=2))

    return {
        "customer_id": customer_id,
        "profile": profile.model_dump(),
        "endpoint": f"/{customer_id}/v1/chat/completions",
        "status": "active"
    }


@app.get("/v1/customers")
async def list_customers():
    return {"customers": list(customers.keys()), "total": len(customers)}


@app.get("/v1/stats")
async def get_stats():
    total = len(routing_log)
    if total == 0:
        return {"total_routes": 0}

    classifications = {}
    for entry in routing_log:
        cls = entry["classification"]
        classifications[cls] = classifications.get(cls, 0) + 1

    savings = [e["cost_savings"] for e in routing_log if e.get("cost_savings")]
    avg_savings = round(sum(savings) / len(savings), 1) if savings else 0

    return {
        "total_routes": total,
        "classifications": classifications,
        "avg_cost_savings_pct": avg_savings,
        "total_savings_estimate": round(sum(savings) / 100 * 0.015, 4) if savings else 0,
        "models_used": list(set(e["model"] for e in routing_log))
    }


@app.get("/v1/logs")
async def get_logs(limit: int = 50):
    return {"logs": routing_log[-limit:]}


if __name__ == "__main__":
    print("=" * 50)
    print("  ModelRouter Engine v1.0")
    print("  Intelligent LLM Query Routing System")
    print("=" * 50)
    print(f"\n* Models: {sum(len(v['models']) for v in MODELS.values())} across 3 tiers")
    print(f"* Endpoint: http://localhost:8001/v1/chat/completions")
    print(f"* Dashboard: http://localhost:8001/dashboard")
    print(f"* Docs: http://localhost:8001/docs")
    print("\n* Router active. Waiting for requests...\n")
    uvicorn.run(app, host="0.0.0.0", port=8001)
