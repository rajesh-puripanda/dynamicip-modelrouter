from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from models import ChatRequest, ChatResponse, ContractInput, RouteDecision
from classifier import classify_prompt, select_model, estimate_costs, MODELS, PREMIUM_BASELINE_COST
from contract_parser import parse_contract
from dashboard import router as dashboard_router
import uvicorn
import time
import json
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


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    prompt = request.messages[-1].content if request.messages else ""
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

    decision = RouteDecision(
        classification=classification,
        confidence=confidence,
        selected_model=route["model_name"],
        provider=route["provider"],
        estimated_cost=route["cost_per_1k"],
        estimated_latency_ms=route["estimated_latency_ms"],
        reason=route["reason"]
    )

    log_entry = {
        "prompt": prompt[:100],
        "classification": classification,
        "confidence": confidence,
        "model": route["model_name"],
        "provider": route["provider"],
        "cost_savings": route["cost_savings_vs_premium"],
        "latency_ms": int((time.time() - start) * 1000),
        "timestamp": time.time()
    }
    routing_log.append(log_entry)

    with open(DATA_DIR / "routing_log.json", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    premium_vs_current = PREMIUM_BASELINE_COST - route["cost_per_1k"]

    return ChatResponse(
        model=route["model_name"],
        provider=route["provider"],
        classification=classification,
        cost_savings=round(premium_vs_current / PREMIUM_BASELINE_COST * 100, 1),
        response=f"[{route['model_name']} via {route['provider']}]\n\n"
                 f"Classification: {classification} (confidence: {confidence:.0%})\n"
                 f"Cost: ${route['cost_per_1k']}/1K tokens vs ${PREMIUM_BASELINE_COST} premium\n"
                 f"Savings: {route['cost_savings_vs_premium']}%\n\n"
                 f"Your prompt was classified as '{classification}' and routed to {route['model_name']}.\n"
                 f"This saves {(costs['savings_amount']*1000):.2f}¢ per 1K tokens vs always using a premium model.\n\n"
                 f"[This is a simulated response — connect OpenRouter for real LLM output]",
        routing_path=routing_path
    )


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
    print(f"* Endpoint: http://localhost:8000/v1/chat/completions")
    print(f"* Dashboard: http://localhost:8000/dashboard")
    print(f"* Docs: http://localhost:8000/docs")
    print("\n* Router active. Waiting for requests...\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
