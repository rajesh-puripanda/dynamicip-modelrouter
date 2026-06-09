import re
import math

SIMPLE_KEYWORDS = [
    "hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye",
    "what time", "what is your name", "who are you",
    "help", "how are you", "good morning", "good evening",
    "yes", "no", "ok", "sure", "please",
    "what is the weather", "tell me a joke", "say hello",
    "how do i", "where is", "can you repeat"
]

COMPLEX_PATTERNS = [
    r"analyze", r"compare and contrast", r"comprehensive", r"detailed analysis",
    r"implications of", r"risk assessment", r"legal", r"liability",
    r"multi-step", r"reasoning", r"complex", r"sophisticated",
    r"code.?review", r"debug", r"architecture", r"design.?pattern",
    r"mathematical proof", r"theorem", r"derivation",
    r"contract", r"obligation", r"jurisdiction", r"compliance",
    r"financial model", r"forecast", r"projection",
    r"research paper", r"literature review", r"methodology",
    r"optimize", r"optimization", r"trade-off",
    r"generate.*diagram", r"create.*plan", r"strategy",
    r"differential", r"calculus", r"statistical",
    r"neural network", r"deep learning", r"transformer"
]

MODELS = {
    "simple": {
        "models": [
            {"name": "GPT-5.4 nano", "provider": "OpenAI", "cost_per_1k": 0.00015, "latency_ms": 300},
            {"name": "Gemini Flash Lite", "provider": "Google", "cost_per_1k": 0.00010, "latency_ms": 280},
            {"name": "Claude Haiku 4.5", "provider": "Anthropic", "cost_per_1k": 0.00025, "latency_ms": 320},
        ]
    },
    "medium": {
        "models": [
            {"name": "GPT-5.4 mini", "provider": "OpenAI", "cost_per_1k": 0.0020, "latency_ms": 650},
            {"name": "Claude Sonnet 4.6", "provider": "Anthropic", "cost_per_1k": 0.0030, "latency_ms": 700},
            {"name": "Grok 4.1 Fast", "provider": "xAI", "cost_per_1k": 0.0015, "latency_ms": 600},
        ]
    },
    "complex": {
        "models": [
            {"name": "GPT-5.4", "provider": "OpenAI", "cost_per_1k": 0.015, "latency_ms": 1500},
            {"name": "Claude Opus 4.6", "provider": "Anthropic", "cost_per_1k": 0.018, "latency_ms": 1800},
            {"name": "Gemini 3.1 Pro", "provider": "Google", "cost_per_1k": 0.012, "latency_ms": 1400},
        ]
    }
}

PREMIUM_BASELINE_COST = 0.015


def classify_prompt(prompt: str) -> dict:
    prompt_lower = prompt.lower().strip()
    word_count = len(prompt.split())
    char_count = len(prompt)

    simple_score = 0
    for kw in SIMPLE_KEYWORDS:
        if kw in prompt_lower:
            simple_score += 1

    complexity_score = 0
    for pattern in COMPLEX_PATTERNS:
        if re.search(pattern, prompt_lower):
            complexity_score += 2

    if word_count > 100 or char_count > 600:
        complexity_score += 3
    if word_count > 50:
        complexity_score += 1

    has_code = bool(re.search(r'```|def |class |function |import |const |var |<[a-z]+>', prompt))
    if has_code:
        complexity_score += 2

    has_numbers = bool(re.search(r'\d+[.,]\d+|\d{4,}', prompt))
    has_math = bool(re.search(r'[+\-*/%=^∫∑√πθ]', prompt))
    if has_math and has_numbers:
        complexity_score += 2

    has_question = "?" in prompt
    if has_question and simple_score > 0 and complexity_score < 2:
        if word_count < 15:
            return {
                "classification": "simple",
                "confidence": 0.85,
                "score": simple_score - complexity_score
            }

    net_score = complexity_score - simple_score

    if net_score <= 0 and word_count < 20:
        classification = "simple"
        confidence = min(0.95, 0.7 + simple_score * 0.05)
    elif net_score >= 4 or has_code:
        classification = "complex"
        confidence = min(0.95, 0.7 + complexity_score * 0.05)
    elif word_count > 80:
        classification = "complex"
        confidence = 0.8
    else:
        classification = "medium"
        confidence = min(0.85, 0.6 + (net_score / 4) * 0.3)

    return {
        "classification": classification,
        "confidence": round(confidence, 2),
        "score": net_score
    }


def select_model(classification: str, confidence: float, profile: dict = None) -> dict:
    tier = MODELS[classification]
    models = tier["models"]

    selected = models[0]

    if profile:
        filtered = [m for m in models if m["provider"] not in profile.get("blocked_providers", [])]
        if filtered:
            models = filtered

        preferred = profile.get("preferred_providers", [])
        preferred_models = [m for m in models if m["provider"] in preferred]
        if preferred_models:
            models = preferred_models

        if profile.get("cost_sensitivity") == "high":
            models.sort(key=lambda m: m["cost_per_1k"])
        elif profile.get("cost_sensitivity") == "low":
            models.sort(key=lambda m: m["latency_ms"])
        else:
            models.sort(key=lambda m: m["cost_per_1k"] * m["latency_ms"])

        selected = models[0]

    premium_model = MODELS["complex"]["models"][0]
    cost_savings = round((premium_model["cost_per_1k"] - selected["cost_per_1k"]) / premium_model["cost_per_1k"] * 100, 1)

    return {
        "model_name": selected["name"],
        "provider": selected["provider"],
        "cost_per_1k": selected["cost_per_1k"],
        "estimated_latency_ms": selected["latency_ms"],
        "confidence": confidence,
        "cost_savings_vs_premium": cost_savings,
        "reason": f"Classified as {classification} (confidence: {confidence:.0%}) → routed to {selected['name']}"
    }


def estimate_costs(prompt: str, classification: str) -> dict:
    word_count = len(prompt.split())
    premium_cost = (word_count / 1000) * PREMIUM_BASELINE_COST
    tier_cost = (word_count / 1000) * MODELS[classification]["models"][0]["cost_per_1k"]
    savings = round((1 - tier_cost / premium_cost) * 100, 1) if premium_cost > 0 else 0

    return {
        "premium_cost": round(premium_cost, 6),
        "actual_cost": round(tier_cost, 6),
        "savings_percent": savings,
        "savings_amount": round(premium_cost - tier_cost, 6)
    }
