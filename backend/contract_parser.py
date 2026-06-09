import re
from models import CustomerProfile

def parse_contract(text: str, customer_name: str) -> CustomerProfile:
    profile = CustomerProfile(
        customer=customer_name,
        region="global",
        latency_target_ms=2000,
        blocked_providers=[],
        preferred_providers=[],
        cost_sensitivity="balanced",
        routing_tiers={
            "simple": {"models": ["GPT-5.4 nano", "Gemini Flash Lite"], "max_cost": 0.001},
            "medium": {"models": ["GPT-5.4 mini", "Claude Sonnet 4.6"], "max_cost": 0.005},
            "complex": {"models": ["GPT-5.4", "Claude Opus 4.6"], "max_cost": 0.02}
        }
    )

    text_lower = text.lower()

    region_patterns = {
        "eu": r"eu|europe|gdpr|general data protection",
        "us": r"us|united states|america|usa|ccpa",
        "india": r"india|in$|digital personal data",
        "global": r"global|worldwide|international"
    }
    for region, pattern in region_patterns.items():
        if re.search(pattern, text_lower):
            profile.region = region
            break

    latency_match = re.search(r'(\d+)\s*ms', text_lower)
    if latency_match:
        profile.latency_target_ms = int(latency_match.group(1))

    provider_names = {
        "deepseek": "DeepSeek",
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "google": "Google",
        "xai": "xAI",
        "meta": "Meta",
        "mistral": "Mistral"
    }

    for key, name in provider_names.items():
        if re.search(rf'(?:do not|cannot|must not|prohibit|forbid|block|avoid|ban).{{0,50}}{key}', text_lower):
            profile.blocked_providers.append(name)

    for key, name in provider_names.items():
        if re.search(rf'(?:prefer|require|must use|only use|recommend).{{0,50}}{key}', text_lower):
            profile.preferred_providers.append(name)

    if re.search(r'cost.{0,20}(?:critical|primary|most important|top priority)', text_lower):
        profile.cost_sensitivity = "high"
    elif re.search(r'quality.{0,20}(?:critical|primary|most important)', text_lower):
        profile.cost_sensitivity = "low"

    return profile
