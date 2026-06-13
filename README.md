# ModelRouter 

**Intelligent LLM Query Routing System** — HackArena 2.0 (Online Zonals) by Ignite Room × IIIT Delhi

> **Team Dynamic IP** — Vardhan (Backend & AI), Rajesh (Frontend & Platform)

---

## The Pitch

**One URL change. Every model. Optimal choice. Zero code changes.**

ModelRouter is a drop-in replacement for your OpenAI-compatible API endpoint. It automatically classifies every incoming prompt by complexity (Simple / Medium / Complex) and routes it to the **cheapest capable model** that meets your constraints — saving 59%+ on inference costs without any code changes.

```python
# Before — hardcoded premium model, every request, forever
client = OpenAI(base_url="https://api.openai.com/v1")
response = client.chat.completions.create(model="gpt-5-4", ...)

# After ModelRouter — right model, every request, automatically
client = OpenAI(base_url="https://router.acme/v1")
response = client.chat.completions.create(model="auto", ...)
```

---

## Why ModelRouter?

### The Problem
- **30+** new LLMs ship every month — no team can evaluate them all
- **50–90%** of enterprise AI inference spend is wasted on over-provisioned models
- **180×** more energy per premium query vs a small model (39 Wh vs 0.22 Wh)
- Most teams pick one premium model and **never revisit** — 10–30× overpay

### The Solution
- **Fine-tuned 1.5B classifier** (Arch-Router-1.5B + GRPO) — 62ms classification
- **Contract-aware routing** — upload SLA docs, ModelRouter extracts constraints
- **OpenAI-compatible** — no SDK changes, no rewrites, no migration
- **3 routing tiers** — Simple (cheap, fast), Medium (balanced), Complex (powerful)

---

## Quick Start

### Prerequisites
- Python 3.11+
- pip

### 1. Clone & Setup
```bash
git clone https://github.com/rajesh-puripanda/dynamicip-modelrouter.git
cd dynamicip-modelrouter
pip install -r backend/requirements.txt
```

### 2. Run the Router
```bash
python backend/main.py
```

### 3. Use It
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="modelrouter-test"
)

response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "What is your return policy?"}]
)

print(response.choices[0].message.content)
```

### 4. Open the Dashboard
Visit **http://localhost:8001/dashboard** — live stats, route log, playground, and contract parser.

---

## API Reference

### `POST /v1/chat/completions`
Drop-in replacement for the OpenAI chat completions endpoint.

**Request:**
```json
{
  "model": "auto",
  "messages": [{"role": "user", "content": "What is your return policy?"}]
}
```

**Response:**
```json
{
  "model": "GPT-5.4 nano",
  "provider": "OpenAI",
  "classification": "simple",
  "cost_savings": 99.0,
  "response": "...",
  "routing_path": ["Prompt In", "Classify → simple (85%)", ...]
}
```

### `POST /v1/contract/parse`
Upload a contract/agreement → ModelRouter extracts routing constraints.

### `GET /v1/stats`
Real-time routing statistics.

### `GET /v1/logs`
Recent routing decisions.

---

## Architecture

```
                    ┌─────────────────────┐
                    │   Next.js Dashboard  │
                    │   (Port 3000)        │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │   FastAPI Backend    │
                    │   (Port 8001)        │
                    │                      │
                    │  ┌────────────────┐  │
                    │  │ Classifier      │  │  ← Arch-Router-1.5B (62ms)
                    │  │ Router Engine   │  │  ← Policy + Score + Route
                    │  │ Contract Parser │  │  ← LLM extraction
                    │  │ Provider Reg.   │  │  ← Model catalog
                    │  └────────────────┘  │
                    └─────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
     ┌────────▼───┐  ┌───────▼──────┐  ┌────▼────────┐
     │ OpenAI      │  │ Anthropic    │  │ Google      │
     │ GPT-5.4     │  │ Claude Opus  │  │ Gemini Pro  │
     │ GPT-5.4 nano│  │ Sonnet 4.6   │  │ Flash Lite  │
     └─────────────┘  └──────────────┘  └─────────────┘
```

### Tech Stack
| Component | Technology |
|-----------|-----------|
| Backend | Python, FastAPI, pydantic |
| Dashboard | HTML + CSS + JS (inline) |
| Classifier | Arch-Router-1.5B (GGUF Q8_0, 1.6GB) |
| LLM Access | OpenRouter (200+ models) |
| Fine-Tuning | GRPO via Unsloth + TRL |
| Deployment | Docker, Docker Compose |

---

## Demo Script (5 minutes)

1. **Welcome** — "ModelRouter is the lowest-friction AI optimization available today."
2. **The Problem** — "80% of your queries don't need GPT-5. You're paying 10-30× more."
3. **One Line Change** — Show the before/after code comparison.
4. **Route a Simple Query** — "What is your return policy?" → Classified: Simple → GPT-5.4 nano
5. **Route a Complex Query** — "Analyze liability exposure..." → Classified: Complex → Opus 4.6
6. **Parse a Contract** — Paste GDPR agreement → ModelRouter extracts EU-only, blocks DeepSeek
7. **Show Dashboard** — Cost savings, model distribution, live log
8. **Future Scope** — Vector search, provider health signals, managed SaaS

---

## Benchmarks (from our fine-tuned model)

| Tier | ModelRouter | Stock Router | Improvement |
|------|------------|-------------|------------|
| Simple | 81.8% | 87.9% | — |
| Medium | **85.7%** | 14.3% | **+71.4pp** 🎯 |
| Complex | 85.7% | 100% | — |
| **Overall** | **83.3%** | — | **59% cost savings** |

- GRPO fine-tune: **2.5 minutes** on RTX 3080 Laptop (8GB)
- Classification: **62ms** per query
- Dataset: 172 labeled prompts (train) + 54 (eval)

---

## Roadmap

### MVP Delivered
- Contract ingestion + LLM extraction pipeline
- Structured CustomerProfile JSON schema
- OpenAI-compatible proxy endpoint
- Arch-Router-1.5B GPU classifier + heuristic fallback
- Policy-filtered, objective-scored routing engine
- Full dashboard: stats, profiles, logs, playground

### Next Iterations
- Vector search over large contract doc sets
- Live provider health signals in routing
- Managed SaaS deployment (zero DevOps)
- Automatic model catalog updates

---

## Project Structure

```
dynamicip-modelrouter/
├── backend/
│   ├── main.py              # FastAPI server + API routes
│   ├── classifier.py        # Prompt classification engine
│   ├── contract_parser.py   # Contract → routing rules
│   ├── models.py            # Pydantic data models
│   ├── dashboard.py         # Dashboard route
│   ├── requirements.txt     # Python dependencies
│   └── Dockerfile           # Container build
├── frontend/
│   └── index.html           # Dashboard (inline CSS/JS)
├── data/                    # Runtime data (gitignored)
├── docker-compose.yml       # Orchestration
├── README.md                # This file
└── AGENTS.md                # Build log & agentic notes
```

---

## Links
- **GitHub**: https://github.com/rajesh-puripanda/dynamicip-modelrouter
- **HackArena 2.0**: Generative & Agentic AI — IIIT Delhi
- **Reference**: [ModelGate](https://github.com/Aaryan-Kapoor/ModelGate-Hackathon) — winning submission inspiration

---

*Built with ❤️ and 🔥 by Team Dynamic IP for HackArena 2.0 Online Zonals*
