# AGENTS.md — ModelRouter Build Log

## Team Dynamic IP
- **Vardhan** — Backend & AI (FastAPI, classifier, routing engine, contract parser)
- **Rajesh** — Frontend & Platform (dashboard, deployment, presentation)

---

## Build Timeline

### Session 1: Foundation
- Reviewed HackArena 2.0 requirements (Round 1: PPT/PDF by June 9, Round 2: 10-min demo June 13-14)
- Analyzed winning hackathon projects (ModelGate — 3rd place KSU Social Good Hackathon)
- Selected ModelRouter concept: intelligent LLM query routing with GRPO fine-tuned classifier
- Built presentation: dark theme, gradient animations, 12 slides (keyboard/touch navigation)

### Session 2: Presentation Overhaul
- User wanted emotional storytelling hook ("Money on Fire" — startup burns $50K/mo on GPT-5)
- Redesigned: pure black + bright red (#FF1744), bold typography, minimal text
- Restructured to align with hackathon requirements:
  1. THE WASTE → Problem Statement (story hook)
  2. THE PROBLEM → Problem Statement (hard stats)
  3. THE FIX → Proposed Solution (one line of code)
  4. HOW IT WORKS → Implementation Approach (architecture)
  5. THE BRAIN + STACK → Tech Stack + Implementation
  6. THE IMPACT → Expected Impact
  7. LIVE DEMO → Implementation
  8. THANK YOU → Future Scope + Team
- Generated Gamma AI prompt, screenshots (Puppeteer), PDF, and PPTX

### Session 3: Prototype Build
- Created GitHub repo: `github.com/rajesh-puripanda/dynamicip-modelrouter`
- Built backend (FastAPI):
  - `classifier.py` — Heuristic + keyword prompt classification (3 tiers)
  - `contract_parser.py` — Contract text → CustomerProfile extraction
  - `models.py` — Pydantic schemas (ChatRequest, RouteDecision, etc.)
  - `main.py` — API server with /v1/chat/completions, /v1/contract/parse, /v1/stats
  - `dashboard.py` — Static dashboard route
- Built frontend (`frontend/index.html`):
  - Dark black/red theme matching presentation
  - Live stats: routes logged, cost savings, classification speed
  - Playground: test prompts with classification & routing preview
  - Route log: real-time routing decisions
  - Contract parser: paste agreement → see extracted constraints
- Docker setup: `docker-compose.yml`, `Dockerfile`
- Documentation: `README.md` (full project docs), `AGENTS.md` (this file)

---

## Key Technical Decisions

1. **Classifier**: Heuristic-based (keyword + regex + length) as MVP. Real fine-tuned Arch-Router-1.5B GGUF model replaces it in production.
2. **No External LLM Dependency**: The prototype works standalone. For real LLM responses, connect OpenRouter (`OPENROUTER_API_KEY`).
3. **Single-File Frontend**: All CSS/JS inline in `index.html` — zero build tools, instant rendering.
4. **Cost Model**: 3 pricing tiers — Simple ($0.00015/1K), Medium ($0.002/1K), Complex ($0.015/1K). Baseline premium at $0.015/1K (GPT-5.4).

---

## Running the Prototype

```bash
cd dynamicip-modelrouter
pip install -r backend/requirements.txt
python backend/main.py
# → http://localhost:8000 (API) + http://localhost:8000/dashboard (Dashboard)
```

Or with Docker:
```bash
docker compose up --build
```

---

## File Map

```
dynamicip-modelrouter/
├── backend/
│   ├── main.py              # FastAPI server — endpoints + middleware
│   ├── classifier.py        # Prompt classification engine (3 tiers)
│   ├── contract_parser.py   # Contract → CustomerProfile extraction
│   ├── models.py            # Data models (pydantic)
│   ├── dashboard.py         # Static dashboard route
│   ├── requirements.txt     # Dependencies
│   └── Dockerfile           # Container
├── frontend/
│   └── index.html           # Dashboard UI (inline, no build)
├── data/                    # Runtime data (gitignored)
│   └── routing_log.json     # Route log
├── docker-compose.yml       # Service orchestration
├── README.md                # Project documentation
└── AGENTS.md                # This file — build log
```

---

## Next Steps for the Team

1. **Connect OpenRouter**: Add `OPENROUTER_API_KEY` env var → real LLM responses instead of simulated ones
2. **Add GGUF Classifier**: Replace heuristic classifier with actual `ModelRouter.Q8_0.gguf` inference using `llama-cpp-python`
3. **Build Contract Library**: Support multiple contract uploads with vector search
4. **Live Dashboard**: Upgrade to Next.js with Recharts for real-time visualizations
5. **CI/CD**: GitHub Actions for automated testing and Docker image build

---

## HackArena 2.0 Submission Checklist

- [x] Team registered on Unstop (Dynamic IP)
- [x] PPT/PDF ready (presentation + exports)
- [x] Prototype running (FastAPI + dashboard)
- [x] GitHub repo with README
- [x] Pitch deck covers: Problem, Solution, Tech Stack, Implementation, Impact, Future Scope
- [ ] Rehearse 10-min demo for Round 2
- [ ] Record demo video (backup for judging)
- [ ] Prepare answers for: "What makes this different from RouteLLM?" and "How do you handle cold start?"
