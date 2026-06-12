# ModelRouter — 3-Minute Demo Video Script

## Overview
**Team:** Dynamic IP (Vardhan, Rajesh)
**Product:** ModelRouter — Intelligent LLM Query Routing System
**Runtime:** ~3 minutes

---

### [0:00–0:30] THE PROBLEM (Visual: Slide 1)

**Narrator:**
"Every month, a typical startup burns $50,000 on GPT-5 API calls. The problem? 80% of those queries are simple FAQs — 'What is your return policy?' — that a tiny model could answer in 300 milliseconds for a fraction of a cent.

That's $40,000 a month — an engineer's salary — going up in smoke. For nothing.

This is the AI industry's dirty secret: most teams pick one premium model and never revisit. The result? 50 to 90% of inference spend is completely wasted."

---

### [0:30–0:55] THE SOLUTION (Visual: Slide 3 + code)

**Narrator:**
"Meet ModelRouter. One line of code changes everything.

You change your base URL from `api.openai.com` to `router.acme/v1`. That's it. Same code. Same SDK. One change.

Now every incoming prompt is automatically classified by complexity — Simple, Medium, or Complex — and routed to the cheapest model that can handle it. A FAQ goes to GPT-5.4 nano at $0.00015 per thousand tokens. A legal analysis goes to Claude Opus 4.6. Same endpoint. Different model. Optimal choice every time."

---

### [0:55–1:25] HOW IT WORKS (Visual: Dashboard screen recording)

**Narrator:**
"Let me show you the live dashboard.

Here's our playground. I'll type a simple query: 'What is your return policy?' — hit Route — and watch ModelRouter classify it as 'Simple' with high confidence, route it to GPT-5.4 nano, saving 99% compared to premium.

Now watch this — I'll type a complex query: 'Analyze liability exposure in Section 4.2 of this contract...' — same endpoint, but ModelRouter classifies it as 'Complex' and routes it to Opus 4.6. The developer didn't change a single line of code. The system adapts automatically.

Here in the route log, you see every decision — classification, model selected, cost savings, and latency."

---

### [1:25–1:50] CONTRACT PARSING (Visual: Contract parser in dashboard)

**Narrator:**
"Here's what makes ModelRouter truly unique. Watch this.

I paste in a data processing agreement — GDPR compliance, EU-only processing, DeepSeek blocked, 1000ms latency target. I click 'Parse Contract' and ModelRouter extracts every constraint automatically:

- Region locked to EU
- DeepSeek blocked from routing
- Latency target set to 1000ms
- Cost sensitivity: high

Now every query that comes through ACME's endpoint is filtered against these rules. Compliance isn't a checkbox — it's baked into every routing decision, automatically, per request."

---

### [1:50–2:20] THE INTELLIGENCE (Visual: Slide 5)

**Narrator:**
"The brains behind ModelRouter is a fine-tuned 1.5 billion parameter model. We started with Qwen2.5, fine-tuned it with GRPO on just 172 labeled prompts — taking only 2 and a half minutes on an RTX 3080 laptop.

The result? A router that achieves 85.7% accuracy on medium-complexity queries — compared to just 14.3% from the stock router. That's a 71 percentage point improvement.

Overall accuracy: 83.3%. Cost savings versus always-premium routing: 59%. And each classification takes just 62 milliseconds."

---

### [2:20–2:45] THE IMPACT (Visual: Slide 6)

**Narrator:**
"The impact is measured at every layer:

- **Business**: 60 to 98% reduction in AI inference spend
- **Users**: Simple queries get fast answers, complex queries get powerful models
- **Data Centers**: Less unnecessary GPU load  
- **Environment**: 180 times less energy per routed-away premium call
- **Compliance**: Contract constraints enforced automatically, per request

One change. Benefits at every layer of the stack."

---

### [2:45–3:00] CLOSE (Visual: Slide 8)

**Narrator:**
"To see ModelRouter in action, visit our GitHub repo — github.com/dynamicip/modelrouter — where you can pull the Docker image, run it in two commands, and start routing smarter today.

We're Team Dynamic IP — Vardhan and Rajesh. This is ModelRouter: one URL change, every model, optimal choice, zero code changes.

Thank you."

---

## Production Notes

### Recording Setup
- **Screen**: 1920×1080, capture full browser window
- **Audio**: Clear narration, no background music
- **Split**: Have the presentation slides open in one window and the live dashboard in another
- **Switching**: Cut between slides and dashboard cleanly

### Dashboard Demo Sequence (timed)
1. Open http://localhost:8000/dashboard
2. Click "Ask Simple FAQ" demo button → show classification, model, 99% savings
3. Click "Analyze Liabilities" → show Complex classification → Opus 4.6
4. Point to Route Log showing both entries
5. Scroll to Contract Parser → paste GDPR agreement → click "Parse"
6. Show extracted profile

### Files to Include
- `ModelRouter-Presentation.pdf` — 8-slide deck
- `demo-script.md` — this script
- Link to GitHub repo
