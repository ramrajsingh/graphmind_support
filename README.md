# Asha: a multilingual voice support agent with FalkorDB memory

A voice support agent whose memory is a FalkorDB graph. A customer reports a problem in Hindi, calls back the next day in Tamil or English, and Asha already knows the open ticket, which fixes failed, and which fix worked for other customers. Every answer can be traced to the Cypher that produced it.

Built for **Graph Hacks: Context for AI Agents** (WeMakeDevs × FalkorDB):
- **Track 02, Agent Memory:** episodic, semantic and procedural memory, isolated per customer.
- **Track 01, Agents That Act on Connected Data:** multi-hop graph queries as tools, with explainable results.

```
browser mic ─► Sarvam Saaras STT ─► Sarvam Translate → English ─► LLM + memory tools ─► FalkorDB (Cypher)
     ▲                                                                   │
     └──── Bulbul TTS ◄── Sarvam Translate → caller's language ◄─────────┘
```

## Memory in FalkorDB

| Memory | Graph | Used for |
|---|---|---|
| **Semantic** (shared, learned) | `(:KnownIssue)-[:RESOLVED_BY {rank}]->(:TroubleshootingStep)`, `(:KnownIssue)-[:AFFECTS]->(:Product)`, vector index on `KnownIssue.embedding` | Matching a new problem to a known issue (vector search), proven fixes |
| **Customer state** (per customer) | `(:Customer)-[:OPENED]->(:Ticket)-[:TRIED\|RESOLVED_WITH]->(:TroubleshootingStep)`, `(:Ticket)-[:MATCHES]->(:KnownIssue)` | What was tried, what's left, ticket priority |
| **Episodic** | `(:Customer)-[:HAD_CALL]->(:Conversation {summary})-[:HAS_MESSAGE]->(:Message)`, `(:Conversation)-[:DISCUSSED]->(:Ticket)` | Call summaries read back at the start of the next call |
| **Procedural** | The agent playbook in `app/agent.py` | One step at a time, never repeat a `TRIED` step, escalate when nothing is left |

**Multi-hop example:** the next fix for a callback is a single traversal, *known issue → ranked fixes, minus what this ticket already tried*:

```cypher
MATCH (c:Customer {customer_id: $cid})-[:OPENED]->(t:Ticket)-[:MATCHES]->(k:KnownIssue)
OPTIONAL MATCH (k)-[rb:RESOLVED_BY]->(s:TroubleshootingStep)
WHERE NOT (t)-[:TRIED]->(s)
RETURN t.id, k.name, collect(s.name)   -- ordered by rb.rank
```

**Learning loop:** when a step fixes a ticket, it's added to the known issue as a `RESOLVED_BY` link, or a new embedded `KnownIssue` is created. The next customer with a similar problem, in any language, gets that fix first.

**Explainable:** every turn returns the Cypher it ran and how long each query took, shown under *Agent memory calls & Cypher* in the UI.

**Per-customer isolation:** the server binds each call to one `customer_id`, and every query starts from that `Customer` node. The agent can't reach another customer's tickets.

## Multilingual (Sarvam AI)

- **Saaras v4** transcribes 22 Indian languages plus English (including mixed speech like Hinglish) and detects the language.
- **Sarvam Translate** puts everything into English, so the memory graph is language-neutral. Replies go back in colloquial style, keeping tech terms in English.
- **Bulbul v4** speaks with a persona voice per language. Typed caller lines are also read aloud in a male or female caller voice, so a typed demo still sounds like a two-person call.

## Setup

```bash
uv venv && uv pip install -r requirements.txt
cp .env.example .env              # FalkorDB, Sarvam and OpenAI (or Anthropic) keys
.venv/bin/python -m scripts.setup_graph     # indexes + seed data (safe to re-run)
.venv/bin/uvicorn app.server:app --port 8000
```

FalkorDB can run locally (`docker run -p 6379:6379 falkordb/falkordb`) or on FalkorDB Cloud. Open http://localhost:8000, pick a caller, press **Start call**, then hold **Hold to talk** or type in any language.

Reset between demo runs. This clears tickets, calls and anything learned, and keeps the seed data:
```bash
.venv/bin/python -m scripts.reset_demo
```

## Demo script (Rahul Verma, Hindi then English)

1. **Call 1, Hindi:** "Mera Acme Pay app update ke baad login nahi ho raha hai." Asha opens a ticket linked to *Login fails after app update* and suggests clearing the cache.
2. "Cache clear kar diya, phir bhi login nahi ho raha." This is recorded as a `TRIED` link.
3. "App bhi update kar liya, abhi bhi nahi chal raha. Main baad mein call karta hoon." Hang up. The call summary is saved to the `Conversation` node.
4. **Call 2, English:** "Hi, my account still isn't working." Asha recalls the ticket and both failed steps, priority goes to high, and she suggests *Re-link 2FA device*.
5. "Yes, I re-linked the 2FA device and now I can log in. Thank you!" The ticket resolves and the fix is confirmed as shared knowledge.

## Files

| Path | What it does |
|---|---|
| `app/graph.py` | FalkorDB connection, query logging, embeddings |
| `app/schema.py` | Graph schema and indexes (vector, full-text, range) |
| `app/memory.py` | Customer-scoped memory as Cypher: snapshot, known fixes, tickets, learning, calls |
| `app/agent.py` | System prompt, tool definitions, Claude tool loop, call summaries |
| `app/agent_openai.py` | OpenAI tool loop (default) using the same prompt and tools |
| `app/voice.py` | Sarvam STT, translation, TTS, voices per language |
| `app/server.py` | FastAPI endpoints for calls and turns |
| `static/index.html` | Push-to-talk UI with the live memory graph and Cypher trace |
| `scripts/setup_graph.py`, `scripts/reset_demo.py` | Seed and reset |
| `data/seed.json` | Products, steps, known issues, demo customers |
