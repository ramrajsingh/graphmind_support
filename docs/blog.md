# “My account still isn't working”: building a voice agent that remembers, on FalkorDB

*How we gave a multilingual support agent a graph memory, and what we learned about context, latency, and making an AI explain itself.*

**Live demo:** https://graphmind-support.onrender.com · **Code:** https://github.com/ramrajsingh/graphmind_support

---

## The sentence every support bot gets wrong

> “Hi, my account still isn't working.”

Read that sentence the way a human support agent would. The word **still** tells you everything: this person has called before, something was already tried, and they're running out of patience.

Now picture what most AI support bots do with it. They greet you like a stranger, ask you to describe the problem, and suggest clearing your app cache, which you did yesterday. In India it gets worse: the customer may have explained the problem in Hindi yesterday and be calling back in English or Tamil today.

The problem isn't that the model is weak. **The bot has no memory of how things relate**: this customer, this ticket, these steps already tried, and the fix that worked for fifty other people with the same symptom. That's a graph problem, so we built the agent's memory as a graph.

## What we built

**GraphMind Support** is a voice agent called Asha that uses **FalkorDB as its context layer**. You talk to it in any of 22 Indian languages. It remembers every customer across calls and languages, never repeats a fix that already failed, learns from fixes that work, and shows the exact Cypher behind every answer.

Here's the demo that convinced us it works:

**Call 1, in Hindi:**
> 🧑 *“Mera Acme Pay app update ke baad login nahi ho raha hai.”* (My Acme Pay app won't log in since the update.)
>
> 🤖 Asha opens a ticket, matches it to the known issue *Login fails after app update*, and suggests clearing the cache, in Hindi.
>
> 🧑 *“Cache clear kar diya, phir bhi nahi ho raha.”* → recorded as a `TRIED` edge.
>
> 🧑 *“App bhi update kar liya… main baad mein call karta hoon.”* → another `TRIED` edge. He hangs up, and a summary of the call is written to the graph.

**Call 2, the next day, in English, with a fresh session and no chat history:**
> 🧑 *“Hi, my account still isn't working.”*
>
> 🤖 *“Let's try re-linking your 2FA device…”*

No “can you describe the problem?”, no cache-clearing again. Asha went straight to the one fix that hadn't been tried, raised the ticket's priority because this is a repeat contact, and answered in a different language from the first call. On the live deployment that turn takes about five seconds, speech in to speech out.

## Why a graph, not a vector store

Our first instinct, like everyone's, was retrieval: embed past tickets and pull back similar ones. But look at what the callback actually needs:

1. Find **this customer's** open ticket
2. Follow it to the **known issue** it matches
3. Get that issue's **fixes, in the order support tries them**
4. **Remove** every fix this ticket already `TRIED`

That's four hops with a *negation* in the middle. Similarity search can't express “minus what this customer already tried”. A graph does it in one query:

```cypher
MATCH (c:Customer {customer_id: $cid})-[:OPENED]->(t:Ticket)
OPTIONAL MATCH (t)-[:MATCHES]->(k:KnownIssue)
OPTIONAL MATCH (t)-[:TRIED]->(f:TroubleshootingStep)
WITH t, k, collect(f.name) AS tried
OPTIONAL MATCH (k)-[rb:RESOLVED_BY]->(s:TroubleshootingStep)
WHERE NOT (t)-[:TRIED]->(s)
WITH t, k, tried, s ORDER BY rb.rank
RETURN t.id, k.name, tried, collect(s.name) AS untried
```

On FalkorDB Cloud this runs in about **31 ms**.

We still use vectors, just for the part they're good at: matching a new, fuzzily worded problem to a known issue. FalkorDB's vector index lives in the same graph, so “find similar issues, then traverse to fixes” is one query, not two systems glued together.

## Four kinds of memory in one graph

FalkorDB's hackathon asked for agents that *remember what they learn*. We ended up modeling four kinds of memory, all in one graph:

| Memory | What it holds | Shape |
|---|---|---|
| **Semantic** | What the company knows | `(:KnownIssue)-[:RESOLVED_BY {rank}]->(:TroubleshootingStep)`, with embeddings on issues |
| **Customer state** | What happened to *this* customer | `(:Customer)-[:OPENED]->(:Ticket)-[:TRIED\|RESOLVED_WITH]->(:Step)` |
| **Episodic** | What was said on each call | `(:Customer)-[:HAD_CALL]->(:Conversation {summary})-[:HAS_MESSAGE]->(:Message)` |
| **Procedural** | How to behave | One step at a time, never repeat a `TRIED` step, escalate when nothing's left |

The part we're proudest of is the **learning loop**. When a caller says a fix worked, Asha writes it back as shared knowledge. If the ticket matched a known issue, the fix is attached to it. If not, a brand-new embedded `KnownIssue` is created from the ticket.

We tested it across languages. Rahul, speaking Hindi, reported a blinking orange router light that matched nothing, and told Asha a factory reset fixed it. Minutes later Ananya described the same problem **in Bengali, in completely different words**. Vector search found Rahul's learned issue, and Asha suggested the factory reset first. Nobody wrote that knowledge into the system; it came from a phone call in another language.

## Multilingual without multiple graphs

Supporting 22 languages could easily turn into 22 copies of everything. Instead we kept **one language-neutral graph** and handled language at the edges with Sarvam AI:

```
🎙 caller (any language) → Saaras v4 speech-to-text → Sarvam Translate → English
   → LLM + FalkorDB tools →
English reply → Sarvam Translate → caller's language → Bulbul v4 speech 🔊
```

The agent reasons and writes memory in English, while each `Message` node also stores the caller's original words and language. A ticket opened in Hindi is continued in English the next day because the graph never knew the difference.

## What we learned

**1. Preloading context beats calling tools.** Our first version made the agent call a `find_known_fixes` tool on callbacks, which meant an extra LLM round trip of two to three seconds. Then we realized the call-start query already traverses everything needed, so we moved the untried fixes into the agent's context when the call connects. The callback turn's agent time dropped to about 1 second, and accuracy went *up*, because the model stopped improvising before checking.

**2. Enforce rules in the server, not the prompt.** We told the model, in bold, to count a repeat contact only once per call. It still sometimes bumped priority on a ticket it had just opened. Prompting harder didn't help. Moving the rule into code did: the memory layer ignores repeat-contact bumps for tickets opened in the current call, and allows at most one per call. **If a rule matters, put it where the data is.**

**3. Parallel tool calls need ordering.** To cut latency, we ran the model's tool calls in parallel. The model promptly issued `open_ticket` and `record_step` in the same step, and the step was recorded against a ticket that didn't exist yet. The fix was to run ticket creation first and everything else in parallel, plus a fallback that lets an empty ticket reference resolve to the newest open ticket.

**4. Vector search always returns something.** Our first test linked “router LED blinking orange” to “Wi-Fi keeps dropping”, because nearest-neighbor search always returns a nearest neighbor, related or not. Measuring real queries showed a clean gap: true matches scored at most 0.44 (cosine distance), unrelated ones at least 0.57. A **0.5 cutoff** made “no match” a real answer, and “no match” is exactly what triggers learning a new issue. We also stopped embedding the product name into the query, since the product is already matched through the `AFFECTS` edge, and including it made unrelated router problems look similar.

**5. Graph results need deterministic order.** Fixes came back in arbitrary order, so Asha sometimes suggested “update the app” before “clear the cache”, which made scripted demos break randomly. Storing a `rank` on each `RESOLVED_BY` edge and sorting in Cypher fixed it.

**6. Latency is a budget, and the database was the cheapest line.** A voice turn includes speech-to-text, translation, the LLM, translation back, and speech synthesis. FalkorDB queries took 28–160 ms, the smallest cost in the pipeline. The big wins came from elsewhere: running independent calls concurrently, logging messages in the background, and synthesizing speech **sentence by sentence in parallel**, which cut text-to-speech from about 3 s to about 1.6 s. Overall a callback turn went from about 14 s in our first prototype to about 5 s.

**7. Keep tech terms out of translation.** Translating *“Clear app cache”* into pure Tamil gives you a sentence no Tamil speaker would actually say. Sarvam's colloquial translation mode keeps the words people really use (“app cache-அ clear பண்ணுங்க”), and STT keyterms keep “Acme Pay” from turning into “एक्मे पे”, which would also break product matching in the graph.

**8. A smaller model wasn't cheaper here.** We tried a mini model for speed. It made an unnecessary tool call, bumped priority wrongly, and forgot to escalate, so it wasn't consistently faster *and* it was less reliable. For an agent acting on memory, a wrong action costs more than a slow one.

## Making the agent explain itself

Every turn returns the exact Cypher it ran, with timings, and the UI shows it beside a live view of the graph, where the nodes and edges used in the answer are highlighted. When Asha says “let's try re-linking your 2FA device”, you can see *why*: which known issue matched, which steps were excluded as `TRIED`, and the rank order of what was left.

Building this changed how we debugged, too. Every bug above showed up first in the Cypher panel, not in the chat transcript.

## Try it

- **Live:** https://graphmind-support.onrender.com. Pick *Rahul Verma*, press **Start call**, and type the lines from the demo above (or hold to talk). The free instance may take a minute to wake up.
- **Code, data model and every Cypher query:** https://github.com/ramrajsingh/graphmind_support (see `docs/cypher.md`)

**Stack:** FalkorDB (graph, vector, full-text) · Sarvam AI (Saaras v4, Translate, Bulbul v4) · OpenAI GPT-4.1 with tool use · FastAPI · vis-network

*Built for Graph Hacks: Context for AI Agents by WeMakeDevs × FalkorDB, Track 02 (Agent Memory) and Track 01 (Connected Data).*
