"""Support-agent memory as Cypher on FalkorDB (schema in app/schema.py).

Every method is scoped to one customer_id, which the server binds per call, so the agent can
only reach other data through that customer's node or the shared knowledge graph. Each query is
recorded in `queries` so the UI can show the Cypher behind every answer.
"""

import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from app.graph import GraphDB, GraphError, embed
from app.schema import key

POOL = ThreadPoolExecutor(max_workers=12)
# Cosine distance cutoff for "same problem": real matches score <= ~0.45, unrelated ones >= ~0.57.
MAX_ISSUE_DISTANCE = 0.5


def pmap(fn, items) -> list:
    return list(POOL.map(fn, items))


@dataclass
class Graph:
    """Nodes and edges touched while answering, sent to the UI's memory panel."""
    nodes: dict[str, dict] = field(default_factory=dict)
    edges: list[dict] = field(default_factory=list)

    def add_node(self, type_: str, name: str, ident: str | None = None, highlight: bool = False) -> str:
        nid = f"{type_}:{ident or key(name)}"
        cur = self.nodes.setdefault(nid, {"id": nid, "name": name, "type": type_})
        cur["highlight"] = cur.get("highlight", False) or highlight
        return nid

    def add_edge(self, src: str, rel: str, tgt: str, highlight: bool = False):
        for e in self.edges:
            if (e["source"], e["type"], e["target"]) == (src, rel, tgt):
                e["highlight"] = e["highlight"] or highlight
                return
        self.edges.append({"source": src, "type": rel, "target": tgt, "highlight": highlight})

    def to_json(self) -> dict:
        return {"nodes": list(self.nodes.values()), "edges": self.edges}


STEP_RELS = {"failed": "TRIED", "worked": "RESOLVED_WITH"}
PRIORITIES = ["normal", "high", "urgent"]


class SupportMemory:
    def __init__(self, db: GraphDB, customer_id: str):
        self.db = db
        self.cid = customer_id
        self.conversation_id: str | None = None
        self.graph = Graph()
        self.queries: list[dict] = []
        self._customer: dict | None = None
        self._tickets: list[dict] | None = None
        # Tickets opened, or already counted as a repeat contact, during this call.
        self._no_repeat_bump: set[str] = set()

    def q(self, label: str, cypher: str, **params) -> list[dict]:
        return self.db.query(cypher, {"cid": self.cid, **params}, log=self.queries, label=label)

    def _cust_node(self, highlight: bool = True) -> str:
        c = self.customer() or {}
        return self.graph.add_node("Customer", c.get("name") or self.cid, self.cid, highlight)

    # --- Reads ---------------------------------------------------------------
    def customer(self) -> dict | None:
        if self._customer is None:
            rows = self.q("customer", """
                MATCH (c:Customer {customer_id: $cid})
                OPTIONAL MATCH (c)-[:OWNS]->(p:Product)
                RETURN c.name AS name, c.preferred_language AS preferred_language, collect(p.name) AS products""")
            self._customer = rows[0] if rows and rows[0]["name"] else None
        return self._customer

    def snapshot(self) -> dict:
        cust = self.customer()
        if not cust:
            return {"customer": None, "products": [], "tickets": []}
        cnode = self._cust_node()
        for p in cust["products"]:
            self.graph.add_edge(cnode, "OWNS", self.graph.add_node("Product", p))
        # One traversal: each ticket with its product, known issue, what was tried, and the known
        # fixes not yet tried, so a callback can go straight to the next step.
        rows = self.q("customer tickets + untried fixes", """
            MATCH (c:Customer {customer_id: $cid})-[:OPENED]->(t:Ticket)
            OPTIONAL MATCH (t)-[:ABOUT]->(p:Product)
            OPTIONAL MATCH (t)-[:MATCHES]->(k:KnownIssue)
            OPTIONAL MATCH (t)-[:TRIED]->(f:TroubleshootingStep)
            OPTIONAL MATCH (t)-[:RESOLVED_WITH]->(w:TroubleshootingStep)
            WITH t, p, k, collect(DISTINCT f.name) AS tried, collect(DISTINCT w.name) AS worked
            OPTIONAL MATCH (k)-[rb:RESOLVED_BY]->(s:TroubleshootingStep)
            WHERE NOT (t)-[:TRIED]->(s)
            WITH t, p, k, tried, worked, s, coalesce(rb.rank, 99) AS rank ORDER BY rank
            RETURN t.id AS ticket, t.status AS status, t.priority AS priority, t.contacts AS contacts,
                   t.summary AS summary, p.name AS product, k.name AS known_issue, tried, worked,
                   collect(DISTINCT {step: s.name, instructions: s.instructions}) AS untried,
                   t.created_at AS created
            ORDER BY created""")
        tickets = []
        for r in rows:
            open_ = r["status"] != "resolved"
            tnode = self.graph.add_node("Ticket", r["ticket"], r["ticket"], open_)
            self.graph.add_edge(cnode, "OPENED", tnode)
            if r["product"]:
                self.graph.add_edge(tnode, "ABOUT", self.graph.add_node("Product", r["product"]), open_)
            if r["known_issue"]:
                self.graph.add_edge(tnode, "MATCHES",
                                    self.graph.add_node("KnownIssue", r["known_issue"], highlight=open_), open_)
            for s in r["tried"]:
                self.graph.add_edge(tnode, "TRIED", self.graph.add_node("TroubleshootingStep", s, highlight=open_), open_)
            for s in r["worked"]:
                self.graph.add_edge(tnode, "RESOLVED_WITH", self.graph.add_node("TroubleshootingStep", s), open_)
            t = {k: r[k] for k in ("ticket", "status", "priority", "summary", "product", "known_issue")}
            t.update(contacts=r["contacts"], tried_and_failed=r["tried"], resolved_with=r["worked"])
            if open_ and r["known_issue"]:
                t["untried_known_fixes"] = [u for u in r["untried"] if u.get("step")]
            tickets.append(t)
        self._tickets = [{"id": t["ticket"], "status": t["status"]} for t in tickets]
        tickets.sort(key=lambda t: t["status"] == "resolved")
        return {"customer": {"name": cust["name"], "customer_id": self.cid,
                             "preferred_language": cust["preferred_language"]},
                "products": cust["products"], "tickets": tickets}

    def past_call_summaries(self, limit: int = 3) -> list[str]:
        """Episodic memory: summaries written when this customer's earlier calls ended."""
        rows = self.q("earlier call summaries", """
            MATCH (c:Customer {customer_id: $cid})-[:HAD_CALL]->(v:Conversation)
            WHERE v.summary IS NOT NULL
            RETURN v.summary AS summary, v.started_at AS started ORDER BY started DESC LIMIT $limit""",
            limit=limit)
        return [r["summary"] for r in rows]

    def find_known_fixes(self, problem: str, product: str | None = None, ticket: str | None = None) -> dict:
        """Semantic memory: known issues similar to the problem, with fixes split into tried/untried
        for this ticket. One query: vector search, then a multi-hop traversal."""
        tid = self._ticket(ticket)["id"] if ticket else None
        vec = embed(problem)  # product is matched through AFFECTS, not the embedding
        if vec is not None:
            search = ("CALL db.idx.vector.queryNodes('KnownIssue', 'embedding', 6, vecf32($vec)) YIELD node AS k, score "
                      "WITH k, score WHERE score < $maxdist")
            order, params = "score ASC", {"vec": vec, "maxdist": MAX_ISSUE_DISTANCE}
        else:
            terms = "|".join(w for w in re.findall(r"[a-z0-9]+", problem.lower()) if len(w) > 2) or "issue"
            search = "CALL db.idx.fulltext.queryNodes('KnownIssue', $terms) YIELD node AS k, score"
            order, params = "score DESC", {"terms": terms}
        rows = self.q("known fixes minus tried", search + """
            OPTIONAL MATCH (k)-[:AFFECTS]->(p:Product)
            WITH k, score, p WHERE $pkey IS NULL OR p IS NULL OR p.key = $pkey
            MATCH (k)-[rb:RESOLVED_BY]->(s:TroubleshootingStep)
            OPTIONAL MATCH (t:Ticket {id: $tid})-[tr:TRIED]->(s)
            WITH k, score, p, s, count(tr) > 0 AS tried, coalesce(rb.rank, 99) AS rank ORDER BY rank
            RETURN k.name AS known_issue, k.symptoms AS symptoms, p.name AS product, score,
                   collect({step: s.name, instructions: s.instructions, tried: tried}) AS fixes
            ORDER BY """ + order + " LIMIT 3",
            pkey=key(product) if product else None, tid=tid, **params)
        matches = []
        for r in rows:
            knode = self.graph.add_node("KnownIssue", r["known_issue"], highlight=True)
            if r["product"]:
                self.graph.add_edge(knode, "AFFECTS", self.graph.add_node("Product", r["product"]))
            for f in r["fixes"]:
                fresh = not f["tried"]
                self.graph.add_edge(knode, "RESOLVED_BY",
                                    self.graph.add_node("TroubleshootingStep", f["step"], highlight=fresh), fresh)
            matches.append({
                "known_issue": r["known_issue"], "symptoms": r["symptoms"], "product": r["product"],
                "untried_fixes": [{"step": f["step"], "instructions": f["instructions"]}
                                  for f in r["fixes"] if not f["tried"]],
                "already_tried": [f["step"] for f in r["fixes"] if f["tried"]],
            })
        return {"matches": matches,
                "note": "" if matches else "No known issue matched; troubleshoot from first principles."}

    def _ticket(self, ref: str | None) -> dict:
        if self._tickets is None:
            self._tickets = self.q("customer tickets", """
                MATCH (c:Customer {customer_id: $cid})-[:OPENED]->(t:Ticket)
                RETURN t.id AS id, t.status AS status, t.created_at AS created ORDER BY created""")
        ref = (ref or "").strip().lower()
        if not ref:
            # The model sometimes records a step alongside opening the ticket, before it has the ID.
            open_ = [t for t in self._tickets if t["status"] != "resolved"]
            if open_:
                return open_[-1]
        for t in self._tickets:
            if ref and (t["id"].lower() == ref or ref in t["id"].lower()):
                return t
        raise GraphError(f"No ticket {ref!r} for this customer")

    # --- Writes --------------------------------------------------------------
    def start_new_issue(self, problem: str, product: str) -> dict:
        """Find known fixes and open a ticket linked to the best match, in one tool call."""
        found = self.find_known_fixes(problem, product)
        best = found["matches"][0]["known_issue"] if found["matches"] else None
        return {**self.open_ticket(product, problem, best), **found}

    def open_ticket(self, product: str, summary: str, known_issue: str | None = None) -> dict:
        if not self.customer():
            raise GraphError("Customer not found; run python -m scripts.setup_graph")
        tid = f"TKT-{uuid.uuid4().hex[:8].upper()}"
        self.q("open ticket", """
            MATCH (c:Customer {customer_id: $cid})
            CREATE (c)-[:OPENED]->(t:Ticket {id: $tid, status: 'open', priority: 'normal', contacts: 1,
                    summary: $summary, created_at: timestamp(), updated_at: timestamp()})
            WITH t OPTIONAL MATCH (p:Product {key: $pkey})
            WITH t, p OPTIONAL MATCH (k:KnownIssue {key: $kkey})
            FOREACH (_ IN CASE WHEN p IS NULL THEN [] ELSE [1] END | CREATE (t)-[:ABOUT]->(p))
            FOREACH (_ IN CASE WHEN k IS NULL THEN [] ELSE [1] END | CREATE (t)-[:MATCHES]->(k))""",
            tid=tid, summary=summary, pkey=key(product), kkey=key(known_issue) if known_issue else "")
        self._no_repeat_bump.add(tid)
        if self._tickets is not None:
            self._tickets.append({"id": tid, "status": "open"})
        self._discussed(tid)
        tnode = self.graph.add_node("Ticket", tid, tid, True)
        self.graph.add_edge(self._cust_node(), "OPENED", tnode, True)
        self.graph.add_edge(tnode, "ABOUT", self.graph.add_node("Product", product), True)
        if known_issue:
            self.graph.add_edge(tnode, "MATCHES", self.graph.add_node("KnownIssue", known_issue, highlight=True), True)
        return {"ticket": tid, "status": "open", "priority": "normal", "contacts": 1, "summary": summary}

    def record_step(self, ticket: str, step: str, outcome: str) -> dict:
        t = self._ticket(ticket)
        rel = STEP_RELS[outcome]
        self.q(f"record step ({rel})", f"""
            MATCH (c:Customer {{customer_id: $cid}})-[:OPENED]->(t:Ticket {{id: $tid}})
            MERGE (s:TroubleshootingStep {{key: $skey}}) ON CREATE SET s.name = $step, s.instructions = ''
            CREATE (t)-[:{rel} {{at: timestamp()}}]->(s)""", tid=t["id"], skey=key(step), step=step)
        self._discussed(t["id"])
        tnode = self.graph.add_node("Ticket", t["id"], t["id"], True)
        self.graph.add_edge(tnode, rel, self.graph.add_node("TroubleshootingStep", step, highlight=True), True)
        result = {"ticket": t["id"], "step": step, "outcome": outcome}
        if outcome == "worked":
            self.update_ticket(t["id"], status="resolved")
            result["learned"] = self._learn_fix(t["id"], step)
        return result

    def _learn_fix(self, tid: str, step: str) -> str:
        """Promote a confirmed fix to shared knowledge so the next customer benefits."""
        rows = self.q("learn fix: existing known issue", """
            MATCH (t:Ticket {id: $tid})-[:MATCHES]->(k:KnownIssue), (s:TroubleshootingStep {key: $skey})
            OPTIONAL MATCH (k)-[existing:RESOLVED_BY]->(s)
            WITH k, s, existing
            FOREACH (_ IN CASE WHEN existing IS NULL THEN [1] ELSE [] END |
                CREATE (k)-[:RESOLVED_BY {learned: true, at: timestamp()}]->(s))
            RETURN k.name AS known_issue, existing IS NULL AS added""", tid=tid, skey=key(step))
        if rows:
            r = rows[0]
            knode = self.graph.add_node("KnownIssue", r["known_issue"], highlight=True)
            self.graph.add_edge(knode, "RESOLVED_BY", self.graph.add_node("TroubleshootingStep", step), True)
            return f"Added this fix to known issue {r['known_issue']!r}" if r["added"] else "Fix was already known"
        # No known issue yet: create one from the ticket, so similar problems find this fix.
        info = self.q("ticket summary", """
            MATCH (t:Ticket {id: $tid}) OPTIONAL MATCH (t)-[:ABOUT]->(p:Product)
            RETURN t.summary AS summary, p.key AS pkey""", tid=tid)[0]
        summary = info["summary"]
        self.q("learn fix: new known issue", """
            MATCH (t:Ticket {id: $tid}), (s:TroubleshootingStep {key: $skey})
            MERGE (k:KnownIssue {key: $kkey})
              ON CREATE SET k.name = $name, k.symptoms = $name, k.learned = true
            CREATE (t)-[:MATCHES]->(k)
            MERGE (k)-[r:RESOLVED_BY]->(s) ON CREATE SET r.learned = true, r.at = timestamp()
            WITH k OPTIONAL MATCH (p:Product {key: $pkey})
            FOREACH (_ IN CASE WHEN p IS NULL THEN [] ELSE [1] END | MERGE (k)-[:AFFECTS]->(p))""",
            tid=tid, skey=key(step), kkey=key(summary), name=summary, pkey=info["pkey"] or "")
        vec = embed(summary)
        if vec is not None:
            self.q("embed new known issue", "MATCH (k:KnownIssue {key: $kkey}) SET k.embedding = vecf32($vec)",
                   kkey=key(summary), vec=vec)
        knode = self.graph.add_node("KnownIssue", summary, highlight=True)
        self.graph.add_edge(knode, "RESOLVED_BY", self.graph.add_node("TroubleshootingStep", step), True)
        return "Created a new known issue from this ticket"

    def update_ticket(self, ticket: str, status: str | None = None, priority: str | None = None,
                      repeat_contact: bool = False, note: str | None = None) -> dict:
        t = self._ticket(ticket)
        bump = repeat_contact and t["id"] not in self._no_repeat_bump
        if bump:  # a repeat contact counts once per call, never for a ticket opened in this call
            self._no_repeat_bump.add(t["id"])
        # Repeat contacts raise priority automatically (2nd contact: high, 3rd: urgent), never lower it.
        row = self.q("update ticket", """
            MATCH (c:Customer {customer_id: $cid})-[:OPENED]->(t:Ticket {id: $tid})
            WITH t, t.contacts + $bump AS contacts
            WITH t, contacts, coalesce($priority, t.priority) AS wanted,
                 CASE WHEN contacts >= 3 THEN 'urgent' WHEN contacts >= 2 THEN 'high' ELSE 'normal' END AS floor
            SET t.contacts = contacts,
                t.status = coalesce($status, t.status),
                t.priority = CASE WHEN $rank[floor] > $rank[wanted] THEN floor ELSE wanted END,
                t.summary = CASE WHEN $note IS NULL THEN t.summary ELSE t.summary + ' [' + $note + ']' END,
                t.updated_at = timestamp()
            RETURN t.id AS ticket, t.status AS status, t.priority AS priority, t.contacts AS contacts,
                   t.summary AS summary""",
            tid=t["id"], bump=1 if bump else 0, status=status, priority=priority, note=note,
            rank={p: i for i, p in enumerate(PRIORITIES)})[0]
        t["status"] = row["status"]
        self._discussed(t["id"])
        self.graph.add_node("Ticket", t["id"], t["id"], True)
        return row

    # --- Episodic memory: calls and messages --------------------------------
    def start_conversation(self, language: str) -> str:
        self.conversation_id = uuid.uuid4().hex
        self.q("start call", """
            MATCH (c:Customer {customer_id: $cid})
            CREATE (c)-[:HAD_CALL]->(:Conversation {id: $vid, started_at: timestamp(), language: $lang})""",
            vid=self.conversation_id, lang=language)
        return self.conversation_id

    def log_message(self, role: str, content: str, original: str = "", language: str = ""):
        self.db.query("""
            MATCH (v:Conversation {id: $vid})
            CREATE (v)-[:HAS_MESSAGE]->(:Message {role: $role, content: $content, original: $original,
                                                  language: $lang, at: timestamp()})""",
                      {"vid": self.conversation_id, "role": role, "content": content, "original": original,
                       "lang": language})

    def end_conversation(self, summary: str):
        self.db.query("MATCH (v:Conversation {id: $vid}) SET v.summary = $summary, v.ended_at = timestamp()",
                      {"vid": self.conversation_id, "summary": summary})

    def _discussed(self, tid: str):
        if self.conversation_id:
            self.db.query("MATCH (v:Conversation {id: $vid}), (t:Ticket {id: $tid}) MERGE (v)-[:DISCUSSED]->(t)",
                          {"vid": self.conversation_id, "tid": tid})
