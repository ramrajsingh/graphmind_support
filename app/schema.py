"""Graph schema for the support agent's memory in FalkorDB.

Three kinds of memory live in one graph:

  Semantic (shared, learned across customers)
    (:KnownIssue)-[:AFFECTS]->(:Product)
    (:KnownIssue)-[:RESOLVED_BY]->(:TroubleshootingStep)

  Customer state (one subgraph per customer, reached only through their Customer node)
    (:Customer)-[:OWNS]->(:Product)
    (:Customer)-[:OPENED]->(:Ticket)-[:ABOUT]->(:Product)
    (:Ticket)-[:MATCHES]->(:KnownIssue)
    (:Ticket)-[:TRIED]->(:TroubleshootingStep)          tried, didn't work
    (:Ticket)-[:RESOLVED_WITH]->(:TroubleshootingStep)  fixed it

  Episodic (what happened on each call)
    (:Customer)-[:HAD_CALL]->(:Conversation {summary})-[:HAS_MESSAGE]->(:Message)
    (:Conversation)-[:DISCUSSED]->(:Ticket)

Procedural memory is the agent's playbook in app/agent.py: one step at a time, never repeat a
TRIED step, and escalate when a known issue has no untried fixes.
"""

from app import config
from app.graph import GraphDB, GraphError

INDEXES = [
    "CREATE INDEX FOR (c:Customer) ON (c.customer_id)",
    "CREATE INDEX FOR (t:Ticket) ON (t.id)",
    "CREATE INDEX FOR (v:Conversation) ON (v.id)",
    "CREATE INDEX FOR (p:Product) ON (p.key)",
    "CREATE INDEX FOR (s:TroubleshootingStep) ON (s.key)",
    "CREATE INDEX FOR (k:KnownIssue) ON (k.key)",
    "CALL db.idx.fulltext.createNodeIndex('KnownIssue', 'name', 'symptoms')",
    f"CREATE VECTOR INDEX FOR (k:KnownIssue) ON (k.embedding) "
    f"OPTIONS {{dimension:{config.EMBED_DIM}, similarityFunction:'cosine'}}",
]


def create_indexes(db: GraphDB):
    for stmt in INDEXES:
        try:
            db.query(stmt)
        except GraphError as e:
            if "already" not in str(e).lower():
                raise


def key(name: str) -> str:
    """Case- and whitespace-insensitive identity for shared nodes, so 'Clear App Cache' merges."""
    return " ".join(name.lower().split())
