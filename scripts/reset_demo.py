"""Reset the demo so the demo script plays out the same way every time.

Deletes tickets, calls and messages, plus anything the agent learned during demos (new known
issues and new RESOLVED_BY fixes). Seeded products, known issues, steps and customers stay.

    python -m scripts.reset_demo
"""

from app.graph import GraphDB

if __name__ == "__main__":
    db = GraphDB()
    counts = {
        "tickets": "MATCH (t:Ticket) DETACH DELETE t RETURN count(t) AS n",
        "messages": "MATCH (m:Message) DETACH DELETE m RETURN count(m) AS n",
        "calls": "MATCH (v:Conversation) DETACH DELETE v RETURN count(v) AS n",
        "learned fixes": "MATCH ()-[r:RESOLVED_BY {learned: true}]->() DELETE r RETURN count(r) AS n",
        "learned known issues": "MATCH (k:KnownIssue {learned: true}) DETACH DELETE k RETURN count(k) AS n",
        "agent-created steps": "MATCH (s:TroubleshootingStep {instructions: ''}) "
                               "WHERE NOT ()-[:RESOLVED_BY]->(s) DETACH DELETE s RETURN count(s) AS n",
    }
    print("Removed " + ", ".join(f"{db.query(q)[0]['n']} {label}" for label, q in counts.items()) + ".")
