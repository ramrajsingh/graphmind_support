"""Create indexes and load seed data into FalkorDB. Safe to re-run (everything is MERGEd).

    python -m scripts.setup_graph
"""

import json
from pathlib import Path

from app.graph import GraphDB, embed
from app.memory import pmap
from app.schema import create_indexes, key

SEED = json.loads((Path(__file__).resolve().parent.parent / "data" / "seed.json").read_text())


def seed(db: GraphDB):
    create_indexes(db)
    for p in SEED["products"]:
        db.query("MERGE (p:Product {key: $key}) SET p.name = $name, p.category = $category, "
                 "p.current_version = $version",
                 {"key": key(p["name"]), "name": p["name"], "category": p["category"],
                  "version": p["current_version"]})
    for s in SEED["steps"]:
        db.query("MERGE (s:TroubleshootingStep {key: $key}) SET s.name = $name, s.instructions = $instructions",
                 {"key": key(s["name"]), "name": s["name"], "instructions": s["instructions"]})
    vectors = pmap(lambda ki: embed(f"{ki['name']}. {ki['symptoms']} {ki['product']}"), SEED["known_issues"])
    for ki, vec in zip(SEED["known_issues"], vectors):
        db.query("""
            MERGE (k:KnownIssue {key: $key}) SET k.name = $name, k.symptoms = $symptoms
            WITH k MATCH (p:Product {key: $pkey}) MERGE (k)-[:AFFECTS]->(p)""",
                 {"key": key(ki["name"]), "name": ki["name"], "symptoms": ki["symptoms"], "pkey": key(ki["product"])})
        for rank, step in enumerate(ki["resolved_by"]):  # rank: the order support tries fixes in
            db.query("MATCH (k:KnownIssue {key: $k}), (s:TroubleshootingStep {key: $s}) "
                     "MERGE (k)-[r:RESOLVED_BY]->(s) SET r.rank = $rank",
                     {"k": key(ki["name"]), "s": key(step), "rank": rank})
        if vec is not None:
            db.query("MATCH (k:KnownIssue {key: $key}) SET k.embedding = vecf32($vec)",
                     {"key": key(ki["name"]), "vec": vec})
    for c in SEED["customers"]:
        db.query("""
            MERGE (c:Customer {customer_id: $cid})
            SET c.name = $name, c.preferred_language = $lang, c.phone = $phone, c.gender = $gender""",
                 {"cid": c["customer_id"], "name": c["name"], "lang": c["preferred_language"],
                  "phone": c["phone"], "gender": c.get("gender", "")})
        for prod in c["owns"]:
            db.query("MATCH (c:Customer {customer_id: $cid}), (p:Product {key: $pkey}) MERGE (c)-[:OWNS]->(p)",
                     {"cid": c["customer_id"], "pkey": key(prod)})
    counts = db.query("MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY label")
    print("graph:", ", ".join(f"{r['label']} {r['n']}" for r in counts))


if __name__ == "__main__":
    seed(GraphDB())
