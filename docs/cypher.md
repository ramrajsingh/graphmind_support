# Cypher queries GraphMind Support depends on

All graph access is in [`app/memory.py`](../app/memory.py) (queries), [`app/schema.py`](../app/schema.py) (indexes) and [`scripts/setup_graph.py`](../scripts/setup_graph.py) (seed). Every query is scoped by `$cid`, the caller's `customer_id`, which the server binds per call. Each turn's queries and their timings are shown in the UI under *Agent memory calls & Cypher*.

## Indexes

```cypher
CREATE INDEX FOR (c:Customer) ON (c.customer_id)
CREATE INDEX FOR (t:Ticket) ON (t.id)
CREATE INDEX FOR (v:Conversation) ON (v.id)
CREATE INDEX FOR (p:Product) ON (p.key)
CREATE INDEX FOR (s:TroubleshootingStep) ON (s.key)
CREATE INDEX FOR (k:KnownIssue) ON (k.key)
CALL db.idx.fulltext.createNodeIndex('KnownIssue', 'name', 'symptoms')
CREATE VECTOR INDEX FOR (k:KnownIssue) ON (k.embedding) OPTIONS {dimension:1536, similarityFunction:'cosine'}
```

## Customer state (read at call start)

### Customer and products: `snapshot()`
```cypher
MATCH (c:Customer {customer_id: $cid})
OPTIONAL MATCH (c)-[:OWNS]->(p:Product)
RETURN c.name AS name, c.preferred_language AS preferred_language, collect(p.name) AS products
```

### Tickets with what was tried and the next untried fixes: `snapshot()`
This is the core multi-hop query. Its result is preloaded into the agent's context, so a callback is answered without an extra tool call.
```cypher
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
ORDER BY created
```

### Ticket lookup: tools that take a ticket reference
```cypher
MATCH (c:Customer {customer_id: $cid})-[:OPENED]->(t:Ticket)
RETURN t.id AS id, t.status AS status, t.created_at AS created ORDER BY created
```

## Semantic memory: known issues and fixes

### Known fixes, minus steps this ticket already tried: `find_known_fixes`, `start_new_issue`
A vector search over known issues, then a traversal to ranked fixes, with each fix marked tried or untried for the ticket. A cosine distance below 0.5 counts as the same problem. Without an OpenAI key, the first line falls back to `CALL db.idx.fulltext.queryNodes('KnownIssue', $terms)`.
```cypher
CALL db.idx.vector.queryNodes('KnownIssue', 'embedding', 6, vecf32($vec)) YIELD node AS k, score
WITH k, score WHERE score < $maxdist
OPTIONAL MATCH (k)-[:AFFECTS]->(p:Product)
WITH k, score, p WHERE $pkey IS NULL OR p IS NULL OR p.key = $pkey
MATCH (k)-[rb:RESOLVED_BY]->(s:TroubleshootingStep)
OPTIONAL MATCH (t:Ticket {id: $tid})-[tr:TRIED]->(s)
WITH k, score, p, s, count(tr) > 0 AS tried, coalesce(rb.rank, 99) AS rank ORDER BY rank
RETURN k.name AS known_issue, k.symptoms AS symptoms, p.name AS product, score,
       collect({step: s.name, instructions: s.instructions, tried: tried}) AS fixes
ORDER BY score ASC LIMIT 3
```

## Writes: tickets and steps

### Open a ticket linked to the product and matched known issue: `start_new_issue`
```cypher
MATCH (c:Customer {customer_id: $cid})
CREATE (c)-[:OPENED]->(t:Ticket {id: $tid, status: 'open', priority: 'normal', contacts: 1,
        summary: $summary, created_at: timestamp(), updated_at: timestamp()})
WITH t OPTIONAL MATCH (p:Product {key: $pkey})
WITH t, p OPTIONAL MATCH (k:KnownIssue {key: $kkey})
FOREACH (_ IN CASE WHEN p IS NULL THEN [] ELSE [1] END | CREATE (t)-[:ABOUT]->(p))
FOREACH (_ IN CASE WHEN k IS NULL THEN [] ELSE [1] END | CREATE (t)-[:MATCHES]->(k))
```

### Record a step outcome: `record_step`
The relationship is `TRIED` when the step failed and `RESOLVED_WITH` when it worked.
```cypher
MATCH (c:Customer {customer_id: $cid})-[:OPENED]->(t:Ticket {id: $tid})
MERGE (s:TroubleshootingStep {key: $skey}) ON CREATE SET s.name = $step, s.instructions = ''
CREATE (t)-[:TRIED {at: timestamp()}]->(s)
```

### Update a ticket, raising priority on repeat contact: `update_ticket`
Repeat contacts count once per call. The second contact raises priority to high and the third to urgent; priority never goes down.
```cypher
MATCH (c:Customer {customer_id: $cid})-[:OPENED]->(t:Ticket {id: $tid})
WITH t, t.contacts + $bump AS contacts
WITH t, contacts, coalesce($priority, t.priority) AS wanted,
     CASE WHEN contacts >= 3 THEN 'urgent' WHEN contacts >= 2 THEN 'high' ELSE 'normal' END AS floor
SET t.contacts = contacts,
    t.status = coalesce($status, t.status),
    t.priority = CASE WHEN $rank[floor] > $rank[wanted] THEN floor ELSE wanted END,
    t.summary = CASE WHEN $note IS NULL THEN t.summary ELSE t.summary + ' [' + $note + ']' END,
    t.updated_at = timestamp()
RETURN t.id AS ticket, t.status AS status, t.priority AS priority, t.contacts AS contacts, t.summary AS summary
```

## Learning loop: promote a confirmed fix to shared knowledge

### Add the fix to the ticket's known issue
```cypher
MATCH (t:Ticket {id: $tid})-[:MATCHES]->(k:KnownIssue), (s:TroubleshootingStep {key: $skey})
OPTIONAL MATCH (k)-[existing:RESOLVED_BY]->(s)
WITH k, s, existing
FOREACH (_ IN CASE WHEN existing IS NULL THEN [1] ELSE [] END |
    CREATE (k)-[:RESOLVED_BY {learned: true, at: timestamp()}]->(s))
RETURN k.name AS known_issue, existing IS NULL AS added
```

### Or create a new known issue from the ticket, with its fix and product
```cypher
MATCH (t:Ticket {id: $tid}) OPTIONAL MATCH (t)-[:ABOUT]->(p:Product)
RETURN t.summary AS summary, p.key AS pkey
```
```cypher
MATCH (t:Ticket {id: $tid}), (s:TroubleshootingStep {key: $skey})
MERGE (k:KnownIssue {key: $kkey})
  ON CREATE SET k.name = $name, k.symptoms = $name, k.learned = true
CREATE (t)-[:MATCHES]->(k)
MERGE (k)-[r:RESOLVED_BY]->(s) ON CREATE SET r.learned = true, r.at = timestamp()
WITH k OPTIONAL MATCH (p:Product {key: $pkey})
FOREACH (_ IN CASE WHEN p IS NULL THEN [] ELSE [1] END | MERGE (k)-[:AFFECTS]->(p))
```
```cypher
MATCH (k:KnownIssue {key: $kkey}) SET k.embedding = vecf32($vec)
```

## Episodic memory: calls

### Earlier call summaries, read at call start: `past_call_summaries`
```cypher
MATCH (c:Customer {customer_id: $cid})-[:HAD_CALL]->(v:Conversation)
WHERE v.summary IS NOT NULL
RETURN v.summary AS summary, v.started_at AS started ORDER BY started DESC LIMIT $limit
```

### Start a call, log each message, link tickets discussed, save the summary at hang-up
```cypher
MATCH (c:Customer {customer_id: $cid})
CREATE (c)-[:HAD_CALL]->(:Conversation {id: $vid, started_at: timestamp(), language: $lang})
```
```cypher
MATCH (v:Conversation {id: $vid})
CREATE (v)-[:HAS_MESSAGE]->(:Message {role: $role, content: $content, original: $original,
                                      language: $lang, at: timestamp()})
```
```cypher
MATCH (v:Conversation {id: $vid}), (t:Ticket {id: $tid}) MERGE (v)-[:DISCUSSED]->(t)
```
```cypher
MATCH (v:Conversation {id: $vid}) SET v.summary = $summary, v.ended_at = timestamp()
```
