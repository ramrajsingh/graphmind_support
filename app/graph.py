"""FalkorDB connection, query helper and embeddings.

Every query the agent's tools run is recorded per call, so the UI can show the exact Cypher
behind each answer.
"""

import threading
import time
from typing import Any

from falkordb import FalkorDB

from app import config


class GraphError(RuntimeError):
    pass


class GraphDB:
    def __init__(self):
        self.db = FalkorDB(host=config.FALKORDB_HOST, port=config.FALKORDB_PORT,
                           username=config.FALKORDB_USERNAME, password=config.FALKORDB_PASSWORD)
        self.g = self.db.select_graph(config.FALKORDB_GRAPH)

    def query(self, cypher: str, params: dict | None = None, log: list | None = None,
              label: str = "") -> list[dict[str, Any]]:
        """Run Cypher and return rows as dicts keyed by the RETURN aliases."""
        t = time.time()
        try:
            res = self.g.query(cypher, params or {})
        except Exception as e:  # the client raises redis ResponseError for Cypher errors
            raise GraphError(f"{label or 'query'} failed: {e}") from e
        if log is not None:
            log.append({"label": label, "cypher": " ".join(cypher.split()), "ms": int((time.time() - t) * 1000),
                        "params": {k: v for k, v in (params or {}).items() if k != "vec"}})
        header = [h[1] if isinstance(h, (list, tuple)) else h for h in (res.header or [])]
        return [dict(zip(header, row)) for row in res.result_set]


_embed_client = None
_embed_lock = threading.Lock()


def embed(text: str) -> list[float] | None:
    """OpenAI embedding for semantic search, or None when no OpenAI key is configured."""
    global _embed_client
    if not config.OPENAI_API_KEY:
        return None
    with _embed_lock:
        if _embed_client is None:
            from openai import OpenAI
            _embed_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _embed_client.embeddings.create(model=config.EMBED_MODEL, input=text).data[0].embedding
