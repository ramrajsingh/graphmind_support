"""FastAPI server: browser voice UI <-> Sarvam <-> LLM agent <-> FalkorDB graph memory.

    uvicorn app.server:app --reload --port 8000
"""

import json
import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import config
from app.agent import CallSession, SupportAgent, render_memory
from app.graph import GraphDB, GraphError
from app.memory import POOL, Graph, SupportMemory
from app.voice import Voice

log = logging.getLogger("support")
ROOT = Path(__file__).resolve().parent.parent
SEED = json.loads((ROOT / "data" / "seed.json").read_text())

app = FastAPI(title="GraphMind Support")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

db = GraphDB()
voice = Voice()
if config.LLM_PROVIDER == "openai":
    from app.agent_openai import OpenAISupportAgent
    agent = OpenAISupportAgent()
else:
    agent = SupportAgent()
calls: dict[str, CallSession] = {}


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/customers")
def customers():
    return [{k: c.get(k) for k in ("customer_id", "name", "phone", "preferred_language", "gender")}
            for c in SEED["customers"]]


GENDER = {c["customer_id"]: c.get("gender", "male") for c in SEED["customers"]}


class StartCall(BaseModel):
    customer_id: str


@app.post("/api/call/start")
def start_call(req: StartCall):
    mem = SupportMemory(db, req.customer_id)
    t0 = time.time()
    try:
        past_f = POOL.submit(mem.past_call_summaries)
        snapshot = mem.snapshot()
        if not snapshot["customer"]:
            raise HTTPException(404, "Customer not in FalkorDB. Run: python -m scripts.setup_graph")
        past = past_f.result()
        lang = snapshot["customer"].get("preferred_language") or "en-IN"
        conversation_id = mem.start_conversation(lang)
    except GraphError as e:
        raise HTTPException(502, str(e))
    s = CallSession(call_id=uuid.uuid4().hex, customer_id=req.customer_id, conversation_id=conversation_id,
                    memory=mem, language=lang, memory_text=render_memory(snapshot, past))
    calls[s.call_id] = s

    first = (snapshot["customer"]["name"] or "").split(" ")[0]
    greeting_en = f"Hi {first}, this is Asha from Acme support. How can I help you today?"
    greeting = voice.from_english(greeting_en, lang)
    audio, spoken = voice.speak(greeting, lang)
    s.messages.append({"role": "user", "content": "[Call connected]"})
    s.messages.append({"role": "assistant", "content": greeting_en})
    _log(s, "assistant", greeting_en, greeting, lang)
    return {
        "call_id": s.call_id, "language": lang, "reply": greeting, "reply_en": greeting_en,
        "audio_chunks": audio, "spoken_language": spoken,
        "memory": json.loads(s.memory_text), "graph": mem.graph.to_json(), "queries": mem.queries,
        "timing_ms": {"memory": int((time.time() - t0) * 1000)},
    }


@app.post("/api/call/{call_id}/turn")
def turn(call_id: str, audio: UploadFile | None = File(None), text: str | None = Form(None)):
    s = calls.get(call_id)
    if not s:
        raise HTTPException(404, "Unknown call")
    timing = {}
    t = time.time()
    if audio is not None:
        data = audio.file.read()
        codec = (audio.content_type or "audio/webm").split("/")[-1].split(";")[0]
        heard = voice.transcribe(data, audio.filename or f"turn.{codec}", codec)
    elif text:
        heard = voice.from_text(text)
    else:
        raise HTTPException(400, "Send audio or text")
    # Typed lines are read aloud in a caller voice while the agent works, so they add no wait.
    caller_f = (POOL.submit(voice.speak_caller, heard.original, heard.language, GENDER.get(s.customer_id, "male"))
                if audio is None and heard.original.strip() else None)
    timing["stt+translate"] = _ms(t)
    if not heard.original.strip():
        return {"heard": "", "reply": "", "audio_chunks": []}
    s.language = heard.language
    _log(s, "user", heard.english, heard.original, heard.language)

    t = time.time()
    s.memory.graph = Graph()  # only this turn's nodes; the UI merges them into its graph
    s.memory.queries = []     # only this turn's Cypher, shown in the UI
    reply_en, trace = agent.respond(s, heard.english, heard.original, heard.language)
    timing["agent"] = _ms(t)

    t = time.time()
    reply = voice.from_english(reply_en, s.language)
    audio_chunks, spoken = voice.speak(reply, s.language)
    timing["translate+tts"] = _ms(t)
    _log(s, "assistant", reply_en, reply, s.language)

    return {
        "heard": heard.original, "heard_en": heard.english, "language": heard.language,
        "reply": reply, "reply_en": reply_en, "audio_chunks": audio_chunks, "spoken_language": spoken,
        "caller_audio_chunks": caller_f.result() if caller_f else [],
        "tools": trace, "graph": s.memory.graph.to_json(), "queries": s.memory.queries, "timing_ms": timing,
    }


@app.post("/api/call/{call_id}/end")
def end_call(call_id: str):
    """End the call and write its summary: the episodic memory the next call starts from."""
    s = calls.pop(call_id, None)
    if not s:
        return {"ok": True, "summary": None}
    if len(s.messages) <= 2:  # greeting only, nothing to remember
        return {"ok": True, "summary": None}
    try:
        summary = agent.summarize(s)
        s.memory.end_conversation(summary)
    except Exception as e:  # don't fail hang-up over the summary
        log.warning("call summary failed: %s", e)
        return {"ok": True, "summary": None}
    return {"ok": True, "summary": summary}


def _log(s: CallSession, role: str, content: str, original: str, language: str):
    """Append the turn to the call's Conversation node in the background; it isn't needed to answer."""
    def write():
        try:
            s.memory.log_message(role, content, original, language)
        except GraphError as e:
            log.warning("message log failed: %s", e)
    POOL.submit(write)


def _ms(t0: float) -> int:
    return int((time.time() - t0) * 1000)
