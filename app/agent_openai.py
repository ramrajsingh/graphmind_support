"""OpenAI version of the support agent: same prompt, tools and memory, Chat Completions tool loop."""

import json

from openai import OpenAI

from app import config
from app.agent import SUMMARY_PROMPT, SYSTEM, TOOLS, CallSession, run_tool, transcript
from app.memory import pmap
from app.graph import GraphError

OPENAI_TOOLS = [
    {"type": "function", "function": {
        "name": t["name"], "description": t["description"], "parameters": t["input_schema"], "strict": True,
    }}
    for t in TOOLS
]


class OpenAISupportAgent:
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY or None)

    def _call(self, s: CallSession, tc) -> tuple[str, dict]:
        try:
            args = json.loads(tc.function.arguments or "{}")
            out = run_tool(s, tc.function.name, args)
            return json.dumps(out, ensure_ascii=False), {"tool": tc.function.name, "input": args, "output": out}
        except (GraphError, ValueError, KeyError) as e:  # JSONDecodeError is a ValueError
            return f"Error: {e}", {"tool": tc.function.name, "input": tc.function.arguments, "error": str(e)}

    def summarize(self, s: CallSession) -> str:
        resp = self.client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[{"role": "system", "content": SUMMARY_PROMPT}, {"role": "user", "content": transcript(s)}])
        return (resp.choices[0].message.content or "").strip()

    def respond(self, s: CallSession, english: str, original: str, language: str) -> tuple[str, list[dict]]:
        """Run one caller turn. Returns (English reply, tool trace)."""
        note = "" if language == "en-IN" else f'\n[Caller spoke {language}; their words: "{original}"]'
        s.messages.append({"role": "user", "content": english + note})
        system = {"role": "system",
                  "content": f"{SYSTEM}\n\n<customer_memory>\n{s.memory_text}\n</customer_memory>"}
        trace: list[dict] = []
        for _ in range(8):
            resp = self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[system, *s.messages],
                tools=OPENAI_TOOLS,
            )
            msg = resp.choices[0].message
            if not msg.tool_calls:
                text = (msg.content or msg.refusal or "").strip()
                s.messages.append({"role": "assistant", "content": text})
                return text, trace
            s.messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [{"id": tc.id, "type": "function",
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                               for tc in msg.tool_calls],
            })
            # start_new_issue runs first so record_step/update_ticket issued alongside it can find the
            # ticket; the rest are independent and run in parallel.
            first = [tc for tc in msg.tool_calls if tc.function.name == "start_new_issue"]
            rest = [tc for tc in msg.tool_calls if tc.function.name != "start_new_issue"]
            results = {tc.id: self._call(s, tc) for tc in first}
            results.update(zip((tc.id for tc in rest), pmap(lambda tc: self._call(s, tc), rest)))
            for tc in msg.tool_calls:
                content, entry = results[tc.id]
                trace.append(entry)
                s.messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
        return "Sorry, give me a moment, I'm having trouble pulling that up.", trace
