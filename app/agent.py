"""Support agent prompt and tools, plus the Claude tool loop. Reasons in English over FalkorDB graph memory."""

import json
from dataclasses import dataclass, field

import anthropic

from app import config
from app.memory import SupportMemory
from app.graph import GraphError

SYSTEM = """You are Asha, a phone support agent for Acme (Acme Pay app, Acme Fiber broadband, Acme Smart Router).

You are speaking on a voice call. Your words are translated into the caller's language and read aloud, so:
- Always write in plain English, at most 2 short sentences (about 35 words), no lists, markdown, emoji or URLs.
- Keep product names and technical terms (Acme Pay, OTP, 2FA, UPI, Wi-Fi, router, app version numbers) exactly as written.
- Give one troubleshooting step at a time, then wait for the caller to say whether it worked.

You have long-term memory of this customer in <customer_memory> below. It is already loaded at the start of the call, so don't call get_customer_context unless you've lost track.
- Never ask the caller to repeat something memory already tells you. Greet returning callers by name and show you remember: mention the open ticket and what they already tried.
- Only suggest troubleshooting steps that come from memory or tools: untried_known_fixes on an open ticket, or find_known_fixes results. Never invent a step or claim it worked for others unless a tool returned it. Never suggest a step listed in tried_and_failed. Suggest untried fixes in the order they are listed.
- If the caller is calling back about an open ticket, call update_ticket with repeat_contact=true once in this call.
- If it's a new problem, call start_new_issue before suggesting the first fix. Every new problem gets a ticket.
- As soon as the caller says whether a step worked, call record_step.
- If no untried fixes remain, call update_ticket with status="escalated" and tell the caller a specialist will call back within 4 hours.
- Don't read out ticket IDs unless asked."""

TOOLS = [
    {
        "name": "get_customer_context",
        "description": "Reload this customer's memory: profile, products, tickets with what was tried and what fixed them, and summaries of earlier calls. Use if memory may be stale.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": True,
    },
    {
        "name": "find_known_fixes",
        "description": "Search shared knowledge for known issues matching the problem and return fixes that worked for other customers, excluding steps already tried on the given ticket. For a brand-new problem use start_new_issue instead.",
        "input_schema": {
            "type": "object",
            "properties": {
                "problem": {"type": "string", "description": "Short English description of the symptom"},
                "product": {"type": ["string", "null"], "description": "Exact product name, if known"},
                "ticket": {"type": ["string", "null"], "description": "Ticket name, to exclude steps already tried"},
            },
            "required": ["problem", "product", "ticket"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "start_new_issue",
        "description": "For a problem not already tracked in memory: opens a ticket linked to the best-matching known issue and returns the known fixes, in one call.",
        "input_schema": {
            "type": "object",
            "properties": {
                "problem": {"type": "string", "description": "One-line English summary of the problem"},
                "product": {"type": "string", "description": "Exact product name"},
            },
            "required": ["problem", "product"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "record_step",
        "description": "Record the outcome of a troubleshooting step on a ticket. A step that worked resolves the ticket and is added to shared knowledge.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket": {"type": "string"},
                "step": {"type": "string", "description": "Step name, matching find_known_fixes wording when possible"},
                "outcome": {"type": "string", "enum": ["worked", "failed"]},
            },
            "required": ["ticket", "step", "outcome"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "update_ticket",
        "description": "Update a ticket: mark a repeat contact (auto-raises priority), escalate, or add a note.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket": {"type": "string"},
                "status": {"type": ["string", "null"], "enum": ["open", "escalated", "resolved", None]},
                "priority": {"type": ["string", "null"], "enum": ["normal", "high", "urgent", None]},
                "repeat_contact": {"type": "boolean"},
                "note": {"type": ["string", "null"]},
            },
            "required": ["ticket", "status", "priority", "repeat_contact", "note"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


# Server-side refusal fallback, for the models that support the "default" form.
FALLBACK = ({"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
            if config.CLAUDE_MODEL in {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}
            else {})


SUMMARY_PROMPT = ("Summarize this support call in one or two sentences for the agent who takes this customer's "
                  "next call: the problem, the steps tried and their outcomes, and how the call ended. "
                  "Plain English, no preamble.")


def transcript(s: "CallSession") -> str:
    lines = []
    for m in s.messages:
        if isinstance(m.get("content"), str) and m["role"] in ("user", "assistant"):
            lines.append(f"{'Customer' if m['role'] == 'user' else 'Agent'}: {m['content']}")
    return "\n".join(lines)


@dataclass
class CallSession:
    call_id: str
    customer_id: str
    conversation_id: str
    memory: SupportMemory
    language: str = "en-IN"
    memory_text: str = ""
    messages: list = field(default_factory=list)


def render_memory(snapshot: dict, past: list[str]) -> str:
    return json.dumps({**snapshot, "earlier_call_summaries": past}, ensure_ascii=False, indent=1)


def run_tool(s: CallSession, name: str, args: dict) -> dict:
    """Execute a memory tool for this call's customer. Shared by the Claude and OpenAI agents."""
    m = s.memory
    if name == "get_customer_context":
        s.memory_text = render_memory(m.snapshot(), m.past_call_summaries())
        return json.loads(s.memory_text)
    if name == "find_known_fixes":
        return m.find_known_fixes(args["problem"], args.get("product"), args.get("ticket"))
    if name == "start_new_issue":
        return m.start_new_issue(args["problem"], args["product"])
    if name == "record_step":
        return m.record_step(args["ticket"], args["step"], args["outcome"])
    if name == "update_ticket":
        return m.update_ticket(args["ticket"], args.get("status"), args.get("priority"),
                               bool(args.get("repeat_contact")), args.get("note"))
    raise ValueError(f"unknown tool {name}")


class SupportAgent:
    def __init__(self):
        self.client = anthropic.Anthropic()

    def summarize(self, s: CallSession) -> str:
        resp = self.client.messages.create(
            model=config.CLAUDE_MODEL, max_tokens=1000, output_config={"effort": "low"},
            system=SUMMARY_PROMPT, messages=[{"role": "user", "content": transcript(s)}])
        return " ".join(b.text for b in resp.content if b.type == "text").strip()

    def respond(self, s: CallSession, english: str, original: str, language: str) -> tuple[str, list[dict]]:
        """Run one caller turn. Returns (English reply, tool trace)."""
        note = "" if language == "en-IN" else f'\n[Caller spoke {language}; their words: "{original}"]'
        s.messages.append({"role": "user", "content": english + note})
        system = [
            {"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": f"<customer_memory>\n{s.memory_text}\n</customer_memory>"},
        ]
        trace: list[dict] = []
        for _ in range(8):
            resp = self.client.beta.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=4000,
                system=system,
                tools=TOOLS,
                messages=s.messages,
                output_config={"effort": config.CLAUDE_EFFORT},
                **FALLBACK,
            )
            if resp.stop_reason == "refusal":
                text = "Sorry, I can't help with that on this line. Is there anything else about your Acme service?"
                s.messages.append({"role": "assistant", "content": text})  # keep history valid for the next turn
                return text, trace
            s.messages.append({"role": "assistant", "content": resp.content})
            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            if resp.stop_reason != "tool_use" or not tool_uses:
                text = " ".join(b.text for b in resp.content if b.type == "text").strip()
                return text, trace
            results = []
            for tu in tool_uses:
                try:
                    out = run_tool(s, tu.name, tu.input)
                    results.append({"type": "tool_result", "tool_use_id": tu.id,
                                    "content": json.dumps(out, ensure_ascii=False)})
                    trace.append({"tool": tu.name, "input": tu.input, "output": out})
                except (GraphError, ValueError, KeyError) as e:
                    results.append({"type": "tool_result", "tool_use_id": tu.id,
                                    "content": f"Error: {e}", "is_error": True})
                    trace.append({"tool": tu.name, "input": tu.input, "error": str(e)})
            s.messages.append({"role": "user", "content": results})
        return "Sorry, give me a moment, I'm having trouble pulling that up.", trace
