"""Tool-calling chat agent + a cheap one-paragraph narrative summary.
Degrades gracefully (chat tab hidden, narrative skipped) if no
ANTHROPIC_API_KEY is set - checked by callers via is_configured(), not
assumed.
"""

import json
import os

from agent.tools import TOOLS, execute_tool

CHAT_MODEL = os.environ.get("AGENT_CHAT_MODEL", "claude-sonnet-5")
NARRATIVE_MODEL = os.environ.get("AGENT_NARRATIVE_MODEL", "claude-haiku-4-5-20251001")
MAX_TOOL_ROUNDS = 6

SYSTEM_PROMPT = (
    "You are a financial analyst assistant for a personal spending dashboard. Answer only "
    "using the tools provided - every number in your answer must come from a tool result, "
    "never invented or estimated. If a tool returns an error or no data, say so plainly "
    "rather than guessing or making something up. Be concise (a few sentences, not an "
    "essay) and cite the category and time horizon when giving a number."
)


def is_configured() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _client():
    import anthropic

    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def _extract_text(content) -> str:
    return "".join(block.text for block in content if block.type == "text")


def chat(user_message: str, view: dict, history: list[dict] | None = None) -> tuple[str, list[dict]]:
    """Returns (assistant_text, updated_history). `history` is a list of
    plain dicts with the Anthropic Messages API shape - pass back what you
    got last time to continue a conversation."""
    if not is_configured():
        return "Chat isn't available - no ANTHROPIC_API_KEY configured.", history or []

    client = _client()
    messages = list(history or [])
    messages.append({"role": "user", "content": user_message})

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.messages.create(
            model=CHAT_MODEL, max_tokens=1024, system=SYSTEM_PROMPT,
            tools=TOOLS, messages=messages,
        )
        messages.append({"role": "assistant", "content": [b.model_dump() for b in response.content]})

        if response.stop_reason != "tool_use":
            return _extract_text(response.content), messages

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = execute_tool(block.name, block.input, view)
                tool_results.append({
                    "type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result),
                })
        messages.append({"role": "user", "content": tool_results})

    return (
        "I wasn't able to finish answering that within the tool-call budget - "
        "try asking a more specific question."
    ), messages


def generate_narrative(view: dict) -> str | None:
    """One short paragraph for the Overview tab banner. Returns None (not an
    error string) if the agent isn't configured or the call fails, so
    callers can silently skip the banner."""
    if not is_configured():
        return None

    health = execute_tool("get_health_score", {}, view)
    anomalies = execute_tool("get_anomalies", {}, view)
    top_categories = (
        view["baseline"].sort_values("total_spend", ascending=False).head(3)["category"].tolist()
        if not view["baseline"].empty else []
    )

    prompt = (
        f"Health score data: {json.dumps(health)}. Top spend categories: {top_categories}. "
        f"Active alerts: {json.dumps(anomalies.get('alerts', [])[:3])}. "
        "Write one short paragraph (2-3 sentences) summarizing this customer's financial "
        "situation and one concrete, specific suggestion. Plain text, no markdown, no preamble."
    )
    try:
        client = _client()
        response = client.messages.create(
            model=NARRATIVE_MODEL, max_tokens=220,
            messages=[{"role": "user", "content": prompt}],
        )
        return _extract_text(response.content).strip() or None
    except Exception:
        return None
