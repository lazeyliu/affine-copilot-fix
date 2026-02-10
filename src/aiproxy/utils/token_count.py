"""Token counting helpers (best-effort)."""
from __future__ import annotations

import json
from typing import Iterable


def _get_encoder(model: str | None):
    try:
        import tiktoken  # type: ignore

        if model:
            return tiktoken.encoding_for_model(model)
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_text_tokens(text: str | None, model: str | None = None) -> int:
    if not text:
        return 0
    encoder = _get_encoder(model)
    if encoder:
        return len(encoder.encode(text))
    # fallback heuristic: ~4 chars per token
    return max(1, len(text) // 4)


def _flatten_message_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                text = item.get("text")
                if isinstance(text, dict):
                    parts.append(str(text.get("value") or text.get("text") or ""))
                else:
                    parts.append(str(text))
            elif isinstance(item, dict) and item.get("type") in ("input_text", "output_text"):
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    if isinstance(content, dict) and "text" in content:
        return str(content.get("text") or "")
    return str(content)


def _flatten_tool_payload(message: dict) -> str:
    parts: list[str] = []
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function") or {}
            if isinstance(function, dict):
                name = function.get("name")
                if isinstance(name, str):
                    parts.append(name)
                args = function.get("arguments")
                if isinstance(args, str):
                    parts.append(args)
                elif isinstance(args, dict):
                    parts.append(json.dumps(args, ensure_ascii=False))
    function_call = message.get("function_call")
    if isinstance(function_call, dict):
        name = function_call.get("name")
        if isinstance(name, str):
            parts.append(name)
        args = function_call.get("arguments")
        if isinstance(args, str):
            parts.append(args)
        elif isinstance(args, dict):
            parts.append(json.dumps(args, ensure_ascii=False))
    tool_result = message.get("tool_result")
    if isinstance(tool_result, dict):
        output = tool_result.get("content") or tool_result.get("output")
        if isinstance(output, str):
            parts.append(output)
        elif isinstance(output, dict):
            parts.append(json.dumps(output, ensure_ascii=False))
    return "".join(parts)


def count_messages_tokens(messages: Iterable[dict], model: str | None = None) -> int:
    total = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content_text = _flatten_message_content(msg.get("content"))
        tool_text = _flatten_tool_payload(msg)
        total += count_text_tokens(content_text + tool_text, model=model)
    return total
