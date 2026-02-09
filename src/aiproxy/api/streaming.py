"""SSE streaming helpers for chat and responses endpoints."""
import json
import time
import openai

from ..services.openai_service import stream_chat_completion
from ..utils.logging import log_event


def _get_first_choice(chunk):
    """Return the first choice from a streaming chunk, if present."""
    choices = getattr(chunk, "choices", None)
    if not choices:
        return None
    return choices[0]


def _dump_delta(delta):
    """Serialize delta objects to dict, stripping None fields."""
    if delta is None:
        return {}
    if hasattr(delta, "model_dump"):
        return delta.model_dump(exclude_none=True)
    if isinstance(delta, dict):
        return {key: value for key, value in delta.items() if value is not None}
    return {}


def _stream_error_payload(message, response_id, created, response_model, error_type):
    """Build a stream-safe error chunk with choices for client compatibility."""
    return {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": response_model,
        "choices": [
            {
                "delta": {"content": ""},
                "index": 0,
                "finish_reason": "error",
            }
        ],
        "error": {"message": message, "type": error_type},
    }


def stream_chat_sse(client, model, messages, response_model, response_id, created, **kwargs):
    """Stream chat.completions and emit SSE in OpenAI format."""
    try:
        stream = stream_chat_completion(client, model, messages, **kwargs)
        for chunk in stream:
            choice = _get_first_choice(chunk)
            if not choice:
                continue
            delta_payload = _dump_delta(getattr(choice, "delta", None))
            finish_reason = getattr(choice, "finish_reason", None)
            if not delta_payload and finish_reason is None:
                continue
            data = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": response_model,
                "choices": [
                    {
                        "delta": delta_payload,
                        "index": getattr(choice, "index", 0),
                        "finish_reason": finish_reason,
                    }
                ],
            }
            yield f"data: {json.dumps(data)}\n\n"

        yield "data: [DONE]\n\n"

    except openai.APIConnectionError as e:
        error_payload = _stream_error_payload(
            f"Failed to connect to API: {e}",
            response_id,
            created,
            response_model,
            "api_connection_error",
        )
        yield f"data: {json.dumps(error_payload)}\n\n"
        yield "data: [DONE]\n\n"
    except openai.RateLimitError as e:
        error_payload = _stream_error_payload(
            f"API request exceeded rate limit: {e}",
            response_id,
            created,
            response_model,
            "rate_limit_error",
        )
        yield f"data: {json.dumps(error_payload)}\n\n"
        yield "data: [DONE]\n\n"
    except openai.APIStatusError as e:
        error_payload = _stream_error_payload(
            f"API returned an API Status Error: {e}",
            response_id,
            created,
            response_model,
            "api_error",
        )
        yield f"data: {json.dumps(error_payload)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        error_payload = _stream_error_payload(
            f"An unexpected error occurred: {e}",
            response_id,
            created,
            response_model,
            "internal_error",
        )
        yield f"data: {json.dumps(error_payload)}\n\n"


def _log_stream_error(
    err,
    request_id: str | None,
    upstream_url: str | None,
    request_payload: dict | None = None,
):
    response = getattr(err, "response", None)
    status = getattr(err, "status_code", None) or getattr(response, "status_code", None)
    body = None
    if response is not None:
        try:
            body = response.text
        except Exception:
            body = None
    if isinstance(body, str) and len(body) > 2000:
        body = body[:2000] + "...(truncated)"
    log_event(
        40,
        "upstream_error",
        request_id=request_id or "",
        upstream_url=upstream_url,
        status=status,
        body=body,
        payload=request_payload,
    )


def stream_responses_sse(
    stream,
    request_id: str | None = None,
    upstream_url: str | None = None,
    request_payload: dict | None = None,
):
    """Stream /v1/responses SSE events from the upstream responses API."""
    try:
        for event in stream:
            if hasattr(event, "model_dump"):
                payload = event.model_dump()
            elif hasattr(event, "dict"):
                payload = event.dict()
            elif isinstance(event, dict):
                payload = event
            else:
                payload = {"type": "response.output_text.delta", "delta": str(event)}
            event_type = payload.get("type", "response.output_text.delta")
            yield f"event: {event_type}\n"
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    except openai.RateLimitError as e:
        _log_stream_error(e, request_id, upstream_url, request_payload)
        error_payload = {"type": "response.failed", "error": {"message": str(e), "type": "rate_limit_error"}}
        yield "event: response.failed\n"
        yield f"data: {json.dumps(error_payload)}\n\n"
    except openai.APIConnectionError as e:
        _log_stream_error(e, request_id, upstream_url, request_payload)
        error_payload = {"type": "response.failed", "error": {"message": str(e), "type": "api_connection_error"}}
        yield "event: response.failed\n"
        yield f"data: {json.dumps(error_payload)}\n\n"
    except openai.APIStatusError as e:
        _log_stream_error(e, request_id, upstream_url, request_payload)
        error_payload = {"type": "response.failed", "error": {"message": str(e), "type": "api_error"}}
        yield "event: response.failed\n"
        yield f"data: {json.dumps(error_payload)}\n\n"
    except Exception as e:
        _log_stream_error(e, request_id, upstream_url, request_payload)
        error_payload = {"type": "response.failed", "error": {"message": str(e), "type": "internal_error"}}
        yield "event: response.failed\n"
        yield f"data: {json.dumps(error_payload)}\n\n"


def stream_responses_sse_from_chat(
    client,
    model,
    messages,
    response_model,
    response_id,
    created,
    request_id: str | None = None,
    upstream_url: str | None = None,
    request_payload: dict | None = None,
    **kwargs,
):
    """Fallback: stream /v1/responses format using chat.completions."""
    try:
        stream = stream_chat_completion(client, model, messages, **kwargs)
        output_text = ""
        created_event = {
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "response",
                "created": created,
                "model": response_model,
                "output": [],
            },
        }
        yield "event: response.created\n"
        yield f"data: {json.dumps(created_event)}\n\n"

        for chunk in stream:
            choice = _get_first_choice(chunk)
            if not choice:
                continue
            delta = getattr(getattr(choice, "delta", None), "content", None)
            if not delta:
                continue
            output_text += delta
            delta_event = {
                "type": "response.output_text.delta",
                "delta": delta,
                "output_index": 0,
                "content_index": 0,
            }
            yield "event: response.output_text.delta\n"
            yield f"data: {json.dumps(delta_event)}\n\n"

        done_event = {
            "type": "response.output_text.done",
            "text": output_text,
            "output_index": 0,
            "content_index": 0,
        }
        yield "event: response.output_text.done\n"
        yield f"data: {json.dumps(done_event)}\n\n"

        completed_event = {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "object": "response",
                "created": created,
                "model": response_model,
                "output": [
                    {
                        "id": f"msg-{int(time.time() * 1000)}",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": output_text}],
                    }
                ],
            },
        }
        yield "event: response.completed\n"
        yield f"data: {json.dumps(completed_event)}\n\n"
    except openai.RateLimitError as e:
        _log_stream_error(e, request_id, upstream_url, request_payload)
        error_payload = {"type": "response.failed", "error": {"message": str(e), "type": "rate_limit_error"}}
        yield "event: response.failed\n"
        yield f"data: {json.dumps(error_payload)}\n\n"
    except openai.APIConnectionError as e:
        _log_stream_error(e, request_id, upstream_url, request_payload)
        error_payload = {"type": "response.failed", "error": {"message": str(e), "type": "api_connection_error"}}
        yield "event: response.failed\n"
        yield f"data: {json.dumps(error_payload)}\n\n"
    except openai.APIStatusError as e:
        _log_stream_error(e, request_id, upstream_url, request_payload)
        error_payload = {"type": "response.failed", "error": {"message": str(e), "type": "api_error"}}
        yield "event: response.failed\n"
        yield f"data: {json.dumps(error_payload)}\n\n"
    except Exception as e:
        _log_stream_error(e, request_id, upstream_url, request_payload)
        error_payload = {"type": "response.failed", "error": {"message": str(e), "type": "internal_error"}}
        yield "event: response.failed\n"
        yield f"data: {json.dumps(error_payload)}\n\n"
