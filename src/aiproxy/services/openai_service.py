"""OpenAI client helpers with timeout/retry support."""
from typing import Any, Iterable, List, Optional

import openai


def create_client(api_key: str, base_url: Optional[str], timeout: float, max_retries: int) -> openai.OpenAI:
    """Create a configured OpenAI client."""
    if base_url:
        return openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
    return openai.OpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)


def create_chat_completion(
    client: openai.OpenAI,
    model: str,
    messages: List[dict],
    **kwargs: Any,
) -> Any:
    """Call chat.completions and return the full response."""
    return client.chat.completions.create(model=model, messages=messages, **kwargs)


def create_response(client: openai.OpenAI, **kwargs: Any) -> Any:
    """Call responses.create and return the full response."""
    return client.responses.create(**kwargs)


def stream_chat_completion(
    client: openai.OpenAI,
    model: str,
    messages: List[dict],
    **kwargs: Any,
) -> Iterable[Any]:
    """Stream chat.completions chunks."""
    return client.chat.completions.create(model=model, messages=messages, stream=True, **kwargs)


def stream_response(client: openai.OpenAI, **kwargs: Any) -> Iterable[Any]:
    """Stream responses events."""
    return client.responses.create(stream=True, **kwargs)
