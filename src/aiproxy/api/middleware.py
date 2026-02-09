"""Flask middleware registration for auth, rate limiting, and logging."""
import time
import uuid
from flask import g, request, Response

from ..core.config import (
    get_access_keys,
    get_access_key_config,
    get_cors_config,
    get_rate_limit_config,
    get_logging_config,
)
from ..utils.http import error_response, get_client_ip
from ..utils.logging import log_event, redact_headers, redact_payload, should_log_request


def _get_allowed_keys():
    config_keys = get_access_keys()
    if config_keys:
        return set(config_keys.keys())
    return set()


def _extract_request_key():
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    for header_name in ("x-api-key", "api-key"):
        header_value = request.headers.get(header_name)
        if header_value:
            return header_value.strip()
    if request.is_json:
        body = request.get_json(silent=True) or {}
        if isinstance(body, dict):
            if "apiKey" in body and isinstance(body["apiKey"], str):
                return body["apiKey"].strip()
            if "api_key" in body and isinstance(body["api_key"], str):
                return body["api_key"].strip()
    return ""


def register_middlewares(app, settings, rate_limiter):
    """Register Flask middlewares on the app."""

    @app.before_request
    def attach_request_context():
        g.request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
        g.request_start = time.time()

    @app.before_request
    def enforce_inbound_key():
        allowed_keys = _get_allowed_keys()
        if not allowed_keys:
            return None
        if request.method == "OPTIONS":
            return None
        if request.method == "GET" and request.path in ("/", "/favicon.ico", "/healthz", "/version"):
            return None
        key = _extract_request_key()
        if key in allowed_keys:
            key_config = get_access_key_config(key)
            if key_config and isinstance(key_config, dict):
                g.allowed_models = key_config.get("models")
            else:
                g.allowed_models = None
            g.inbound_key = key
            return None
        return error_response("Unauthorized", 401, "authentication_error")

    @app.before_request
    def enforce_rate_limit():
        if request.method == "OPTIONS":
            return None
        if request.method == "GET" and request.path in ("/", "/favicon.ico", "/healthz", "/version"):
            return None
        rate_limit = get_rate_limit_config()
        limit = rate_limit.get("requests")
        window_seconds = rate_limit.get("window_seconds")
        if not limit or not window_seconds:
            return None
        key = getattr(g, "inbound_key", None) or get_client_ip()
        allowed, remaining, reset_seconds = rate_limiter.allow(key, limit, window_seconds)
        g.rate_limit_limit = limit
        g.rate_limit_remaining = remaining
        g.rate_limit_reset = reset_seconds
        if not allowed:
            response, status = error_response("Rate limit exceeded", 429, "rate_limit_error")
            response.headers["Retry-After"] = str(reset_seconds or 0)
            response.headers["X-RateLimit-Limit"] = str(limit)
            response.headers["X-RateLimit-Remaining"] = "0"
            response.headers["X-RateLimit-Reset"] = str(reset_seconds or 0)
            return response, status
        return None

    @app.after_request
    def add_headers(response):
        response.headers["X-Request-ID"] = getattr(g, "request_id", "")
        if hasattr(g, "rate_limit_limit") and g.rate_limit_limit:
            response.headers["X-RateLimit-Limit"] = str(g.rate_limit_limit)
            response.headers["X-RateLimit-Remaining"] = str(g.rate_limit_remaining)
            response.headers["X-RateLimit-Reset"] = str(g.rate_limit_reset)

        cors = get_cors_config()
        origins = cors.get("origins", [])
        request_origin = request.headers.get("Origin")
        allow_origin = None
        if "*" in origins:
            allow_origin = "*"
        elif request_origin and request_origin in origins:
            allow_origin = request_origin
            response.headers["Vary"] = "Origin"
        if allow_origin:
            response.headers["Access-Control-Allow-Origin"] = allow_origin
            response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, X-API-KEY, API-KEY, X-Request-ID"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            if cors.get("allow_credentials"):
                response.headers["Access-Control-Allow-Credentials"] = "true"

        latency_ms = None
        if hasattr(g, "request_start"):
            latency_ms = int((time.time() - g.request_start) * 1000)
        if request.path == "/healthz":
            return response
        if should_log_request(response.status_code):
            log_event(
                20,
                "request",
                request_id=getattr(g, "request_id", ""),
                method=request.method,
                path=request.path,
                status=response.status_code,
                latency_ms=latency_ms,
                model=getattr(g, "resolved_model", None),
                provider=getattr(g, "resolved_provider", None),
                provider_url=getattr(g, "resolved_provider_url", None),
                upstream_url=getattr(g, "upstream_url", None),
                client_ip=get_client_ip(),
            )
            log_cfg = get_logging_config()
            if log_cfg.get("include_headers") or log_cfg.get("include_body"):
                detail = {}
                if log_cfg.get("include_headers"):
                    detail["headers"] = redact_headers(dict(request.headers), log_cfg.get("redact_headers", []))
                if log_cfg.get("include_body") and request.is_json:
                    body = request.get_json(silent=True)
                    detail["body"] = redact_payload(body, log_cfg.get("redact_keys", []))
                if detail:
                    log_event(20, "request_detail", request_id=getattr(g, "request_id", ""), **detail)
        return response

    @app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
    @app.route('/<path:path>', methods=['OPTIONS'])
    def options_handler(path):
        return Response(status=204)
