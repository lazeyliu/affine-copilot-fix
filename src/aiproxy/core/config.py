"""Config loader and helpers for model/provider mapping."""
import json
import logging
import os
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
CONFIG_PATH = os.getenv("CONFIG_PATH", DEFAULT_CONFIG_PATH)

_CONFIG_CACHE = {
    "config": None,
    "mtime": None,
}

logger = logging.getLogger("aiproxy.config")


def _normalize_models(models, created_default):
    """Normalize model entries for consistent downstream usage."""
    normalized = []
    if not isinstance(models, list):
        return normalized
    for item in models:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        provider = item.get("provider")
        target_model = item.get("model")
        if not model_id or not provider or not target_model:
            continue
        created_value = item.get("created", created_default)
        try:
            created_value = int(created_value)
        except (TypeError, ValueError):
            created_value = created_default
        permission_value = item.get("permission", [])
        if not isinstance(permission_value, list):
            permission_value = []
        responses_cfg = _normalize_responses(item.get("responses", {}))
        normalized.append(
            {
                "id": model_id,
                "provider": provider,
                "model": target_model,
                "created": created_value,
                "owned_by": item.get("owned_by", "openai"),
                "permission": permission_value,
                "root": item.get("root", model_id),
                "parent": item.get("parent", None),
                "responses": responses_cfg,
            }
        )
    return normalized


def load_config():
    """Load and cache config.json with a simple mtime check."""
    path = CONFIG_PATH
    try:
        mtime = os.path.getmtime(path)
    except FileNotFoundError:
        return {"providers": {}, "models": [], "defaults": {}}

    cached = _CONFIG_CACHE["config"]
    if cached is not None and _CONFIG_CACHE["mtime"] == mtime:
        return cached

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"providers": {}, "models": [], "defaults": {}}

    if not isinstance(raw, dict):
        raw = {}

    created_default = int(time.time())
    config = {
        "providers": raw.get("providers", {}) if isinstance(raw.get("providers"), dict) else {},
        "models": _normalize_models(raw.get("models", []), created_default),
        "defaults": raw.get("defaults", {}) if isinstance(raw.get("defaults"), dict) else {},
        "access_keys": raw.get("access_keys", {}),
        "server": raw.get("server", {}) if isinstance(raw.get("server"), dict) else {},
        "rate_limit": raw.get("rate_limit", {}) if isinstance(raw.get("rate_limit"), dict) else {},
        "cors": raw.get("cors", {}) if isinstance(raw.get("cors"), dict) else {},
        "logging": raw.get("logging", {}) if isinstance(raw.get("logging"), dict) else {},
        "responses": raw.get("responses", {}) if isinstance(raw.get("responses"), dict) else {},
    }

    config_errors = validate_config(config)
    config["errors"] = config_errors
    if config_errors:
        logger.warning("Config validation warnings: %s", "; ".join(config_errors))

    _CONFIG_CACHE["config"] = config
    _CONFIG_CACHE["mtime"] = mtime
    return config


def get_models_response():
    """Return an OpenAI-compatible /v1/models list from config or env fallback."""
    config = load_config()
    models = config.get("models", [])
    if not models:
        return []

    response_models = []
    for item in models:
        response_models.append(
            {
                "id": item["id"],
                "object": "model",
                "created": item["created"],
                "owned_by": item.get("owned_by", "openai"),
                "permission": item.get("permission", []),
                "root": item.get("root", item["id"]),
                "parent": item.get("parent", None),
            }
        )
    return response_models


def _normalize_access_keys(access_keys):
    """Normalize access_keys into a dict of key -> config."""
    normalized = {}
    if isinstance(access_keys, str):
        if access_keys.strip():
            normalized[access_keys.strip()] = {"models": None}
        return normalized
    if isinstance(access_keys, dict):
        for key, value in access_keys.items():
            if not isinstance(key, str) or not key.strip():
                continue
            models = None
            if isinstance(value, dict):
                models_value = value.get("models")
                if isinstance(models_value, list):
                    models = [item for item in models_value if isinstance(item, str) and item.strip()]
            elif isinstance(value, list):
                models = [item for item in value if isinstance(item, str) and item.strip()]
            normalized[key.strip()] = {"models": models}
    elif isinstance(access_keys, list):
        for key in access_keys:
            if isinstance(key, str) and key.strip():
                normalized[key.strip()] = {"models": None}
    return normalized


def _normalize_cors(cors):
    """Normalize CORS configuration to a consistent dict."""
    if not isinstance(cors, dict):
        return {"origins": [], "allow_credentials": False}
    origins = cors.get("origins", [])
    if isinstance(origins, str):
        origins = [item.strip() for item in origins.split(",") if item.strip()]
    elif isinstance(origins, list):
        origins = [item for item in origins if isinstance(item, str) and item.strip()]
    else:
        origins = []
    allow_credentials = bool(cors.get("allow_credentials", False))
    return {"origins": origins, "allow_credentials": allow_credentials}


def _normalize_rate_limit(rate_limit):
    """Normalize rate limit settings and coerce to ints when possible."""
    if not isinstance(rate_limit, dict):
        return {"requests": None, "window_seconds": None}
    requests = rate_limit.get("requests")
    window_seconds = rate_limit.get("window_seconds")
    try:
        requests = int(requests) if requests is not None else None
    except (TypeError, ValueError):
        requests = None
    try:
        window_seconds = int(window_seconds) if window_seconds is not None else None
    except (TypeError, ValueError):
        window_seconds = None
    return {"requests": requests, "window_seconds": window_seconds}


def _normalize_responses(responses_cfg):
    """Normalize responses configuration."""
    mode = "auto"
    if isinstance(responses_cfg, dict):
        mode = responses_cfg.get("mode", mode)
    elif isinstance(responses_cfg, str):
        mode = responses_cfg
    mode = str(mode).lower()
    if mode not in ("auto", "native", "chat"):
        mode = "auto"
    return {"mode": mode}


def _normalize_logging(logging_cfg):
    """Normalize logging configuration."""
    if not isinstance(logging_cfg, dict):
        logging_cfg = {}
    sample_rate = logging_cfg.get("sample_rate", 1.0)
    try:
        sample_rate = float(sample_rate)
    except (TypeError, ValueError):
        sample_rate = 1.0
    sample_rate = max(0.0, min(1.0, sample_rate))
    include_headers = bool(logging_cfg.get("include_headers", False))
    include_body = bool(logging_cfg.get("include_body", False))
    redact_headers = logging_cfg.get("redact_headers", ["authorization", "x-api-key", "api-key"])
    redact_keys = logging_cfg.get("redact_keys", ["api_key", "apiKey", "access_keys"])
    if isinstance(redact_headers, str):
        redact_headers = [item.strip() for item in redact_headers.split(",") if item.strip()]
    if isinstance(redact_keys, str):
        redact_keys = [item.strip() for item in redact_keys.split(",") if item.strip()]
    if not isinstance(redact_headers, list):
        redact_headers = []
    if not isinstance(redact_keys, list):
        redact_keys = []
    return {
        "sample_rate": sample_rate,
        "include_headers": include_headers,
        "include_body": include_body,
        "redact_headers": redact_headers,
        "redact_keys": redact_keys,
    }


def validate_config(config):
    """Validate config shape and relationships; return list of warnings."""
    errors = []
    providers = config.get("providers", {})
    if not isinstance(providers, dict) or not providers:
        errors.append("providers must be a non-empty object")
        providers = {}

    for name, provider in providers.items():
        if not isinstance(name, str) or not name.strip():
            errors.append("provider name must be a non-empty string")
        if not isinstance(provider, dict):
            errors.append(f"provider '{name}' must be an object")
            continue
        if not provider.get("base_url"):
            errors.append(f"provider '{name}' missing base_url")
        if not provider.get("api_key"):
            errors.append(f"provider '{name}' missing api_key")

    models = config.get("models", [])
    if not isinstance(models, list) or not models:
        errors.append("models must be a non-empty list")
        models = []

    model_ids = {item.get("id") for item in models if isinstance(item, dict)}
    for item in models:
        if not isinstance(item, dict):
            errors.append("model entry must be an object")
            continue
        if not item.get("id"):
            errors.append("model entry missing id")
        if not item.get("provider"):
            errors.append(f"model '{item.get('id', '')}' missing provider")
        if item.get("provider") and item.get("provider") not in providers:
            errors.append(f"model '{item.get('id', '')}' provider not found: {item.get('provider')}")
        if not item.get("model"):
            errors.append(f"model '{item.get('id', '')}' missing model")

    defaults = config.get("defaults", {})
    if isinstance(defaults, dict):
        default_model = defaults.get("model")
        if default_model and default_model not in model_ids:
            errors.append(f"defaults.model not found in models: {default_model}")

    access_keys = _normalize_access_keys(config.get("access_keys", {}))
    for key, value in access_keys.items():
        allowed = value.get("models") if isinstance(value, dict) else None
        if allowed:
            for model_id in allowed:
                if model_id not in model_ids:
                    errors.append(f"access_keys '{key}' references unknown model: {model_id}")

    server = config.get("server", {})
    if isinstance(server, dict) and "port" in server:
        try:
            port = int(server.get("port"))
            if port <= 0 or port > 65535:
                errors.append("server.port must be between 1 and 65535")
        except (TypeError, ValueError):
            errors.append("server.port must be an integer")

    rate_limit = _normalize_rate_limit(config.get("rate_limit", {}))
    if rate_limit["requests"] is not None and rate_limit["requests"] <= 0:
        errors.append("rate_limit.requests must be > 0")
    if rate_limit["window_seconds"] is not None and rate_limit["window_seconds"] <= 0:
        errors.append("rate_limit.window_seconds must be > 0")

    cors = _normalize_cors(config.get("cors", {}))
    if cors["origins"] and not isinstance(cors["origins"], list):
        errors.append("cors.origins must be a list or comma-separated string")

    logging_cfg = _normalize_logging(config.get("logging", {}))
    if not (0.0 <= logging_cfg["sample_rate"] <= 1.0):
        errors.append("logging.sample_rate must be between 0 and 1")

    responses_cfg = _normalize_responses(config.get("responses", {}))
    if responses_cfg["mode"] not in ("auto", "native", "chat"):
        errors.append("responses.mode must be one of: auto, native, chat")

    return errors


def get_access_key_config(key):
    """Return config for a specific access key, or None."""
    if not key:
        return None
    config = load_config()
    access_keys = _normalize_access_keys(config.get("access_keys", {}))
    return access_keys.get(key)


def get_access_keys():
    """Return normalized access keys mapping."""
    config = load_config()
    access_keys = _normalize_access_keys(config.get("access_keys", {}))
    return access_keys


def get_config_errors():
    """Return validation warnings for current config."""
    config = load_config()
    return config.get("errors", [])


def get_server_port():
    """Return server port from config.json, if set."""
    config = load_config()
    server = config.get("server", {})
    port = server.get("port")
    try:
        return int(port)
    except (TypeError, ValueError):
        return None


def get_cors_config():
    """Return normalized CORS config."""
    config = load_config()
    return _normalize_cors(config.get("cors", {}))


def get_rate_limit_config():
    """Return normalized rate limit config."""
    config = load_config()
    return _normalize_rate_limit(config.get("rate_limit", {}))


def get_logging_config():
    """Return normalized logging config."""
    config = load_config()
    return _normalize_logging(config.get("logging", {}))


def get_responses_config():
    """Return normalized responses config."""
    config = load_config()
    return _normalize_responses(config.get("responses", {}))


def resolve_model_config(requested_id):
    """Resolve a public model ID into provider credentials and target model."""
    config = load_config()
    models = config.get("models", [])
    models_by_id = {item["id"]: item for item in models}

    if not requested_id:
        requested_id = config.get("defaults", {}).get("model")

    if requested_id in models_by_id:
        model_entry = models_by_id[requested_id]
        provider = config.get("providers", {}).get(model_entry["provider"], {})
        provider_responses = _normalize_responses(provider.get("responses", {}))
        return {
            "id": model_entry["id"],
            "provider_name": model_entry["provider"],
            "model": model_entry["model"],
            "base_url": provider.get("base_url", ""),
            "api_key": provider.get("api_key", ""),
            "owned_by": model_entry.get("owned_by", "openai"),
            "responses": model_entry.get("responses", _normalize_responses({})),
            "provider_responses": provider_responses,
            "source": "config",
        }

    if models:
        return None

    fallback_model = requested_id or "unknown-model"
    return {
        "id": fallback_model,
        "model": fallback_model,
        "base_url": "",
        "api_key": "",
        "owned_by": "openai",
        "responses": _normalize_responses({}),
        "source": "env",
    }
