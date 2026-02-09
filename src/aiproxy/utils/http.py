"""HTTP helpers and error responses."""
from flask import jsonify, request


def get_client_ip() -> str:
    """Resolve client IP with basic X-Forwarded-For support."""
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.remote_addr or "unknown"


def error_response(message: str, status: int = 400, error_type: str = "invalid_request_error", param=None, code=None):
    """Return OpenAI-style error payload."""
    payload = {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }
    return jsonify(payload), status
