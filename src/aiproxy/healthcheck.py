"""Container healthcheck entrypoint."""
import json
import os
import sys
import urllib.request


def _resolve_port() -> int:
    """Resolve port from config.json or fall back to PORT/default."""
    port = int(os.getenv("PORT", "4000"))
    config_path = os.getenv("CONFIG_PATH", "/app/config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        port = int(config.get("server", {}).get("port", port))
    except Exception:
        return port
    return port


def main() -> int:
    """Return exit code 0 if /healthz is reachable."""
    port = _resolve_port()
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2)
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
