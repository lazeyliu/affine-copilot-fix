"""Application factory and entrypoint."""
import os
import time

from flask import Flask

from .config import get_config_errors, get_server_port
from ..utils.logging import log_event, setup_logging
from ..api.middleware import register_middlewares
from .ratelimit import RateLimiter
from ..api.handlers import register_routes
from .settings import get_settings


def create_app() -> Flask:
    """Create and configure the Flask application."""
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_dir)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    template_root = os.path.join(base_dir, "templates")
    static_root = os.path.join(base_dir, "static")
    app = Flask(__name__, template_folder=template_root, static_folder=static_root)
    app.config["MAX_CONTENT_LENGTH"] = settings.max_content_length
    app.config["APP_STARTED_AT"] = time.time()
    app.config["SETTINGS"] = settings

    register_middlewares(app, settings, RateLimiter())
    register_routes(app, settings)
    return app


def run() -> None:
    """Run the Flask development server."""
    app = create_app()
    settings = app.config["SETTINGS"]

    if settings.strict_config:
        config_errors = get_config_errors()
        if config_errors:
            for err in config_errors:
                log_event(40, "config_error", error=err)
            raise SystemExit("Strict config enabled; fix config.json errors.")

    port = get_server_port() or int(os.getenv("PORT", 4000))
    debug = os.getenv("FLASK_DEBUG", "").lower() in ("1", "true", "yes", "on")
    app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    run()
