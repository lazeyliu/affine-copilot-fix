"""Environment-driven settings for AIProxy."""
from dataclasses import dataclass
from functools import lru_cache
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    log_level: str
    app_version: str
    create_log: bool
    upstream_timeout: float
    upstream_max_retries: int
    max_body_mb: float
    strict_config: bool
    log_dir: str

    @property
    def max_content_length(self) -> int:
        return int(self.max_body_mb * 1024 * 1024)


@lru_cache
def get_settings() -> Settings:
    """Load settings from environment variables."""
    return Settings(
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        app_version=os.getenv("APP_VERSION", "0.2.0"),
        create_log=os.getenv("CREATE_LOG", "False").lower() in ("1", "true", "yes", "on"),
        upstream_timeout=float(os.getenv("UPSTREAM_TIMEOUT", "60")),
        upstream_max_retries=int(os.getenv("UPSTREAM_MAX_RETRIES", "2")),
        max_body_mb=float(os.getenv("MAX_BODY_MB", "4")),
        strict_config=os.getenv("STRICT_CONFIG", "").lower() in ("1", "true", "yes", "on"),
        log_dir=os.getenv(
            "LOG_DIR",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "logs")),
        ),
    )
