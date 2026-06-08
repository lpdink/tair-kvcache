import logging
import os

logger = logging.getLogger(__name__)

# Environment variable takes precedence over the default (WARNING).
# Can be further overridden at runtime via set_log_level() from connector config.
_default_level = os.environ.get("KVCM_LOG_LEVEL", "WARNING").upper()
logger.setLevel(getattr(logging, _default_level, logging.WARNING))

logger.propagate = False
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "[KVCM] %(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s",
    "%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
logger.addHandler(handler)


def set_log_level(level: str) -> None:
    """Dynamically adjust the KVCM logger level.

    Called by the connector __init__ after parsing user config,
    so that startup parameters take priority over the environment variable.

    Args:
        level: Log level string, e.g. "DEBUG", "INFO", "WARNING", "ERROR".
               Invalid values are ignored with a warning.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if numeric_level is None:
        logger.warning("Invalid log level: %r, keeping current level", level)
        return
    logger.setLevel(numeric_level)
    logger.info("KVCM log level set to %s", level.upper())
