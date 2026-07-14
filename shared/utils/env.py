import os


def require_env(name: str) -> str:
    """Fetch a required env var, or raise a clear error pointing at .env
    instead of a bare KeyError.
    """
    value = os.environ.get(name)
    if not value:
        raise ValueError(
            f"Missing required environment variable '{name}'. "
            f"Check that it's set in your .env file (see .env.example)."
        )
    return value
