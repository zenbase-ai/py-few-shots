from datetime import datetime, timezone


def utcnow() -> float:
    return datetime.now(timezone.utc).timestamp()
