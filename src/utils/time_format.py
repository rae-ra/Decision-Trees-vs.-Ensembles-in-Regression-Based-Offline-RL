def format_duration(seconds: float) -> str:
    """
    Convert seconds into a human-readable duration string.
    Examples:
        3661 -> "1h 1m 1s"
        61.5 -> "1m 1.5s"
        5.0 -> "5.0s"
    """
    parts = []
    h = int(seconds // 3600)
    if h > 0:
        parts.append(f"{h}h")
    seconds %= 3600
    m = int(seconds // 60)
    if m > 0:
        parts.append(f"{m}m")
    s = round(seconds % 60, 2)
    parts.append(f"{s}s")
    return " ".join(parts)
