from src.logger import load_failures

def analyze_failures():
    logs = load_failures()

    if not logs:
        return {"adjust_threshold": 0.75}

    low_conf = sum(1 for l in logs if l["confidence"] < 0.5)
    low_quality = sum(1 for l in logs if l["quality"] < 0.6)

    total = len(logs)

    # 🔥 adjust threshold dynamically
    if total == 0:
        return {"adjust_threshold": 0.75}

    error_rate = (low_conf + low_quality) / total

    if error_rate > 0.3:
        return {"adjust_threshold": 0.85}
    elif error_rate < 0.1:
        return {"adjust_threshold": 0.7}

    return {"adjust_threshold": 0.75}