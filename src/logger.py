import json
import os

LOG_PATH = "logs/failures.json"


def log_failure(data, path=LOG_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    try:
        with open(path, "r") as f:
            logs = json.load(f)
    except:
        logs = []

    logs.append(data)

    with open(path, "w") as f:
        json.dump(logs, f, indent=2)


# 🔥 ADD THIS FUNCTION (missing)
def load_failures(path=LOG_PATH):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return []