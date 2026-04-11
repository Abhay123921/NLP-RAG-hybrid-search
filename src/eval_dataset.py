import json


def load_eval_dataset(path="data/eval.json"):
    with open(path, "r") as f:
        return json.load(f)