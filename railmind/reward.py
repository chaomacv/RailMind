import json

from .schema import labels, parse_answer


def category_reward(completion, solution):
    """Paper set reward: exact=1, strict superset=0.5, otherwise=0."""
    try:
        predicted = set(parse_answer(completion)["anomaly_classes"])
        gt = set(labels(json.loads(solution)["anomaly_classes"]))
    except (ValueError, KeyError, TypeError):
        return 0.0
    if predicted == gt:
        return 1.0
    return 0.5 if gt < predicted else 0.0
