from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def _prepare_metrics(y_true, y_pred):
    """
    Prepares a classification metrics dictionary but does NOT save it.
    Returns a dict ready to be passed to Core.
    """
    metrics = {
        "classification": {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
    }
    return metrics
