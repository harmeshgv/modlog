from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def _prepare_metrics(y_true, y_pred):
    """
    Prepares a regression metrics dictionary but does NOT save it.
    Returns a dict ready to be passed to Core.
    """
    metrics = {
        "regression": {
            "MSE": mean_squared_error(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred),
        }
    }
    return metrics
