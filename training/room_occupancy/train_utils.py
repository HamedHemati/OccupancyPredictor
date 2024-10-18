from sklearn.metrics import accuracy_score, roc_auc_score


def get_default_search_space():
    """Return the default search space for Random Search."""
    search_space = {
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "n_estimators": [50, 100, 150, 200],
        "gamma": [0, 0.1, 0.2, 0.3],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [0.5, 1.0, 1.5, 2.0],
        "reg_alpha": [0, 0.1, 0.5, 1.0],
    }

    return search_space


def calculate_metrics(y_test, y_logits, y_pred):
    """Calculate metrics : accuracy and roc-auc."""
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_logits)

    print(f"accuracy: {accuracy * 100:.2f}%")
    print(f"roc-auc: {roc_auc:.2f}")

    # Store all in a dict
    metrics = {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
    }

    return metrics
