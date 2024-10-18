import xgboost as xgb


def get_model(random_seed=0):
    """Return a XGBoost model."""
    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        seed=random_seed,
    )

    return xgb_model
