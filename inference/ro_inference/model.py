import xgboost as xgb


def load_model(model_bin_path):
    """Load XGBoost model from bin and retunr it."""
    model = xgb.Booster()
    print(f"Loading model from {model_bin_path}")
    model.load_model(model_bin_path)

    return model
