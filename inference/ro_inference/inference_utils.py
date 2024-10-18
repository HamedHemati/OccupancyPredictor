import xgboost as xgb


def convert_json_to_model_input(input_data):
    """Convert JSON input to DMatrix for model prediction"""
    feature_names = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
    dmatrix = xgb.DMatrix(input_data, feature_names=feature_names)

    return dmatrix
