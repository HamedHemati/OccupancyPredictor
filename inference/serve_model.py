import argparse
import json
import logging
import time
import numpy as np
from flask import Flask, request, jsonify

from ro_inference.model import load_model
from ro_inference.inference_utils import convert_json_to_model_input


# Initialize flask app for model serving
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],  # Logs messages to console
)
logger = logging.getLogger(__name__)


# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--saved_model_path", type=str, default="./output/model.bin")
args = parser.parse_args()

# Load pre-trained model
model = load_model(args.saved_model_path)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get request data (JSON format)
        data = request.get_json(force=True)

        # Make prediction
        input_DMatrix = convert_json_to_model_input(np.array(data["input"]).reshape(1, -1))
        preds = model.predict(input_DMatrix)
        response = {"prediction": preds[0].tolist()}

        # Log prediction
        logger.info(
            json.dumps({"input": data["input"], "prediction": response["prediction"], "timestamp": time.time()})
        )

        time.sleep(0.1)  # TODO: I've added this for testing and benchmarking purposes, remove for production

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error happened during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000, threaded=False)  # TODO: Turn on threaded for production
