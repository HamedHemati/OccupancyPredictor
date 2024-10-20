import argparse
import requests
import json
import time
from training.room_occupancy.data_utils import load_dataset

# API endpoint and headers (application/json)
url = "http://127.0.0.1:9000/predict"
headers = {"Content-Type": "application/json"}


def send_request(input_data):
    """Send a POST request to the API and return the result"""
    try:
        payload = {"input": input_data}
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=5)
        if response.status_code == 200:
            return f"Success: {response.json()}"
        else:
            return f"Failed with status code: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"


def send_requests(args):
    """Run all requests concurrently using ThreadPoolExecutor"""
    x_df, _ = load_dataset(args.file_path)
    # Shuffle x_df rows
    x_df = x_df.sample(n=len(x_df)).reset_index(drop=True)

    # Iterate over each row in the dataset and send a request
    for i in range(len(x_df)):
        start_time = time.time()
        input_data = x_df.iloc[i].values.tolist()
        print(send_request(input_data))
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Completed request in {total_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate requests to the API using a file of input data")
    parser.add_argument("--file_path", type=str)
    args = parser.parse_args()

    send_requests(args)
