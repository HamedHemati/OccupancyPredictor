import argparse
import requests
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# API endpoint and headers (application/json)
url = "http://127.0.0.1:9000/predict"
headers = {"Content-Type": "application/json"}


def generate_random_input():
    """Generate random input data for the API"""
    return [
        random.uniform(10, 20),
        random.uniform(31, 35),
        random.uniform(200, 400),
        random.uniform(300, 500),
        random.uniform(0.001, 0.002),
    ]


def send_request():
    """Send a POST request to the API and return the result"""
    try:
        # Generate random input data
        payload = {"input": generate_random_input()}
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=5)
        if response.status_code == 200:
            return f"Success: {response.json()}"
        else:
            return f"Failed with status code: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"


def run_concurrent_requests(total_requests):
    """Run all requests concurrently using ThreadPoolExecutor"""
    start_time = time.time()

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(send_request) for _ in range(total_requests)]

        for future in as_completed(futures):
            print(future.result())

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Completed {total_requests} requests in {total_time:.2f} seconds")


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description="Generate random requests to the API")
    parser.add_argument("--n_requests", type=int, default=10)
    args = parser.parse_args()

    print(f"Sending {args.n_requests} random requests to the API...")
    run_concurrent_requests(args.n_requests)
