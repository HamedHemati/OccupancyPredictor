# Occupancy Predictor

## Repository Structure

    ├── training                      # Training sub-directory
        ├── room_occupancy            # Training modules as a package
            ├── ...                   
        ├── train.py                  # Training script
        ├── Dockerfile                # Dockerfile for training
        ├── ...                   
    
    ├── inference                     # Inference sub-directory
        ├── ro_inference              # Inference modules as a package
            ├── ...                   
        ├── servce_model.py           # Model serving script
        ├── Dockerfile                # Dockerfile for inference
        ├── ...                   

    ├── tests                         # Tests
      ├── ...                       
    ├── kubernets                     # Kubernetes deployment files
      ├── ...                       
    ├── scripts                       # Misc. scripts for benchmarking, etc.
      ├── ...                       
    ├── data                          # Data directory
      ├── ...                       

## Deployment (Kuberentes)

To deploy the services to Kubernetes, use the following commands:

```bash
# Deploy the inference service
kubectl apply -f kubernetes/deploy_inference_api.yaml

# Deploy the logging and monitoring service
kubectl apply -f kubernetes/deploy_logging.yaml
```

## Training

To train the model in a container, use the following command:

```bash
docker run -v ./my_data:/src/my_data -v ./out:/src/out \
--env TRAINING_DATA="/src/my_data/datatraining.txt" \
--env TEST_DATA="/src/original/datatest.txt" \
--env SAVE_PATH="/src/out" occupancy_training:latest
```

You need to provide the training data and the test data as environment variables. The trained model will be saved in the directory specified by the `SAVE_PATH` environment variable.

## Inference

For serving the model in a containerized environment, you can use the following command:

```bash
docker run -p 9000:9000 neuperc/occupancy_inference:latest
```

The Flask server will be running on port 9000, therefore you can send requests to `http://localhost:9000/predict`.

Run the command below in a terminal to check if the API is running:

```bash
 curl -X POST http://localhost:9000/predict -H "Content-Type: application/json" \
-d '{
    "input": [21.0, 25.0, 200.0, 400.0, 0.004]
}'
```

## Running Tests

To run the tests, use `pytest`. Make sure the dependencies are installed. Run the following command from the root directory of the repository:

```bash
PYTHONPATH=. pytest
```
