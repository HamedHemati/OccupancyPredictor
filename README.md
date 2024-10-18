# Occupancy Predictor

TBA ...

## Repository Structure

    ├── training                      # Training sub-directory
        ├── room_occupancy
            ├── ...                   # Training modules as a package
        ├── train.py                  # Training script
        ├── Dockerfile                # Dockerfile for training
        ├── ...                   
    
    ├── inference                     # Inference sub-directory
        ├── ro_inference
            ├── ...                   # Inference modules as a package
        ├── servce_model.py           # Model serving script
        ├── Dockerfile                # Dockerfile for inference
        ├── ...                   

    ├── tests                         # Tests
      ├── ...                       
    ├── kubernets                     # Kubernetes deployment files
      ├── ...                       
    ├── scripts                       # Miscellaneous scripts for benchmarking, etc.
      ├── ...                       
    ├── data                          # Data directory
      ├── ...                       

## Testing

```bash
PYTHONPATH=. pytest
```
