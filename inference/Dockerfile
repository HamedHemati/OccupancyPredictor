# Base image with Python
FROM python:3.10-slim

# Working DIR
WORKDIR /app

# Copy and install requirements for pip 
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app files into the container
COPY . .

# Expose 9000 (Flask app runs on port 9000)
EXPOSE 9000

# Run serve_model
CMD ["python", "serve_model.py", "--saved_model_path", "/app/pretrained_models/best_model.bin"] 
