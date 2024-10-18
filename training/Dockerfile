# Base image with Python
FROM python:3.10-slim

# Working DIR
WORKDIR /src

# Copy and install requirements for pip
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the src files
COPY . .

# Run model training
CMD ["python", "train.py"]
