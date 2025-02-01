# Use the official Python 3.12.8-slim image as the base image
FROM python:3.12.8-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Create a virtual environment
RUN python3 -m venv /opt/venv

# Activate the virtual environment and install packages from requirements.txt
# Use a single RUN command to reduce the number of layers in the image
RUN . /opt/venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Set the environment variable to activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Command to run when the container starts
CMD ["python", "app.py"]
