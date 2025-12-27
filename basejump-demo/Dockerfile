FROM python:3.11-bullseye

# Copy files to container
COPY . /app/basejump-demo
WORKDIR /app

# Install Requirements
# Install CPU-only pytorch first to avoid installing large files for GPU with CUDA
# This is primarily for docker since your computer likely already has the necessary installations
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu 
RUN pip install basejump-core
RUN pip install -e basejump-demo