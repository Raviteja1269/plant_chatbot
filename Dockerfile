# Use an official PyTorch image as a base
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 appuser

# Create cache directories and set permissions
RUN mkdir -p /app/models/sentence_transformer && \
    mkdir -p /app/models/qwen && \
    mkdir -p /.cache/huggingface && \
    mkdir -p /.cache/torch && \
    mkdir -p /.cache/sentence_transformers

# Set environment variables for cache and model locations
ENV HF_HOME="/.cache/huggingface"
ENV TORCH_HOME="/.cache/torch"
ENV SENTENCE_TRANSFORMERS_HOME="/app/models/sentence_transformer"

# Install Python dependencies
RUN pip install --no-cache-dir \
    pandas \
    torch \
    sentence-transformers \
    transformers \
    numpy \
    faiss-cpu \
    fastapi \
    uvicorn[standard] \
    pydantic \
    python-multipart \
    huggingface_hub \
    accelerate>=0.26.0

# Create a script to download models
COPY <<EOF /app/download_models.py
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Download and save sentence transformer model
print("Downloading sentence transformer model...")
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
model.save("/app/models/sentence_transformer")
print("Sentence transformer model saved successfully!")

# Download and save Qwen model and tokenizer
print("Downloading Qwen model and tokenizer...")
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map=None
)

tokenizer.save_pretrained("/app/models/qwen")
model.save_pretrained("/app/models/qwen")
print("Qwen model and tokenizer saved successfully!")
EOF

# Download models during build
RUN python /app/download_models.py

# Only set TRANSFORMERS_OFFLINE after downloading models
ENV TRANSFORMERS_OFFLINE=1

# Copy the main.py and modify it to use local paths
COPY main.py /app/main.py
RUN sed -i 's|"sentence-transformers/all-mpnet-base-v2"|"/app/models/sentence_transformer"|g' main.py && \
    sed -i 's|"Qwen/Qwen2.5-0.5B-Instruct"|"/app/models/qwen"|g' main.py

# Copy the data file
COPY updated_plant_data_chunks_and_embeddings.csv /app/updated_plant_data_chunks_and_embeddings.csv

# Set proper permissions
RUN chown -R appuser:appuser /app && \
    chown -R appuser:appuser /.cache && \
    chmod -R 755 /app/models

# Switch to non-root user
USER appuser

# Set the environment variable to point to the app directory
ENV PYTHONPATH=/app

# Expose the port
EXPOSE 5000

# Reduce model loading time by setting specific environment variables
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TORCH_NUM_THREADS=1

# Command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--timeout-keep-alive", "300"]