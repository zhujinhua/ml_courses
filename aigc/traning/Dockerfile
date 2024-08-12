# Use an official PyTorch base image
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set environment variables
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
RUN pip3 install vllm
RUN pip3 install langchain
RUN pip3 install langchain_openai


# Clone and install Qwen2 model (if necessary)
RUN git clone https://www.modelscope.cn/qwen/qwen2-7b-instruct.git /qwen2-7b-instruct && \
    cd /qwen2-7b-instruct 

# Set the working directory
WORKDIR /app

# Copy your app code to the container
COPY . /app

# Expose the port that the vLLM server will use
EXPOSE 8088

# Command to start the vLLM server
CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", "--model", "qwen2-7b-instruct", "--port", "8088"]
