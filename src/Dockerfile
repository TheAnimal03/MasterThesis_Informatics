FROM nvcr.io/nvidia/pytorch:20.11-py3

WORKDIR /workspace/

# Copy source code
COPY . /yolo/

# Install dependencies
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    zip htop screen libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install seaborn thop

# install mish-cuda
RUN cd / && \
    git clone https://github.com/JunnYu/mish-cuda && \
    cd mish-cuda && \
    python setup.py build install

# Install requirements
RUN cd yolo && \
    pip install -r requirements.txt
