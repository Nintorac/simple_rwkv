FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

# Update, install
RUN apt-get update && \
    apt-get install -y build-essential ninja-build git wget

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda create -y --name py311 python=3.11 && \
    /opt/conda/bin/conda clean -ya

ENV PATH /home/user/.local/bin:/opt/conda/envs/py311/bin:$PATH


RUN pip install --upgrade pip setuptools wheel

# Create user instead of using root
ENV USER='user'
RUN groupadd -r user && useradd -r -g $USER $USER
RUN mkdir -p /home/$USER/app
RUN chown -R $USER:$USER /home/$USER
USER $USER

# Define workdir
WORKDIR /home/$USER/app
# install torch, there are issues with poetry otherwise
RUN pip install torch==2.0.1
# Install project
COPY pyproject.toml .


# Copy rest
COPY simple_rwkv simple_rwkv
COPY simple_rwkv simple_rwkv
COPY obsidian_serve.py .
copy logging.conf .

RUN pip install .

# Download model
# RUN python -m simple_rwkv.get_models

# Publish port
EXPOSE 50051:50051
# Enjoy
