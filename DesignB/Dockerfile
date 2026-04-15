# DesignB – Custom CUDA Marching Cubes Pipeline
# Base: CUDA 12.0 devel (matches host nvcc 12.0) + Ubuntu 22.04
FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONUNBUFFERED=1

# ── System packages ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    git \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    g++-12 \
    gcc-12 \
    && rm -rf /var/lib/apt/lists/*

# ── Set default compiler and Python to 12 ──────────────────────
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 100

# ── pip ──────────────────────────────────────────────────────────
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && pip install --upgrade pip

# ── PyTorch (CUDA 12.6 build) ────────────────────────────────────
RUN pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu126 \
    && pip install \
    numpy \
    scikit-image \
    trimesh \
    matplotlib \
    pandas \
    pyyaml

WORKDIR /workspace

# ── Copy pre-built CUDA marching-cubes extension ────────────────
# The .so was compiled on the host with Python 3.12 + torch cu126.
# It is ABI-compatible with the torch/cu126 installed above.
# The entrypoint copies it to the mounted project at runtime.
COPY external/marching_cubes_cuda_ext/*.so /opt/mc_ext/
RUN echo "CUDA ext available: $(ls /opt/mc_ext/)"

# ── Copy install helper ──────────────────────────────────────────
# Tiny entrypoint that copies the pre-built .so into the mounted
# project tree (external/marching_cubes_cuda_ext/) if not present.
COPY DesignB/docker_entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
