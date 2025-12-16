# Base image with CUDA 12.1 and Ubuntu 22.04
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# -----------------------------------------------------------------------------
# 1. Install system dependencies (including vim)
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    python3 \
    python3-pip \
    python3-dev \
    software-properties-common \
    lsb-release \
    curl \
    gnupg \
    libz3-dev \
    zlib1g-dev \
    libncurses5-dev \
    libxml2-dev \
    libedit-dev \
    libsqlite3-dev \
    swig \
    vim \
 && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 2. Install Miniconda
# -----------------------------------------------------------------------------
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /root/miniconda3 && \
    rm /tmp/miniconda.sh

# -----------------------------------------------------------------------------
# 3. Install NVM (Node Version Manager)
# -----------------------------------------------------------------------------
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash

# -----------------------------------------------------------------------------
# 4. Install Vim Runtime (e.g. awesome vimrc)
# -----------------------------------------------------------------------------
RUN git clone --depth=1 https://github.com/amix/vimrc.git /root/.vim_runtime && \
    sh /root/.vim_runtime/install_awesome_vimrc.sh

# -----------------------------------------------------------------------------
# 5. Set basic environment variables for non-interactive shells
# -----------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    MLIRENV=/root/mlir_playground \
    LLVM_PROJECTS="clang;mlir" \
    LLVM_ENABLE_PROJECTS="clang;mlir;lld" \
    LLVM_ENABLE_RUNTIMES="libcxx;libcxxabi" \
    LLVM_TARGETS_TO_BUILD="X86;NVPTX" \
    CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda

# -----------------------------------------------------------------------------
# 6. Clone and prepare MLIR Gym repository
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 6. Clone and prepare MLIR Gym repository with optional GitHub token
# -----------------------------------------------------------------------------
ARG GITHUB_TOKEN=""
RUN mkdir -p /root/compiler_gym && \
    cd /root/compiler_gym && \
    if [ -n "$GITHUB_TOKEN" ]; then \
      git clone https://${GITHUB_TOKEN}@github.com/MPSLab-ASU/MLIR-Gym.git mlir_playground; \
    else \
      git clone https://github.com/MPSLab-ASU/MLIR-Gym.git mlir_playground; \
    fi

RUN echo "Step 6"
# -----------------------------------------------------------------------------
# 7. Build and install protobuf (v3.21.12)
# -----------------------------------------------------------------------------
RUN git clone https://github.com/protocolbuffers/protobuf.git /root/protobuf && \
    cd /root/protobuf && \
    git checkout v3.21.12 && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/root/protobuf/install \
             -Dprotobuf_BUILD_TESTS=OFF \
             -DBUILD_SHARED_LIBS=ON && \
    make -j$(nproc) && make install

# -----------------------------------------------------------------------------
# 8. Install Bazel 8.0.1
# -----------------------------------------------------------------------------
RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor -o /etc/apt/trusted.gpg.d/bazel.gpg && \
    echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" > /etc/apt/sources.list.d/bazel.list && \
    apt-get update && \
    apt-get install -y bazel-8.0.1 && \
    ln -s /usr/bin/bazel-8.0.1 /usr/bin/bazel

# -----------------------------------------------------------------------------
# 9. Build and install gRPC (v1.69.0)
# -----------------------------------------------------------------------------
RUN git clone https://github.com/grpc/grpc.git /root/grpc && \
    cd /root/grpc && \
    git checkout v1.69.0 && \
    git submodule update --init --recursive && \
    mkdir -p cmake/build && cd cmake/build && \
    cmake ../.. -DgRPC_INSTALL=ON \
             -DgRPC_BUILD_TESTS=OFF \
             -DCMAKE_INSTALL_PREFIX=/root/grpc/install && \
    make -j$(nproc) && make install

# -----------------------------------------------------------------------------
# 11. Build and install Google Test
# -----------------------------------------------------------------------------
RUN git clone https://github.com/google/googletest.git /root/googletest && \
    cd /root/googletest && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/root/googletest/install && \
    make -j$(nproc) && make install

# -----------------------------------------------------------------------------
# 12. Clone and build LLVM/MLIR (Release 20.x)
# -----------------------------------------------------------------------------
# Enable ccache permanently
ENV CCACHE_DIR=/ccache CCACHE_MAXSIZE=20G
RUN mkdir /ccache && chmod 777 /ccache

# clone repo
RUN git clone --depth=1 --branch=release/20.x https://github.com/llvm/llvm-project.git /root/llvm-project && \
    cd /root/llvm-project && \
    mkdir build

# llvm and mlir dependencies
RUN apt-get update && apt-get install -y \
    ccache
RUN python3 -m pip install \
    numpy \
    pybind11


RUN cd /root/llvm-project/build && \
    cmake -G Ninja ../llvm \
      -DLLVM_ENABLE_PROJECTS="clang;mlir;lld" \
      -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
      -DMLIR_ENABLE_CUDA_RUNNER=ON \
      -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
      -DPython3_EXECUTABLE=$(which python3) \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DLLVM_INSTALL_UTILS=ON \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
      -DCMAKE_INSTALL_PREFIX=/usr/local/llvm-20 \
      -DLLVM_CCACHE_BUILD=ON \
      -DLLVM_BUILD_EXAMPLES=ON \
      -DLLVM_USE_LINKER=gold && \
    ninja && ninja install && \
    ln -s /usr/local/llvm-20 /usr/local/llvm

# -----------------------------------------------------------------------------
# 13. Set additional environment variables (for non-interactive shells)
# -----------------------------------------------------------------------------
ENV MLIR_CUDA_LIBRARIES="/usr/local/cuda/lib64/stubs/libcuda.so" \
    MLIR_CUDA_INCLUDE_DIRS="/usr/local/cuda/include" \
    LLVM_DIR=/usr/local/llvm/lib/cmake/llvm \
    MLIR_DIR=/usr/local/llvm/lib/cmake/mlir \
    Protobuf_DIR=/root/protobuf/install/lib/cmake/protobuf \
    gRPC_DIR=/root/grpc/install/lib/cmake/grpc \
    GTEST_DIR=/root/googletest/install/lib/cmake/GTest \
    PATH="/usr/local/bin:/root/miniconda3/bin:/root/protobuf/install/bin:/root/grpc/install/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/lib:/root/protobuf/install/lib:/root/grpc/install/lib:${LD_LIBRARY_PATH}" \
    CMAKE_PREFIX_PATH="/usr/local:/root/protobuf/install:/root/grpc/install:${CMAKE_PREFIX_PATH}"

# -----------------------------------------------------------------------------
# 14. Install .bashrc using proper heredoc syntax
# -----------------------------------------------------------------------------
COPY bashrc /root/.bashrc

# -----------------------------------------------------------------------------
# 14-1. Install abseil
# -----------------------------------------------------------------------------
RUN git clone https://github.com/abseil/abseil-cpp.git && \
    cd abseil-cpp && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/root/abseil-cpp/install && \
    make -j32 && \
    make install

# -----------------------------------------------------------------------------
# 14-2. Install utf8-range
# -----------------------------------------------------------------------------
RUN git clone https://github.com/protocolbuffers/utf8_range.git && \
    cd utf8_range && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/root/utf8_range/install && \
    make -j32 && \
    make install

# -----------------------------------------------------------------------------
# 15. Prepare backend tests (assumes /root/backend is available in your context)
# -----------------------------------------------------------------------------
RUN cd /root/compiler_gym/mlir_playground/backend/tests/ && \
    mkdir -p build && \
    cd build && \
    cmake .. -DMLIR_DIR=$MLIR_DIR && \
    make -j32
