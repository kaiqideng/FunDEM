# 1) Install dependencies (requires sudo)
sudo apt-get update
sudo apt-get install -y build-essential ninja-build gdb
sudo apt-get install -y gcc-12 g++-12

# 2) CUDA env (pick ONE)
# Option A
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Option B (example)
export CUDA_HOME=/usr/local/cuda-12.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# check
nvidia-smi
nvcc --version
ninja --version
cmake --version

rm -rf build && mkdir -p build

# 3) Debug build (Ninja)
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DDEMSPH_CUDA_ARCH=native \
  -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-12

cmake --build build -j

# 4) Release build (Ninja)
rm -rf build && mkdir -p build

cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DDEMSPH_CUDA_ARCH=native \
  -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-12

cmake --build build -j

# Debug
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# Release
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build