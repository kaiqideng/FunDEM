# RUN YOUR OWN FUNDEM

# 1) compilation env
# update
sudo apt-get update

# basic build tools
sudo apt-get install -y build-essential cmake ninja-build git

# (optional but recommended) pkg-config
sudo apt-get install -y pkg-config

# check
cmake --version
ninja --version
g++ --version

# 2) CUDA env
# check
which nvcc
nvcc --version

# CUDA PATH / LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# check
nvcc --version

echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 3）check nvidia
# check
nvidia-smi

# 4）release or debug (Ninja, Default DCMAKE_CUDA_ARCHITECTURES=75)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j

# CUDA_ARCHITECTURES (V100: 70, T4: 75, A100: 80, RTX 30xx: 86, L4: 89, H100: 90)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build -j

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="75;86"
cmake --build build -j

# 5) run
./build/...