export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

export PATH=/usr/local/cuda/bin:$PATH 

nvcc main_proj.cu -o 3D_Siddon_Proj -I /home/huifeng/NVIDIA_GPU_Computing_SDK/C/common/inc -lcutil_x86_64 -L /home/huifeng/NVIDIA_GPU_Computing_SDK/C/lib     --ptxas-options=-v 

./3D_Siddon_Proj
