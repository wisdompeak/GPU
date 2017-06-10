export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-7.0/lib64
export PATH=/usr/local/cuda-7.0/bin:$PATH 
#nvcc main_recon.cu -o 3D_Iter_Recon -I /home/huifeng/NVIDIA_GPU_Computing_SDK/C/common/inc -lcutil_x86_64 -L /home/huifeng/NVIDIA_GPU_Computing_SDK/C/lib     --ptxas-options=-v 
#nvcc main_recon.cu -o 3D_Iter_Recon -I /home/huifeng/NVIDIA_GPU_Computing_SDK/C/common/inc -lcutil_x86_64 -I /usr/local/cuda-7.0/samples/common/inc -I /usr/local/cuda-7.0/include  -L /home/huifeng/NVIDIA_GPU_Computing_SDK/C/lib   -L /usr/local/cuda-7.0/lib64 --ptxas-options=-v 

nvcc -c main_recon.cu \
    #--ptxas-options=-v -arch=sm_35
    #-I /usr/local/cuda-7.0/samples/common/inc \
    #-I /usr/local/cuda-7.0/include \
    #-I /home/huifeng/NVIDIA_GPU_Computing_SDK/C/common/inc \

nvcc -o obj main_recon.o \
    #-L /usr/local/cuda-7.0/lib64 \
	#-lcutil_x86_64 \    
    #-L /home/huifeng/NVIDIA_GPU_Computing_SDK/C/lib \

#nvprof ./obj
./obj