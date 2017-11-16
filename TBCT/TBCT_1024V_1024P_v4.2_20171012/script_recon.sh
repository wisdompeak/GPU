export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-5.5/lib64
export PATH=/usr/local/cuda-5.5/bin:$PATH 

#nvcc main_recon.cu -o 3D_Iter_Recon -I /home/huifeng/NVIDIA_GPU_Computing_SDK/C/common/inc -lcutil_x86_64 -L /home/huifeng/NVIDIA_GPU_Computing_SDK/C/lib     --ptxas-options=-v 

nvcc -c main_recon.cu -Xcompiler -fopenmp\
    -arch=sm_35\
    -use_fast_math\
    #--ptxas-options=-v\
    #-I /usr/local/cuda-7.0/samples/common/inc \
    #-I /usr/local/cuda-7.0/include \
    #-I /home/huifeng/NVIDIA_GPU_Computing_SDK/C/common/inc \

nvcc -o obj main_recon.o \
    -lgomp\
    -lcufft\
    #-L /usr/local/cuda-7.0/lib64 \
	#-lcutil_x86_64 \    
    #-L /home/huifeng/NVIDIA_GPU_Computing_SDK/C/lib \

#nvprof ./obj
./obj
