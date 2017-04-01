#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
#export PATH=$PATH:/usr/local/cuda/bin
#export PATH=$PATH:/usr/bin/gcc


nvcc -c main.cu \
     #-I /home/huifeng/Cuda-7.0-Samples/samples/common/inc/ \ 

nvcc -o obj main.o
    #-L /home/huifeng/NVIDIA_GPU_Computing_SDK/C/lib -lcutil_x86_64 \
	#-L /usr/local/cuda-7.0/lib64  -lcufft  \
	#-lgsl -lgslcblas -lm  \
    #--ptxas-options=-v 

nvprof ./obj
#./obj

#rm obj
