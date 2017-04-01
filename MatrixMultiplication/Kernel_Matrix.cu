__global__ void kernel_MatMul_GPU(Matrix A, Matrix B, Matrix C, clock_t* time)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x ;
    int i = blockIdx.y;
    
    int tid = threadIdx.x;
    int bid = blockIdx.x + blockIdx.y * gridDim.x;
 
    //只在 thread 0（即 threadIdx.x = 0 的时候）进行记录，每个 block 都会记录开始时间及结束时间
    if (tid == 0) time[bid] = clock();

    float sum = 0;
    float y=0;
    float c=0;
    for (int k=0; k<A.width; k++)
    {
        y = A.elements[i*A.width+k]*B.elements[k*B.width+j] - c;
        c = sum + y -sum -y;
        sum = sum + y;
        // Kahan’s Summation Formula:  http://blog.csdn.net/sunmc1204953974/article/details/51107850
    }
    
    //__syncthreads();
    
    C.elements[i*C.width+j] = sum;
                  
    //计算时间的动作，只在 thread 0（即 threadIdx.x = 0 的时候）进行，每个 block 都会记录开始时间及结束时间
    if (tid == 0) time[bid + BLOCK_NUM] = clock();

}