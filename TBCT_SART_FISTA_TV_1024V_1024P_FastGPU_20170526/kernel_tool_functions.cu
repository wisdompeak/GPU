__global__ void getSum_kernel(float *g_idata, float *result, long long n)
{
	//load shared_mem
    unsigned int THREAD_NUM = blockDim.x;    
	extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    
    sdata[tid] = 0;    
	for (long long i=tid; i<n; i+=THREAD_NUM)    
        sdata[tid] += g_idata[i];

    __syncthreads();
    
    unsigned int k = THREAD_NUM/2;
   
    while (k>0)
    {
        if (tid<k)
            sdata[tid]+=sdata[tid+k];
        k = k/2;
        __syncthreads();       
    }
        
	if (tid == 0) result[0]  = sdata[0];

//     __syncthreads();      
    
}




__global__ void L2_norm_kernel(float *d_1, float *d_2)
{
	
	long long i = blockIdx.z* blockDim.x*gridDim.x*gridDim.y + blockIdx.y* blockDim.x*gridDim.x + blockIdx.x*blockDim.x + threadIdx.x;
	d_1[i] = (d_1[i]-d_2[i])*(d_1[i]-d_2[i]);
}



__global__ void TV_norm_kernel(float *d_TV, float *d_volume)       
{
    
	unsigned int i = blockIdx.y* blockDim.x*gridDim.x + blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y* blockDim.x*gridDim.x + blockIdx.x*blockDim.x + (threadIdx.x+1);
    unsigned int k = blockIdx.y* blockDim.x*gridDim.x + (blockIdx.x+1)*blockDim.x + threadIdx.x;
    unsigned int l = (blockIdx.y+1)* blockDim.x*gridDim.x + blockIdx.x*blockDim.x + threadIdx.x;
    
    if ((threadIdx.x<M-1)&&(blockIdx.x<N-1)&&(blockIdx.y<ZETA-1))
        d_TV[i]=sqrt( (d_volume[i]-d_volume[j])*(d_volume[i]-d_volume[j])+(d_volume[i]-d_volume[k])*(d_volume[i]-d_volume[k])+(d_volume[i]-d_volume[l])*(d_volume[i]-d_volume[l]) );
    else if ((threadIdx.x==M-1)&&(blockIdx.x<N-1)&&(blockIdx.y<ZETA-1))
        d_TV[i]=sqrt( (d_volume[i]-d_volume[k])*(d_volume[i]-d_volume[k])+(d_volume[i]-d_volume[l])*(d_volume[i]-d_volume[l]) );
    else if ((threadIdx.x<M-1)&&(blockIdx.x==N-1)&&(blockIdx.y<ZETA-1))
        d_TV[i]=sqrt( (d_volume[i]-d_volume[j])*(d_volume[i]-d_volume[j])+(d_volume[i]-d_volume[l])*(d_volume[i]-d_volume[l]) );    
    else if ((threadIdx.x<M-1)&&(blockIdx.x<N-1)&&(blockIdx.y==ZETA-1))
        d_TV[i]=sqrt( (d_volume[i]-d_volume[j])*(d_volume[i]-d_volume[j])+(d_volume[i]-d_volume[k])*(d_volume[i]-d_volume[k]) );    
    else if ((threadIdx.x==M-1)&&(blockIdx.x==N-1)&&(blockIdx.y<ZETA-1))
        d_TV[i]=abs( d_volume[i]-d_volume[l]);
    else if ((threadIdx.x==M-1)&&(blockIdx.x<N-1)&&(blockIdx.y==ZETA-1))
        d_TV[i]=abs( d_volume[i]-d_volume[k]);
    else if ((threadIdx.x<M-1)&&(blockIdx.x==N-1)&&(blockIdx.y==ZETA-1))
        d_TV[i]=abs( d_volume[i]-d_volume[j]);
         
}



__global__ void substract_3d_kernel(float *d_1, float *d_2, float *d_result)
{
	
	long i = blockIdx.y* blockDim.x*gridDim.x + blockIdx.x*blockDim.x + threadIdx.x;
	d_result[i] = d_1[i] -d_2[i];
}