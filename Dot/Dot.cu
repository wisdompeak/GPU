const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = (N+threadsPerBlock-1)/threadsPerBlock;

__global__ void dot( float *a, float *b, float *c )
{
  __shared__ float cache[threadsPerBlock]; // shared memory只给一个block用
  int i = threadIdx.x + blockIdx.x * blockDim.x;  //用于全局变量的寻址
  int j = threadIdx.x;  // 用于本block内的shared memory寻址
  
  cache[j] += a[i]*b[i];

  __syncthreads();

  int k=threadsPerBlock/2;
  while (k>0)
  {
    if (j<k) cache[j]+=cache[j+k];
    __syncthreads();
    k = k/2;
  }


  if (threadIdx.x==0)
    c[blockIdx.x]=cache[0];
}

/***可以想象成为
for (blockIdx.x = 1; blockIdx.x<M; blockIdx.x++)
{
   __shared__ float cache[threadsPerBlock];
   for (int threadIdx.x = 1; threadIdx.x<N; threadIdx.x++)
   {
       cache[threadIdx.x] = a[threadIdx.x + blockIdx.x * blockDim.x] * b[threadIdx.x + blockIdx.x * blockDim.x];
   }
   
   int k=threadsPerBlock/2;
   while (k>0)
   {
       for (int threadIdx.x = 1; threadIdx.x<N; threadIdx.x++)
       {
           if (threadIdx.x<k)  cache[threadIdx.x]+=cache[threadIdx.x+k];
       }
       k = k/2;
   }
   
   c[blockIdx.x]=cache[0];
}

/****
