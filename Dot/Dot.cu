const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = (N+threadsPerBlock-1)/threadsPerBlock;

__global__ void dot( float *a, float *b, float *c )
{
  __shared__ float cache[threadsPerBlock];

  int i = threadIdx.x + blockIdx.x*threadsPerBlock;
  int j = threadIdx.x;

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
