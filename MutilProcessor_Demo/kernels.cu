
__global__ void kernel_forward_projection(float *d_a, float *d_b)
{
	int idx = blockDim.x * gridDim.x * blockIdx.y +  blockDim.x * blockIdx.x + threadIdx.x; 
    d_b[idx]=d_a[idx]+0.6f;
}

__global__ void kernel_back_projection(float *d_a, float *d_b)
{
	int idx = blockDim.x * gridDim.x * blockIdx.y +  blockDim.x * blockIdx.x + threadIdx.x; 
    d_a[idx]=d_b[idx]/2.0f;
}

__global__ void kernel_add(float *d_a, float *d_b)
{
	int idx = blockDim.x * gridDim.x * blockIdx.y +  blockDim.x * blockIdx.x + threadIdx.x; 
    d_a[idx]=d_a[idx]+d_b[idx];
}