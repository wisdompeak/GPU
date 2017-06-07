void host_forward(int i, float* d_recon, float * d_proj)
{
    dim3  dimblock(256,1);
    dim3  dimgrid(1024,1024/2,1);    
           
    cudaSetDevice(i);        
    kernel_forward_projection<<<dimgrid, dimblock>>>(d_recon, d_proj);    
}

void host_backprj(int i, float* d_recon, float * d_proj)
{
    dim3  dimblock(256,1);
    dim3  dimgrid(1024,1024/2,1);    
           
    cudaSetDevice(i);        
    kernel_back_projection<<<dimgrid, dimblock>>>(d_recon, d_proj);    
}

void AddProjection(float* d_proj_addr[])
{
    size_t single_gpu_chunk_size = sizeof(float)*Num_elem/2;
    
    dim3  dimblock(256,1);
    dim3  dimgrid(1024,1024/2,1);
    
    cudaSetDevice(0);                   
    for (int i=1; i<2; i++)
    {
        cudaDeviceEnablePeerAccess(i,0);
        kernel_add<<<dimgrid, dimblock>>>(d_proj_addr[0],d_proj_addr[i]);
    }
    
    for (int i=1; i<2; i++)
    {
        cudaMemcpyAsync(d_proj_addr[i],d_proj_addr[0],single_gpu_chunk_size,cudaMemcpyDefault);
    }
    
    for (int i=1; i<2; i++)
    {
        cudaSetDevice(i);
        cudaStreamSynchronize(0);
    }            
}