#include "../FFT_1d_GPU.h"
#include "kernel_FDK.cu"

void Reconstruction_FDK(float *h_volume, float *h_proj_data)
{   
    cudaSetDevice(Default_GPU);
    
    size_t size_volume = sizeof(float)*M*N*ZETA;
    
    float *h_proj_data_filtered;
    checkCuda( cudaHostAlloc((void**)&h_proj_data_filtered, sizeof(float)*R*Z_prj*N_source*Nviews, cudaHostAllocDefault) ); // host pinned memory   
    
	float *d_proj_single = NULL;
	cudaMalloc((void**)&d_proj_single, sizeof(float)*R*Z_prj*N_source);
    
	float *d_volume = NULL;
	cudaMalloc((void**)&d_volume, size_volume);  
    cudaMemset(d_volume,0,size_volume);
    
    float *d_volume_times = NULL;
	cudaMalloc((void**)&d_volume_times, size_volume);    
    
    float *d_volume_update = NULL;
	cudaMalloc((void**)&d_volume_update, size_volume);        
    
	// setup execution parameters for projection / correction  
    dim3  dimGrid(Z_prj*2, N_source, Nviews);
    dim3  dimBlock(R/2);
    
	//setup execution parameters for backprojection (reconstruction volume) 
	dim3  dimGrid_backprj(4*N,ZETA,1); 
    dim3  dimBlock_backprj(M/4);
    
    dim3  dimGrid_V(2,N,ZETA); 
    dim3  dimBlock_V(M/2);
        
    cout<<"Doing the ramp filtering ..."<<endl;
    
    kernel_FDK_projection_adjust<<<dimGrid,dimBlock >>>(h_proj_data, h_proj_data_filtered);
    cudaDeviceSynchronize();
    
    RampFiltering_GPU(h_proj_data_filtered,h_proj_data_filtered);
    
//     SaveDeviceDataToFile(h_proj_data_filtered,R*Z_prj*N_source*Nviews,"../FilteredProjection.dat");
    
    cout<<"Doing the FDK reconstruction ..."<<endl;
    
	for (int j=0; j<Nviews; j=j+1)
	{
        
    	float t_theta = (float)(j*us_rate + initialAngle);
        t_theta = (t_theta + shiftAngle) *PI/180.0f;
        t_theta = t_theta;
                        
        float sin_theta = sin(t_theta);
        float cos_theta = cos(t_theta);

        cudaMemcpy(d_proj_single,h_proj_data_filtered+R*Z_prj*N_source*j,sizeof(float)*R*Z_prj*N_source,cudaMemcpyDefault);
        cudaMemset(d_volume_times,0,size_volume);
        cudaMemset(d_volume_update,0,size_volume);
        
        kernel_FDK_reconstruction<<<dimGrid_backprj,dimBlock_backprj>>>(d_volume_update,d_volume_times,d_proj_single,sin_theta,cos_theta);          

        kernel_volume_divide<<<dimGrid_V,dimBlock_V>>>(d_volume,d_volume_update, d_volume_times); 
        
        cudaStreamSynchronize(0);            
                                
        if (j % 20 == 0)
           printf(" - have done %d projections... \n", j);					
        	
	}    
    
	cudaMemcpy(h_volume,d_volume,size_volume,cudaMemcpyDefault);
    
    cout<<"OK!"<<endl;
    
    cudaFreeHost(h_proj_data_filtered);
    cudaFree(d_proj_single);
    cudaFree(d_volume);
    cudaFree(d_volume_times);
    cudaFree(d_volume_update);
    
    
}