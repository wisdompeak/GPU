#include "FDK/FFT_1d_GPU.h"
#include "FDK/kernel_FDK.cu"

void Reconstruction_FDK(float *h_volume, float *h_proj_data)
{   
    cudaSetDevice(Default_GPU);
    
    float *h_proj_data_filtered;
    checkCuda( cudaHostAlloc((void**)&h_proj_data_filtered, sizeof(float)*R*Z_prj*Nviews, cudaHostAllocDefault) ); // host pinned memory   
    
	float *d_proj_single = NULL;
	cudaMalloc((void**)&d_proj_single, sizeof(float)*R*Z_prj);
    
	// setup execution parameters for projection / correction  
    dim3  dimGrid(2, Z_prj, Nviews);
    dim3  dimBlock(R/2);
    
	//setup execution parameters for backprojection (reconstruction volume) 
	dim3  dimGrid_backprj(4,N,ZETA); 
    dim3  dimBlock_backprj(M/4);
        
    cout<<"Doing the ramp filering ..."<<endl;
    
    kernel_FDK_projection_adjust<<<dimGrid,dimBlock >>>(h_proj_data, h_proj_data_filtered);
    
    RampFiltering_GPU(h_proj_data_filtered,h_proj_data_filtered);
    
//     SaveDeviceDataToFile(h_proj_data_filtered,R*Z_prj*Nviews,"../FilteredProjection.dat");
    
    cout<<"Doing the FDK reconstruction ..."<<endl;
    
	for (int j=0; j<Nviews; j=j+1)
	{
        
    	float t_theta = (float)(j*us_rate + initialAngle);
        t_theta = (t_theta + shiftAngle) *PI/180.0f;
        t_theta = t_theta;
                        
        float sin_theta = sin(t_theta);
        float cos_theta = cos(t_theta);

        cudaMemcpy(d_proj_single,h_proj_data_filtered+R*Z_prj*j,sizeof(float)*R*Z_prj,cudaMemcpyDefault);

        kernel_FDK_reconstruction<<<dimGrid_backprj,dimBlock_backprj>>>(h_volume,d_proj_single,sin_theta,cos_theta);          
        
        cudaStreamSynchronize(0);            
                                
        if (j % 20 == 0)
           printf(" - have done %d projections... \n", j);					
        	
	}    
    
    cout<<"OK!"<<endl;
    
    cudaFreeHost(h_proj_data_filtered);
    cudaFree(d_proj_single);
    
}