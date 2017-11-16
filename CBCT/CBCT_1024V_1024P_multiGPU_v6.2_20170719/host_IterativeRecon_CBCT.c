void Reconstruction_3D_ray_driven_CBCT(float *h_volume, float *h_proj_data, float beta_temp)

{    
    // set Timer starter
    cudaEvent_t start,stop;
    cudaEventCreate(&start);   
    cudaEventCreate(&stop);       
    cudaEventRecord(start,0);    //start GPU timing    
    
	size_t size_proj_single = sizeof(float)*Z_prj*R;
	size_t size_volume = sizeof(float)*M*N*ZETA;    

    // allocate device memory for correction array result
	float *d_proj_correction = NULL;
	checkCuda (cudaMalloc((void**)&d_proj_correction, size_proj_single));
    
    float *d_volume;
	cudaMalloc((void**)&d_volume, size_volume);
    checkCuda (cudaMemcpy(d_volume,h_volume,size_volume,cudaMemcpyHostToDevice));    
    
    float *h_proj_single = NULL;        

	// setup execution parameters for projection / correction  
    dim3  dimGrid(2, Z_prj);
    dim3  dimBlock(R/2);
    
	//setup execution parameters for backprojection (reconstruction volume) 
	dim3  dimGrid_backprj(4,N,ZETA); 
    dim3  dimBlock_backprj(M/4);

	for (int j=0; j<Nviews; j=j+1)
	{
        
    	float t_theta = (float)(j*us_rate + initialAngle);
        t_theta = (t_theta + shiftAngle) *PI/180.0f;
        t_theta = -t_theta;
            
        h_proj_single = h_proj_data + Z_prj * R * j;
            
        float sin_theta = sin(t_theta);
        float cos_theta = cos(t_theta);

        forward_ray_driven_3d_kernel_correction<<<dimGrid,dimBlock >>>(d_volume, d_proj_correction, h_proj_single, sin_theta, cos_theta, 1); 
          
        backprj_ray_driven_3d_kernel<<<dimGrid_backprj, dimBlock_backprj>>>(d_volume, d_proj_correction, beta_temp,sin_theta,cos_theta, 1);
        
        cudaStreamSynchronize(0);            
                                
        if (j % 20 == 0)
           printf(" - have done %d projections... \n", j);					
        	
	}   
    		
    // End Timer
	cudaEventRecord(stop, 0);   // end GPU timing
	cudaEventSynchronize(stop); // blocks CPU execution until the specified event is recorded.
    float time_elapsed;
    cudaEventElapsedTime(&time_elapsed, start, stop);  //  this value has a resolution of approximately one half microsecond.
    cudaEventDestroy(start);
    cudaEventDestroy(stop);  
    printf("GPU kernel Timing (cudaEventRecord): %f (ms)\n", time_elapsed );    
     
    cudaMemcpy(h_volume,d_volume,size_volume,cudaMemcpyDeviceToHost);
    cudaFree(d_proj_correction);
    cudaFree(d_volume);    
    
}


void Forward_3D_ray_driven_siddon(float *h_volume, float *h_proj_data)
{
	
    // set Timer starter
    cudaEvent_t start,stop;
    cudaEventCreate(&start);   
    cudaEventCreate(&stop);       
    cudaEventRecord(start,0);    //start GPU timing
            
    float *d_volume;
	checkCuda(cudaMalloc((void**)&d_volume, sizeof(float)*M*N*ZETA));
    cudaMemcpy(d_volume,h_volume,sizeof(float)*M*N*ZETA,cudaMemcpyHostToDevice);
    
    float *d_proj_data;
	checkCuda(cudaMalloc((void**)&d_proj_data, sizeof(float)*R*Z_prj*Nviews));

    float *d_proj_single = NULL;    

    
    // setup execution parameters for projection and correction  
    dim3  dimGrid(2,Z_prj);
    dim3  dimBlock(R/2,1);
  
    printf("  * generating forward projections ... \n");
    
    float t_theta;
    
    for (int j=0; j<Nviews; j=j+1)
    {        
        t_theta = (float)j*us_rate+initialAngle;
        t_theta = (t_theta + shiftAngle) *PI/180.0f;    
        t_theta = -t_theta;
        
		float sin_theta = sin(t_theta);
        float cos_theta = cos(t_theta);
        
        d_proj_single = d_proj_data + Z_prj * R * j;
        
        forward_ray_driven_3d_kernel_correction<<<dimGrid,dimBlock >>>(d_volume, d_proj_single, d_proj_single, sin_theta, cos_theta, 0); 
        
	}
    
    cudaMemcpy(h_proj_data, d_proj_data, sizeof(float)*R*Z_prj*Nviews, cudaMemcpyDefault);
        
    // End Timer
	cudaEventRecord(stop, 0);   // end GPU timing
	cudaEventSynchronize(stop); // blocks CPU execution until the specified event is recorded.
    float time_elapsed;
    cudaEventElapsedTime(&time_elapsed, start, stop);  //  this value has a resolution of approximately one half microsecond.
    cudaEventDestroy(start);
    cudaEventDestroy(stop);      
    
    cudaFree(d_volume);
    cudaFree(d_proj_data);
    
    printf("  * Have generated %d projections. \n",Nviews);    
    printf("  * Processing time: %f (ms)\n", time_elapsed );      

}


void CheckMatchedJointOperator(float *h_volume)
{
    size_t size_proj_single = sizeof(float)*R*Z_prj;
    size_t size_volume = sizeof(float)*M*N*ZETA;

    float *h_proj_single = (float *)malloc(sizeof(float)*R*Z_prj); 
	bzero(h_proj_single,size_proj_single);

    float *h_volume_backprj = (float *)malloc(sizeof(float)*M*N*ZETA); 
	bzero(h_volume_backprj,size_volume);    
	    
    // allocate device memory for a single projection data
    float *d_proj_single = NULL;
    cudaMalloc((void**)&d_proj_single, size_proj_single);
	cudaMemset(d_proj_single, 0, size_proj_single );
    
	// allocate device memory for the 3D volumn 
    float *d_volume = NULL;
    cudaMalloc((void**)&d_volume, size_volume);
	cudaMemcpy(d_volume, h_volume, size_volume, cudaMemcpyHostToDevice);
        
    float *d_volume_backprj = NULL;   // the exact adjoint operator
    cudaMalloc((void**)&d_volume_backprj, size_volume);
	cudaMemset(d_volume_backprj, 0, size_volume);        
   
    // setup execution parameters for projection and correction  
    dim3  dimGrid(2,Z_prj);
    dim3  dimBlock(R/2,1);
    
    dim3  dimGrid2(2,N,ZETA);
    dim3  dimBlock2(M/2,1);            
  
    printf("  * generating forward projections ... \n");
    
    float t_theta;
    
    for (int j=0; j<=0; j=j+1)
    {                
        t_theta = (float)j*us_rate+initialAngle;
        t_theta = (t_theta + shiftAngle) *PI/180.0f;    
        t_theta = -t_theta;
        
		float sin_theta = sin(t_theta);
        float cos_theta = cos(t_theta);
        
        forward_ray_driven_3d_kernel_correction<<<dimGrid,dimBlock >>>(d_volume, d_proj_single, d_proj_single, sin_theta, cos_theta, 0);
        
        cudaMemcpy(h_proj_single, d_proj_single, size_proj_single, cudaMemcpyDeviceToHost);
        
        backprj_ray_driven_3d_kernel<<<dimGrid2, dimBlock2 >>>(d_volume_backprj, d_proj_single, 1.0, sin_theta, cos_theta, 0);
        
        cudaMemcpy(h_volume_backprj, d_volume_backprj, size_volume, cudaMemcpyDeviceToHost);
        
	}
        
    double InnerProduct1 = 0.0f;
    double InnerProduct2 = 0.0f;    

    for (int i=0; i<R*Z_prj; i++)
         InnerProduct1 = InnerProduct1 + (double)h_proj_single[i]*(double)h_proj_single[i]; 

    printf("The 1st inner product = %f\n",InnerProduct1);
    
	for (int i=0; i<M*N*ZETA; i++)
        InnerProduct2 = InnerProduct2 + (double)h_volume[i] * (double)h_volume_backprj[i];   
    
    printf("The 2nd inner product = %f\n",InnerProduct2); 
            
    free(h_proj_single);    
    free(h_volume_backprj);
    cudaFree(d_proj_single);	        
    cudaFree(d_volume);
    cudaFree(d_volume_backprj);
}
