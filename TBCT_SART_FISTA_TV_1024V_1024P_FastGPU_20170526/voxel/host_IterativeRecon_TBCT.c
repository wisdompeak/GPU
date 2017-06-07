void Reconstruction_3D_ray_driven_TBCT(float *d_volume, float *d_proj_data, float beta_temp)
    /* 
     * Reference: "Accelerating simultaneous algebraic reconstruction technique with motion compensation using CUDA-enabled GPU" 
     * Wai-Man Pang, CUHK
     * Eq. (3) & (4)
     */	
{
       
    // set Timer starter
    cudaEvent_t start,stop;
    cudaEventCreate(&start);   
    cudaEventCreate(&stop);       
    cudaEventRecord(start,0);    //start GPU timing

	// allocate device memory for correction array result
	float *d_proj_correction = NULL;
	size_t d_proj_single_size = sizeof(float)*Z_prj*R*N_source;
	cudaMalloc((void**)&d_proj_correction, d_proj_single_size);
	cudaMemset(d_proj_correction, 0, d_proj_single_size );
   
    float *d_proj_single = NULL;    

    size_t d_volume_size = sizeof(float)*M*N*ZETA;

	// allocate device memory for result volume with all length sum in each voxel 
	float *d_volumn_length = NULL;
    cudaMalloc((void**)&d_volumn_length, d_volume_size);
    
	// allocate device memory for result volume with all weighted length sum in each voxel 
	float *d_volumn_weightedSum = NULL;
    cudaMalloc((void**)&d_volumn_weightedSum, d_volume_size);    

	// setup execution parameters for projection / correction  
    dim3  dimGrid(Z_prj,N_source);
    dim3  dimBlock(R,1);
    
	//setup execution parameters for backprojection (reconstruction volume) 
	dim3  dimGrid_backprj(N*2,ZETA,N_source); 
    dim3  dimBlock_backprj(M/2,1);

	for (int j=0; j<Nviews; j=j+1)
        {
        
            float t_theta = (float)(j*us_rate + initialAngle);
            t_theta = (t_theta + shiftAngle) *PI/180.0f;
            t_theta = -t_theta;
            
            d_proj_single = d_proj_data + Z_prj*R*N_source*j;
            
            float sin_theta = sin(t_theta);
            float cos_theta = cos(t_theta);
            
            forward_ray_driven_3d_kernel_correction<<<dimGrid,dimBlock >>>(d_volume, d_proj_correction, d_proj_single, sin_theta, cos_theta, 1);

            cudaStreamSynchronize(0);

            backprj_ray_driven_3d_kernel<<<dimGrid_backprj, dimBlock_backprj>>>(d_volume, d_proj_correction, beta_temp,sin_theta,cos_theta, 0);
        
            cudaMemset(d_proj_correction, 0, d_proj_single_size);            
	
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

    cudaFree(d_proj_correction);	
        
}


void Forward_3D_ray_driven_siddon_TBCT(float *d_volume, float *d_proj_data)
{
	float t_theta;
            
    // set Timer starter
    cudaEvent_t start,stop;
    cudaEventCreate(&start);   
    cudaEventCreate(&stop);       
    cudaEventRecord(start,0);    //start GPU timing

	// assgin device memory pointer for a single projection
    float *d_proj_single = NULL;      

    // setup execution parameters for projection 
    dim3  dimGrid(Z_prj,N_source);
    dim3  dimBlock(R,1);
  
    printf("  * generating forward projections ... \n");
    
    for (int j=0; j<Nviews; j=j+1)
    {        
        t_theta = (float)j*us_rate+initialAngle;
        t_theta = (t_theta + shiftAngle) *PI/180.0f;    
        t_theta = -t_theta;
        
		float sin_theta = sin(t_theta);
        float cos_theta = cos(t_theta);
        
        d_proj_single = d_proj_data + Z_prj*R*N_source*j;
        
        forward_ray_driven_3d_kernel_correction<<<dimGrid,dimBlock >>>(d_volume, d_proj_single, d_proj_single, sin_theta, cos_theta, 0); 
	}

    printf("  * Have generated %d projections. \n",Nviews);
    
    // End Timer
	cudaEventRecord(stop, 0);   // end GPU timing
	cudaEventSynchronize(stop); // blocks CPU execution until the specified event is recorded.
    float time_elapsed;
    cudaEventElapsedTime(&time_elapsed, start, stop);  //  this value has a resolution of approximately one half microsecond.
    cudaEventDestroy(start);
    cudaEventDestroy(stop);  
    printf("  * Processing time: %f (ms)\n", time_elapsed );           

}



void CheckMatchedJointOperator(float *h_volume)
{
    size_t d_proj_single_size = sizeof(float)*R*Z_prj*N_source;
    size_t d_volume_size = sizeof(float)*M*N*ZETA;

    float *h_proj_single = (float *)malloc(d_proj_single_size); 
	bzero(h_proj_single,d_proj_single_size);

    float *h_volume_backprj = (float *)malloc(d_volume_size); 
	bzero(h_volume_backprj,d_volume_size);    
	    
    // allocate device memory for a single projection data
    float *d_proj_single = NULL;
    cudaMalloc((void**)&d_proj_single, d_proj_single_size);
	cudaMemset(d_proj_single, 0, d_proj_single_size );
    
	// allocate device memory for the 3D volumn 
    float *d_volume = NULL;
    cudaMalloc((void**)&d_volume, d_volume_size);
	cudaMemcpy(d_volume, h_volume, d_volume_size, cudaMemcpyHostToDevice);
        
    float *d_volume_backprj = NULL;   // the exact adjoint operator
    cudaMalloc((void**)&d_volume_backprj, d_volume_size);
	cudaMemset(d_volume_backprj, 0, d_volume_size);        
   
    // setup execution parameters for projection and correction  
    dim3  dimGrid(Z_prj,N_source);
    dim3  dimBlock(R,1);
    
	dim3  dimGrid_backprj(N*2,ZETA,N_source); 
    dim3  dimBlock_backprj(M/2,1);         
  
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
        
        cudaMemcpy(h_proj_single, d_proj_single, d_proj_single_size, cudaMemcpyDeviceToHost);
        
        backprj_ray_driven_3d_kernel<<<dimGrid_backprj, dimBlock_backprj>>>(d_volume_backprj, d_proj_single, 1.0, sin_theta, cos_theta, 1);
        
        cudaMemcpy(h_volume_backprj, d_volume_backprj, d_volume_size, cudaMemcpyDeviceToHost);
        
	}
        
    double InnerProduct1 = 0.0f;
    double InnerProduct2 = 0.0f;    

    for (int i=0; i<R*Z_prj*N_source; i++)
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

