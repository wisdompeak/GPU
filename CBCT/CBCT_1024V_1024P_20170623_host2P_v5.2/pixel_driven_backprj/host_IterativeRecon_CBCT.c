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
	cudaMalloc((void**)&d_proj_correction, size_proj_single);
	cudaMemset(d_proj_correction, 0, size_proj_single );          
    
    float *d_volume;
	cudaMalloc((void**)&d_volume, size_volume);
    cudaMemcpy(d_volume,h_volume,size_volume,cudaMemcpyHostToDevice); 
    
    float *h_proj_single = NULL;
        
	// allocate device memory for result volume with all length sum in each voxel 
	float *d_volumn_length = NULL;
    cudaMalloc((void**)&d_volumn_length, size_volume);
	cudaMemset(d_volumn_length, 0, size_volume );
    
	// allocate device memory for result volume with all weighted length sum in each voxel 
	float *d_volumn_weightedSum = NULL;
    cudaMalloc((void**)&d_volumn_weightedSum, size_volume);    
	cudaMemset(d_volumn_weightedSum, 0, size_volume );    

	// setup execution parameters for projection / correction  
    dim3  dimGrid(2, Z_prj);
    dim3  dimBlock(R/2);

    dim3  dimGrid2(4, Z_prj);
    dim3  dimBlock2(R/4);
    
    dim3  dimGrid_V(N,ZETA);
    dim3  dimBlock_V(M);
    
	for (int j=0; j<Nviews; j=j+1)
	{
        
    	float t_theta = (float)(j*us_rate + initialAngle);
        t_theta = (t_theta + shiftAngle) *PI/180.0f;
        t_theta = -t_theta;
            
        h_proj_single = h_proj_data + Z_prj * R * j;
            
        float sin_theta = sin(t_theta);
        float cos_theta = cos(t_theta);
        
        cudaMemset(d_volumn_length, 0,  size_volume);
        cudaMemset(d_volumn_weightedSum, 0,  size_volume);        

        forward_ray_driven_3d_kernel_correction<<<dimGrid,dimBlock >>>(d_volume, d_proj_correction, h_proj_single, sin_theta, cos_theta, 1); 

        backprj_ray_driven_3d_kernel<<<dimGrid2, dimBlock2>>>(d_volumn_weightedSum, d_volumn_length, d_proj_correction, sin_theta,cos_theta);
        
        cudaStreamSynchronize(0);
        
        update<<< dimGrid_V,dimBlock_V >>>(d_volume, d_volumn_weightedSum, d_volumn_length, beta_temp);                                                       
        
        cudaMemset(d_proj_correction, 0, Z_prj*R*sizeof(float) );	        
            
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
    cudaFree(d_volume);    

    cudaFree(d_volumn_length);
    cudaFree(d_volumn_weightedSum);    
    cudaFree(d_proj_correction);    
    
}



void Forward_3D_ray_driven_siddon(float *h_volume, float *h_proj_data)
{
	
    // set Timer starter
    cudaEvent_t start,stop;
    cudaEventCreate(&start);   
    cudaEventCreate(&stop);       
    cudaEventRecord(start,0);    //start GPU timing
            
    float *d_volume;
	cudaMalloc((void**)&d_volume, sizeof(float)*M*N*ZETA);
    cudaMemcpy(d_volume,h_volume,sizeof(float)*M*N*ZETA,cudaMemcpyHostToDevice);
    
    float *d_proj_data;
	cudaMalloc((void**)&d_proj_data, sizeof(float)*R*Z_prj*Nviews);

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
        
    float *d_volumn_weightedSum = NULL;   // the exact adjoint operator
    cudaMalloc((void**)&d_volumn_weightedSum, size_volume);
	cudaMemset(d_volumn_weightedSum, 0, size_volume);      
    
    float *d_volume_3 = NULL;   // Not useful in this function
    cudaMalloc((void**)&d_volume_3, size_volume);
	cudaMemset(d_volume_3, 0, size_volume);      
   
    // setup execution parameters for projection and correction  
    dim3  dimGrid(2,Z_prj);
    dim3  dimBlock(R/2,1);          
  
    printf("  * generating forward projections ... \n");
    
    float t_theta;
    
    for (int j=100; j<=100; j=j+1)
    {                
        t_theta = (float)j*us_rate+initialAngle;
        t_theta = (t_theta + shiftAngle) *PI/180.0f;    
        t_theta = -t_theta;
        
		float sin_theta = sin(t_theta);
        float cos_theta = cos(t_theta);
        
        forward_ray_driven_3d_kernel_correction<<<dimGrid,dimBlock >>>(d_volume, d_proj_single, d_proj_single, sin_theta, cos_theta, 0);
        
        cudaMemcpy(h_proj_single, d_proj_single, size_proj_single, cudaMemcpyDeviceToHost);
                
        backprj_ray_driven_3d_kernel<<<dimGrid, dimBlock>>>(d_volumn_weightedSum, d_volume_3, d_proj_single, sin_theta,cos_theta);
        
        cudaMemcpy(h_volume_backprj, d_volumn_weightedSum, size_volume, cudaMemcpyDeviceToHost);
        
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
    cudaFree(d_volumn_weightedSum);
    cudaFree(d_volume_3);
    
}
