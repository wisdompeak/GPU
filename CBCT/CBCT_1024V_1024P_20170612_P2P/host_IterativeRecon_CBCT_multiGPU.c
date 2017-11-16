// This function is used for radon server (Tesla K40) which it is fast for p2p transfer
void Reconstruction_3D_ray_driven_CBCT(float *d_volume, float *d_proj_data, float beta_temp)
{        
    int num_devices;    
    cudaGetDeviceCount(&num_devices);
    if (num_devices<Number_of_Devices)
    {
        printf("The totoal number of GPUs = %d, which is fewer than the required %d GPUs\n", num_devices, Number_of_Devices);
        exit(0);
    }
    
    // set Timer starter
    cudaSetDevice(Default_GPU);
    cudaEvent_t start,stop;
    cudaEventCreate(&start);   
    cudaEventCreate(&stop);       
    cudaEventRecord(start,0);    //start GPU timing    

	size_t size_proj_single = sizeof(float)*Z_prj*R;
    size_t size_proj_single_portion= sizeof(float)*R*Z_prj/Number_of_Devices;
    size_t size_volume= sizeof(float)*M*N*ZETA;
    
    float *d_proj_single = NULL;    
           
    // proj_correction_single each GPU
    float * d_proj_addr[Number_of_Devices];
    float * d_proj_portion_addr[Number_of_Devices];
    float * d_volume_addr[Number_of_Devices];        
    for (int i=0; i<Number_of_Devices; i++)
    {
       cudaSetDevice(i); 
       cudaMalloc((void**)&d_proj_addr[i], size_proj_single);  //blocking actions
       cudaMalloc((void**)&d_proj_portion_addr[i], size_proj_single_portion); 
       cudaMalloc((void**)&d_volume_addr[i], size_volume);        
    }             
    
	// setup execution parameters for projection / correction  
    dim3  dimGrid(2, Z_prj/Number_of_Devices);
    dim3  dimBlock(R/2);
    
	//setup execution parameters for backprojection (reconstruction volume) 
	dim3  dimGrid_backprj(4,N,ZETA/Number_of_Devices); 
    dim3  dimBlock_backprj(M/4);

	for (int j=0; j<Nviews; j=j+1)
	{
        
    	cudaSetDevice(Default_GPU);
        
        float t_theta = (float)(j*us_rate + initialAngle);
        t_theta = (t_theta + shiftAngle) *PI/180.0f;
        t_theta = -t_theta;
            
        d_proj_single = d_proj_data + Z_prj * R * j;
            
        float sin_theta = sin(t_theta);
        float cos_theta = cos(t_theta);
        
        /************ Forward **********************/
        
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i); 
            cudaMemcpyAsync(d_volume_addr[i],d_volume,size_volume,cudaMemcpyDefault);
        }                
        
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i); 
            forward_ray_driven_3d_kernel_correction_multiGPU<<<dimGrid,dimBlock >>>(d_volume_addr[i], d_proj_portion_addr[i], d_proj_single+R*Z_prj/Number_of_Devices*i, sin_theta, cos_theta, i, 1); 
        }               
        
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i); 
            cudaMemcpyAsync(d_proj_addr[Default_GPU]+R*Z_prj/Number_of_Devices*i,d_proj_portion_addr[i],size_proj_single_portion,cudaMemcpyDefault);
        }  
        
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i); 
            cudaDeviceSynchronize();            
        }
        
        
        /************ Backprojection **********************/
              
        for (int i=0; i<Number_of_Devices; i++)
        {
        	cudaSetDevice(i);
            cudaMemcpyAsync(d_proj_addr[i],d_proj_addr[Default_GPU],size_proj_single,cudaMemcpyDefault);
        }    
                
        for (int i=0; i<Number_of_Devices; i++)
        {              
            cudaSetDevice(i);
            backprj_ray_driven_3d_kernel_multiGPU<<<dimGrid_backprj, dimBlock_backprj>>>(d_volume+M*N*ZETA/Number_of_Devices*i, d_proj_addr[i], beta_temp, sin_theta, cos_theta, i, 1); 
        }        
                
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
        }
        
        if (j % 20 == 0)
           printf(" - have done %d projections... \n", j);					
        	
	}   
		
    // End Timer
	cudaSetDevice(Default_GPU);
    cudaEventRecord(stop, 0);   // end GPU timing
	cudaEventSynchronize(stop); // blocks CPU execution until the specified event is recorded.
    float time_elapsed;
    cudaEventElapsedTime(&time_elapsed, start, stop);  //  this value has a resolution of approximately one half microsecond.
    cudaEventDestroy(start);
    cudaEventDestroy(stop);  
    printf("GPU kernel Timing (cudaEventRecord): %f (ms)\n", time_elapsed );    
         
    for (int i=0; i<Number_of_Devices; i++)
    {
        cudaSetDevice(i);
        cudaFree(d_proj_addr[i]);
        cudaFree(d_proj_portion_addr[i]);
        cudaFree(d_volume_addr[i]);
	}    
    
    cudaSetDevice(Default_GPU);
    
}


void Forward_3D_ray_driven_siddon(float *d_volume, float *d_proj_data)
{
	
    for (int i=0; i<Number_of_Devices; i++) 
    {
        if (i==Default_GPU) continue;
        cudaSetDevice(i);
        cudaError_t cuda_info=cudaDeviceEnablePeerAccess(Default_GPU,0);
        fprintf(stderr, "CUDA Runtime Error: %s :%d - %d\n", cudaGetErrorString(cuda_info),i,Default_GPU);
    }
    
    // set Timer starter
	cudaSetDevice(Default_GPU);
    cudaEvent_t start,stop;
    cudaEventCreate(&start);   
    cudaEventCreate(&stop);       
    cudaEventRecord(start,0);    //start GPU timing
        
	// assgin device memory pointer for a single projection
    float *d_proj_single = NULL;    
    
    // proj_single_portion for each GPU
    size_t size_proj_single_portion= sizeof(float)*R*Z_prj/Number_of_Devices;
    size_t size_volume= sizeof(float)*M*N*ZETA;
    
    float * d_proj_portion_addr[Number_of_Devices];
    float * d_volume_addr[Number_of_Devices];    
    for (int i=0; i<Number_of_Devices; i++)
    {
       cudaSetDevice(i); 
       cudaMalloc((void**)&d_proj_portion_addr[i], size_proj_single_portion); 
       cudaMalloc((void**)&d_volume_addr[i], size_volume); 
    } 
    
    for (int i=0; i<Number_of_Devices; i++)
    {
       cudaSetDevice(i); 
       cudaMemcpyAsync(d_volume_addr[i],d_volume,size_volume,cudaMemcpyDefault);
    }        
    for (int i=0; i<Number_of_Devices; i++)
    {
       cudaSetDevice(i); 
       cudaDeviceSynchronize();
    }        
    
    cudaSetDevice(Default_GPU); 

    // setup execution parameters for projection and correction  
    dim3  dimGrid(2,Z_prj/Number_of_Devices);
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
        
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i); 
            forward_ray_driven_3d_kernel_correction_multiGPU<<<dimGrid,dimBlock >>>(d_volume_addr[i], d_proj_portion_addr[i], d_proj_portion_addr[i], sin_theta, cos_theta, i, 0); 
        }
        
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i); 
            cudaMemcpyAsync(d_proj_single+R*Z_prj/Number_of_Devices*i,d_proj_portion_addr[i],size_proj_single_portion,cudaMemcpyDefault);
        }        
        
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i); 
            cudaDeviceSynchronize();
        }        
               
	}
    
    // End Timer
	cudaSetDevice(Default_GPU);
    cudaEventRecord(stop, 0);   // end GPU timing
	cudaEventSynchronize(stop); // blocks CPU execution until the specified event is recorded.
    float time_elapsed;
    cudaEventElapsedTime(&time_elapsed, start, stop);  //  this value has a resolution of approximately one half microsecond.
    cudaEventDestroy(start);
    cudaEventDestroy(stop);  
    printf("  * Have generated %d projections. \n",Nviews);    
    printf("  * Processing time: %f (ms)\n", time_elapsed );    
    
    
    for (int i=0; i<Number_of_Devices; i++)
    {
        cudaSetDevice(i);
        cudaFree(d_proj_portion_addr[i]);
        cudaFree(d_volume_addr[i]);
	}    
    
    cudaSetDevice(Default_GPU);    

}


void CheckMatchedJointOperator(float *h_volume)
{
    cudaSetDevice(Default_GPU);   
    
    size_t size_proj_single = sizeof(float)*R*Z_prj;
    size_t d_volume_size = sizeof(float)*M*N*ZETA;

    float *h_proj_single = (float *)malloc(sizeof(float)*R*Z_prj); 
	bzero(h_proj_single,size_proj_single);

    float *h_volume_backprj = (float *)malloc(sizeof(float)*M*N*ZETA); 
	bzero(h_volume_backprj,d_volume_size);    
	    
    // allocate device memory for a single projection data
    float *d_proj_single = NULL;
    cudaMalloc((void**)&d_proj_single, size_proj_single);
	cudaMemset(d_proj_single, 0, size_proj_single );
    
	// allocate device memory for the 3D volumn 
    float *d_volume = NULL;
    cudaMalloc((void**)&d_volume, d_volume_size);
	cudaMemcpy(d_volume, h_volume, d_volume_size, cudaMemcpyHostToDevice);
        
    float *d_volume_backprj = NULL;   // the exact adjoint operator
    cudaMalloc((void**)&d_volume_backprj, d_volume_size);
	cudaMemset(d_volume_backprj, 0, d_volume_size);        
   
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
        
        cudaMemcpy(h_volume_backprj, d_volume_backprj, d_volume_size, cudaMemcpyDeviceToHost);
        
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