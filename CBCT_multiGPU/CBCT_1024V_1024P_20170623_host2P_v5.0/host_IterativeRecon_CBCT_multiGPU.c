// This function is used for aristotle server (Tesla K20) which it is slow for p2p transfer
void Reconstruction_3D_ray_driven_CBCT(float *h_volume, float *h_proj_data, float beta_temp)
{        
    // set Timer starter
    cudaSetDevice(Default_GPU);
    cudaEvent_t start,stop;
    cudaEventCreate(&start);   
    cudaEventCreate(&stop);       
    cudaEventRecord(start,0);    //start GPU timing    

	size_t size_proj_single = sizeof(float)*Z_prj*R;
    size_t size_volume= sizeof(float)*M*N*ZETA;
    size_t size_volume_portion= sizeof(float)*M*N*ZETA/Number_of_Devices;
            
    float *h_proj_single = NULL;
	cudaHostAlloc((void**)&h_proj_single, size_proj_single, cudaHostAllocDefault);
    float *h_proj_sumLen = NULL;
	cudaHostAlloc((void**)&h_proj_sumLen, size_proj_single, cudaHostAllocDefault);
    float *h_proj_weightedLen = NULL;
	cudaHostAlloc((void**)&h_proj_weightedLen, size_proj_single, cudaHostAllocDefault);
    
    // device memory allocation for each GPU
    float * d_proj_sumLen_addr[Number_of_Devices];
    float * d_proj_weightedLen_addr[Number_of_Devices];
    float * d_proj_addr[Number_of_Devices];
    float * d_volume_addr[Number_of_Devices];        
    for (int i=0; i<Number_of_Devices; i++)
    {
       cudaSetDevice(i); 
       cudaMalloc((void**)&d_proj_sumLen_addr[i], size_proj_single);  
       cudaMalloc((void**)&d_proj_weightedLen_addr[i], size_proj_single); 
       cudaMalloc((void**)&d_proj_addr[i], size_proj_single); 
       cudaMalloc((void**)&d_volume_addr[i], size_volume_portion); 
       cudaMemset(d_volume_addr[i], 0, size_volume_portion);
    }             
    
    for (int i=0; i<Number_of_Devices; i++)
    {
        cudaSetDevice(i);
        cudaMemcpyAsync(d_volume_addr[i],h_volume+M*N*ZETA/Number_of_Devices*i,size_volume_portion,cudaMemcpyDefault);
    }
    for (int i=0; i<Number_of_Devices; i++)
    {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }        
    
	// setup execution parameters for projection / correction  
    dim3  dimGrid_forward(2, Z_prj);
    dim3  dimBlock_forward(R/2);
    
	//setup execution parameters for backprojection
	dim3  dimGrid_backprj(4,N,ZETA/Number_of_Devices); 
    dim3  dimBlock_backprj(M/4);

	for (int j=0; j<Nviews; j=j+1)
	{
                
        float t_theta = (float)(j*us_rate + initialAngle);
        t_theta = (t_theta + shiftAngle) *PI/180.0f;
        t_theta = -t_theta;                    
            
        float sin_theta = sin(t_theta);
        float cos_theta = cos(t_theta);
        
        /************ Forward **********************/
        
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i); 
            forward_ray_driven_3d_kernel_correction_separate<<<dimGrid_forward,dimBlock_forward >>>(d_volume_addr[i], d_proj_sumLen_addr[i], d_proj_weightedLen_addr[i], sin_theta, cos_theta, i);
        }
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
        }             
        
        memset(h_proj_sumLen, 0, size_proj_single);
        memset(h_proj_weightedLen, 0, size_proj_single);
                
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i); 
            kernel_add_proj<<<dimGrid_forward,dimBlock_forward>>>(h_proj_weightedLen, d_proj_weightedLen_addr[i]);
            kernel_add_proj<<<dimGrid_forward,dimBlock_forward>>>(h_proj_sumLen, d_proj_sumLen_addr[i]);
            cudaDeviceSynchronize(); 
        }       
                
        cudaSetDevice(Default_GPU); 
        kernel_divide_proj<<<dimGrid_forward,dimBlock_forward>>>(h_proj_single, h_proj_data+Z_prj*R*j, h_proj_sumLen, h_proj_weightedLen);
        cudaDeviceSynchronize(); 
                
        // test purpose
//         if (j==0)
//         {
//             SaveDeviceDataToFile(h_proj_sumLen,R*Z_prj,"../projCheck.data");        
//             exit(0);
//         }  
        
//         cout<<h_proj_single[R*400+400]<<endl;
                
        /************ Backprojection **********************/
                      
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i);
            cudaMemcpyAsync(d_proj_addr[i],h_proj_single,size_proj_single,cudaMemcpyDefault);
        }    
        
        for (int i=0; i<Number_of_Devices; i++)
        {              
            cudaSetDevice(i);
            backprj_ray_driven_3d_kernel_multiGPU<<<dimGrid_backprj, dimBlock_backprj>>>(d_volume_addr[i], d_proj_addr[i], beta_temp, sin_theta, cos_theta, i, 1); 
        }                        
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
        }
        
        if (j % 20 == 0)
           printf(" - have done %d projections... \n", j);					
        	
	}   
		
	for (int i=0; i<Number_of_Devices; i++)
    {              
        cudaMemcpyAsync(h_volume+M*N*ZETA/Number_of_Devices*i,d_volume_addr[i],size_volume/Number_of_Devices,cudaMemcpyDefault);
	}                        
    cudaDeviceSynchronize();

    // End Timer
	cudaSetDevice(Default_GPU);
    cudaEventRecord(stop, 0);   // end GPU timing
	cudaEventSynchronize(stop); // blocks CPU execution until the specified event is recorded.
    float time_elapsed;
    cudaEventElapsedTime(&time_elapsed, start, stop);  //  this value has a resolution of approximately one half microsecond.
    cudaEventDestroy(start);
    cudaEventDestroy(stop);  
    printf("GPU kernel Timing (cudaEventRecord): %f (ms)\n", time_elapsed );    

    cudaFreeHost(h_proj_single);
    cudaFreeHost(h_proj_sumLen);
    cudaFreeHost(h_proj_weightedLen);
    
    for (int i=0; i<Number_of_Devices; i++)
    {
        cudaSetDevice(i);
        cudaFree(d_proj_addr[i]);
        cudaFree(d_volume_addr[i]);
        cudaFree(d_proj_sumLen_addr[i]);
        cudaFree(d_proj_weightedLen_addr[i]);
	}    
    
    cudaSetDevice(Default_GPU);
    
}


void Forward_3D_ray_driven_siddon(float *h_volume, float *h_proj_data)
{
	
//     for (int i=0; i<Number_of_Devices; i++) 
//     {
//         if (i==Default_GPU) continue;
//         cudaSetDevice(i);
//         cudaError_t cuda_info=cudaDeviceEnablePeerAccess(Default_GPU,0);
//         fprintf(stderr, "CUDA Runtime Error: %s :%d - %d\n", cudaGetErrorString(cuda_info),i,Default_GPU);
//     }   
    
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
       
    // proj_single_portion for each GPU
    size_t size_proj_single_portion = sizeof(float)*R*Z_prj/Number_of_Devices;
    size_t size_proj_single = sizeof(float)*R*Z_prj;    
    size_t size_volume= sizeof(float)*M*N*ZETA; 
        
    float *h_proj_single = NULL;
	cudaHostAlloc((void**)&h_proj_single, size_proj_single,cudaHostAllocDefault);
            
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
       cudaMemcpyAsync(d_volume_addr[i],h_volume,size_volume,cudaMemcpyDefault);
    }        
    for (int i=0; i<Number_of_Devices; i++)
    {
       cudaSetDevice(i); 
       cudaDeviceSynchronize();
    }        

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
                
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i); 
            forward_ray_driven_3d_kernel_correction_multiGPU<<<dimGrid,dimBlock >>>(d_volume_addr[i], d_proj_portion_addr[i], d_proj_portion_addr[i], sin_theta, cos_theta, i, 0); 
        }        
        
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i); 
            cudaMemcpyAsync(h_proj_single+R*Z_prj/Number_of_Devices*i,d_proj_portion_addr[i],size_proj_single_portion,cudaMemcpyDefault);
        }                
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i); 
            cudaDeviceSynchronize();
        }
                  
        cudaSetDevice(Default_GPU); 
        cudaMemcpyAsync(h_proj_data+Z_prj*R*j,h_proj_single,size_proj_single,cudaMemcpyDefault); 
                
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
        
    cudaFree(h_proj_single);
    
    for (int i=0; i<Number_of_Devices; i++)
    {
        cudaFree(d_proj_portion_addr[i]);
        cudaFree(d_volume_addr[i]);
	}    
    
    cudaSetDevice(Default_GPU);    

}


void CheckMatchedJointOperator(float *h_volume)
{        
    // set Timer starter
    cudaSetDevice(Default_GPU);
    cudaEvent_t start,stop;
    cudaEventCreate(&start);   
    cudaEventCreate(&stop);       
    cudaEventRecord(start,0);    //start GPU timing    

	size_t size_proj_single = sizeof(float)*Z_prj*R;
    size_t size_volume= sizeof(float)*M*N*ZETA;
    size_t size_volume_portion= sizeof(float)*M*N*ZETA/Number_of_Devices;
            
    float *h_proj_weightedLen = NULL;
	cudaHostAlloc((void**)&h_proj_weightedLen, size_proj_single, cudaHostAllocDefault);
    float *h_volume_backprj = NULL;
	cudaHostAlloc((void**)&h_volume_backprj, size_volume, cudaHostAllocDefault);    
    memset(h_volume_backprj,0,size_volume);
    
    // device memory allocation for each GPU
    float * d_proj_sumLen_addr[Number_of_Devices];
    float * d_proj_weightedLen_addr[Number_of_Devices];
    float * d_proj_addr[Number_of_Devices];
    float * d_volume_addr[Number_of_Devices];        
    for (int i=0; i<Number_of_Devices; i++)
    {
       cudaSetDevice(i); 
       cudaMalloc((void**)&d_proj_sumLen_addr[i], size_proj_single);  
       cudaMalloc((void**)&d_proj_weightedLen_addr[i], size_proj_single); 
       cudaMalloc((void**)&d_proj_addr[i], size_proj_single); 
       cudaMalloc((void**)&d_volume_addr[i], size_volume_portion); 
       cudaMemset(d_volume_addr[i], 0, size_volume_portion);
    }             
    
    for (int i=0; i<Number_of_Devices; i++)
    {
        cudaSetDevice(i);
        cudaMemcpyAsync(d_volume_addr[i],h_volume+M*N*ZETA/Number_of_Devices*i,size_volume_portion,cudaMemcpyDefault);
    }
    for (int i=0; i<Number_of_Devices; i++)
    {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }        
    
	// setup execution parameters for projection / correction  
    dim3  dimGrid_forward(2, Z_prj);
    dim3  dimBlock_forward(R/2);
    
	//setup execution parameters for backprojection
	dim3  dimGrid_backprj(4,N,ZETA/Number_of_Devices); 
    dim3  dimBlock_backprj(M/4);

	for (int j=0; j<=0; j=j+1)
	{
                
        float t_theta = (float)(j*us_rate + initialAngle);
        t_theta = (t_theta + shiftAngle) *PI/180.0f;
        t_theta = -t_theta;                    
            
        float sin_theta = sin(t_theta);
        float cos_theta = cos(t_theta);
        
        /************ Forward **********************/
        
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i); 
            forward_ray_driven_3d_kernel_correction_separate<<<dimGrid_forward,dimBlock_forward >>>(d_volume_addr[i], d_proj_sumLen_addr[i], d_proj_weightedLen_addr[i], sin_theta, cos_theta, i);
        }
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
        }             
        
        memset(h_proj_weightedLen, 0, size_proj_single);
                
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i); 
            kernel_add_proj<<<dimGrid_forward,dimBlock_forward>>>(h_proj_weightedLen, d_proj_weightedLen_addr[i]);
            cudaMemset(d_volume_addr[i],0,size_volume_portion);
            cudaDeviceSynchronize(); 
        }                       
                
        /************ Backprojection **********************/
                      
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i);
            cudaMemcpyAsync(d_proj_addr[i],h_proj_weightedLen,size_proj_single,cudaMemcpyDefault);
        }    
        
        for (int i=0; i<Number_of_Devices; i++)
        {              
            cudaSetDevice(i);
            backprj_ray_driven_3d_kernel_multiGPU<<<dimGrid_backprj, dimBlock_backprj>>>(d_volume_addr[i], d_proj_addr[i], 1.0f, sin_theta, cos_theta, i, 0); 
        }                        
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
        }        				
        	
	}   
		
	for (int i=0; i<Number_of_Devices; i++)
    {              
        cudaMemcpyAsync(h_volume_backprj+M*N*ZETA/Number_of_Devices*i,d_volume_addr[i],size_volume_portion,cudaMemcpyDefault);
	}                        
    cudaDeviceSynchronize();
        
    double InnerProduct1 = 0.0f;
    double InnerProduct2 = 0.0f;    

    for (int i=0; i<R*Z_prj; i++)
         InnerProduct1 = InnerProduct1 + (double)h_proj_weightedLen[i]*(double)h_proj_weightedLen[i]; 

    printf("The 1st inner product = %f\n",InnerProduct1);
    
	for (int i=0; i<M*N*ZETA; i++)
        InnerProduct2 = InnerProduct2 + (double)h_volume[i] * (double)h_volume_backprj[i];   
    
    printf("The 2nd inner product = %f\n",InnerProduct2); 
    
	// Test Purpose
    // SaveDeviceDataToFile(h_volume_backprj,M*N*ZETA,"../projCheck2.data");        
    
    // End Timer
	cudaSetDevice(Default_GPU);
    cudaEventRecord(stop, 0);   // end GPU timing
	cudaEventSynchronize(stop); // blocks CPU execution until the specified event is recorded.
    float time_elapsed;
    cudaEventElapsedTime(&time_elapsed, start, stop);  //  this value has a resolution of approximately one half microsecond.
    cudaEventDestroy(start);
    cudaEventDestroy(stop);  
    printf("GPU kernel Timing (cudaEventRecord): %f (ms)\n", time_elapsed );    

    cudaFreeHost(h_proj_weightedLen);
    cudaFreeHost(h_volume_backprj);
    
    for (int i=0; i<Number_of_Devices; i++)
    {
        cudaSetDevice(i);
        cudaFree(d_proj_addr[i]);
        cudaFree(d_volume_addr[i]);
        cudaFree(d_proj_sumLen_addr[i]);
        cudaFree(d_proj_weightedLen_addr[i]);
	}    
    
    cudaSetDevice(Default_GPU);
    
}