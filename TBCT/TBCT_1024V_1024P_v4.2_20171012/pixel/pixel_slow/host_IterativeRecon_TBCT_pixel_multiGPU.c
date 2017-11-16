void Reconstruction_3D_ray_driven_TBCT(float *h_volume, float *h_proj_data, float beta_temp)
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
       
    size_t size_proj_single_portion = sizeof(float)*R*Z_prj/Number_of_Devices*N_source;
    size_t size_volume= sizeof(float)*M*N*ZETA; 
        
    float *h_proj_single = NULL;
    
    float *h_volume_weightedLen = NULL;
	float *h_volume_sumLen = NULL;
    cudaHostAlloc((void**)&h_volume_weightedLen, size_volume,cudaHostAllocDefault);    
	cudaHostAlloc((void**)&h_volume_sumLen, size_volume,cudaHostAllocDefault);    
	memset(h_volume_sumLen, 0, size_volume);
    memset(h_volume_weightedLen, 0, size_volume);
        
    float * d_proj_portion_addr[Number_of_Devices];
    float * d_proj_correction_portion_addr[Number_of_Devices];  
    float * d_volume_addr[Number_of_Devices];    
    float * d_volume_weightedSum_addr[Number_of_Devices];    
    float * d_volume_length_addr[Number_of_Devices];        
    for (int i=0; i<Number_of_Devices; i++)
    {
       cudaSetDevice(i); 
       cudaMalloc((void**)&d_proj_portion_addr[i], size_proj_single_portion); 
       cudaMemset(d_proj_portion_addr[i],0,size_proj_single_portion);       
       cudaMalloc((void**)&d_proj_correction_portion_addr[i], size_proj_single_portion); 
       cudaMemset(d_proj_correction_portion_addr[i],0,size_proj_single_portion);       
       cudaMalloc((void**)&d_volume_addr[i], size_volume); 
       cudaMemset(d_volume_addr[i],0,size_volume);       
       cudaMalloc((void**)&d_volume_weightedSum_addr[i], size_volume); 
       cudaMemset(d_volume_weightedSum_addr[i],0,size_volume);
       cudaMalloc((void**)&d_volume_length_addr[i], size_volume); 
       cudaMemset(d_volume_length_addr[i],0,size_volume);
    }        

    // setup execution parameters for projection and correction  
    dim3  dimGrid(2,Z_prj/Number_of_Devices,N_source);
    dim3  dimBlock(R/2,1);
    
    dim3  dimGrid_backprj(1,Z_prj/Number_of_Devices,N_source);
    dim3  dimBlock_backprj(R/1,1);
    
    dim3  dimGrid_V(N,ZETA);
    dim3  dimBlock_V(M);    
      
    float t_theta;
    
    for (int j=0; j<Nviews; j=j+1)
    {                
        t_theta = (float)j*us_rate+initialAngle;
        t_theta = (t_theta + shiftAngle) *PI/180.0f;    
        t_theta = -t_theta;
        
		float sin_theta = sin(t_theta);
        float cos_theta = cos(t_theta);
        
        h_proj_single = h_proj_data + R*Z_prj*N_source*j;

        
        /************* forward *************/
        
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i); 
            cudaMemcpyAsync(d_volume_addr[i],h_volume,size_volume,cudaMemcpyDefault);              
            for (int k=0; k<N_source; k++)
                cudaMemcpyAsync(d_proj_portion_addr[i]+R*Z_prj/Number_of_Devices*k,h_proj_single+R*Z_prj*k+R*Z_prj/Number_of_Devices*i,size_proj_single_portion/N_source,cudaMemcpyDefault);            
        }        
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i); 
            cudaDeviceSynchronize();
        }          
                
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i); 
            forward_ray_driven_3d_kernel_correction_multiGPU<<<dimGrid,dimBlock >>>(d_volume_addr[i], d_proj_correction_portion_addr[i], d_proj_portion_addr[i], sin_theta, cos_theta, i, 1); 
        }        
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i); 
            cudaDeviceSynchronize();
        }
        
        
        /**** for test use ****/
        
//         if (j==1)
//         {
//             
//             for (int i=0; i<Number_of_Devices; i++)
//         {
//             cudaSetDevice(i); 
//             for (int k=0; k<N_source; k++)
//             {
//                 cudaMemcpyAsync(h_proj_single+R*Z_prj*k+R*Z_prj/Number_of_Devices*i,d_proj_correction_portion_addr[i]+R*Z_prj/Number_of_Devices*k,size_proj_single_portion/N_source,cudaMemcpyDefault);
//             }            
//         }    
//         
// 
//             SaveDeviceDataToFile(h_proj_single,R*Z_prj*N_source,"../projCheck.data");        
//             exit(0);
//         }          
        
        /************* backprj *************/
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i); 
            cudaMemset(d_volume_weightedSum_addr[i], 0, size_volume);
            cudaMemset(d_volume_length_addr[i], 0, size_volume);
        }              
        memset(h_volume_sumLen, 0, size_volume);
        memset(h_volume_weightedLen, 0, size_volume);        
        
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i); 
            backprj_ray_driven_3d_kernel_correction_multiGPU<<<dimGrid_backprj,dimBlock_backprj >>>(d_volume_weightedSum_addr[i], d_volume_length_addr[i], d_proj_correction_portion_addr[i], sin_theta, cos_theta, i); 
        }        
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i); 
            cudaDeviceSynchronize();
        }        
                
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i); 
            kernel_add_proj<<<dimGrid_V,dimBlock_V>>>(h_volume_weightedLen, d_volume_weightedSum_addr[i]);
            kernel_add_proj<<<dimGrid_V,dimBlock_V>>>(h_volume_sumLen, d_volume_length_addr[i]);
            cudaDeviceSynchronize(); 
        } 
                 
        cudaSetDevice(0);
        cudaMemcpy(d_volume_weightedSum_addr[0],h_volume_weightedLen,size_volume,cudaMemcpyDefault);              
        cudaMemcpy(d_volume_length_addr[0],h_volume_sumLen,size_volume,cudaMemcpyDefault);                      
        update<<< dimGrid_V,dimBlock_V >>>(h_volume, d_volume_weightedSum_addr[0], d_volume_length_addr[0], beta_temp);       
        
        cudaDeviceSynchronize();         
        
        
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
    printf("  * Have generated %d projections. \n",Nviews);    
    printf("  * Processing time: %f (ms)\n", time_elapsed );    
        
    cudaFreeHost(h_volume_weightedLen);
    cudaFreeHost(h_volume_sumLen);    
    
    for (int i=0; i<Number_of_Devices; i++)
    {
        cudaSetDevice(i);  
        cudaFree(d_proj_portion_addr[i]);
        cudaFree(d_proj_correction_portion_addr[i]);
        cudaFree(d_volume_addr[i]);
        cudaFree(d_volume_weightedSum_addr[i]);
        cudaFree(d_volume_length_addr[i]);
        if (i!=Default_GPU)
            cudaDeviceReset();
	}    
    
    cudaSetDevice(Default_GPU);    

}



void Forward_3D_ray_driven_siddon_TBCT(float *h_volume, float *h_proj_data)
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
       
    size_t size_proj_single_portion = sizeof(float)*R*Z_prj/Number_of_Devices*N_source;
    size_t size_proj_single = sizeof(float)*R*Z_prj*N_source;    
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
    dim3  dimGrid(2,Z_prj/Number_of_Devices,N_source);
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
            for (int k=0; k<N_source; k++)
            {
                cudaMemcpyAsync(h_proj_single+R*Z_prj*k+R*Z_prj/Number_of_Devices*i,d_proj_portion_addr[i]+R*Z_prj/Number_of_Devices*k,size_proj_single_portion/N_source,cudaMemcpyDefault);
            }            
        }                
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i); 
            cudaDeviceSynchronize();
        }
        
        cudaSetDevice(Default_GPU); 
        
        cudaMemcpyAsync(h_proj_data+R*Z_prj*N_source*j,h_proj_single,size_proj_single,cudaMemcpyDefault); 
        
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
        
    cudaFreeHost(h_proj_single);
    
    for (int i=0; i<Number_of_Devices; i++)
    {
        cudaSetDevice(i);  
        cudaFree(d_proj_portion_addr[i]);
        cudaFree(d_volume_addr[i]);
        if (i!=Default_GPU)
            cudaDeviceReset();
	}    
    
    cudaSetDevice(Default_GPU);    

}



void CheckMatchedJointOperator(float *h_volume)
{
       
    size_t size_proj_single_portion = sizeof(float)*R*Z_prj/Number_of_Devices*N_source;
    size_t size_volume= sizeof(float)*M*N*ZETA; 
        
    float *h_proj_single = NULL;    
    float *h_volume_weightedLen = NULL;
    cudaHostAlloc((void**)&h_proj_single, sizeof(float)*R*Z_prj*N_source,cudaHostAllocDefault);    
    cudaHostAlloc((void**)&h_volume_weightedLen, size_volume,cudaHostAllocDefault);    
            
    float * d_proj_portion_addr[Number_of_Devices];
    float * d_proj_correction_portion_addr[Number_of_Devices];  
    float * d_volume_addr[Number_of_Devices];    
    float * d_volume_weightedSum_addr[Number_of_Devices];    
    float * d_volume_length_addr[Number_of_Devices];        
    for (int i=0; i<Number_of_Devices; i++)
    {
       cudaSetDevice(i); 
       cudaMalloc((void**)&d_proj_portion_addr[i], size_proj_single_portion); 
       cudaMalloc((void**)&d_proj_correction_portion_addr[i], size_proj_single_portion); 
       cudaMalloc((void**)&d_volume_addr[i], size_volume); 
       cudaMalloc((void**)&d_volume_weightedSum_addr[i], size_volume); 
       cudaMalloc((void**)&d_volume_length_addr[i], size_volume); 
       
    }        

    // setup execution parameters for projection and correction  
    dim3  dimGrid(2,Z_prj/Number_of_Devices,N_source);
    dim3  dimBlock(R/2,1);
    
    dim3  dimGrid_backprj(1,Z_prj/Number_of_Devices,N_source);
    dim3  dimBlock_backprj(R/1,1);
    
    dim3  dimGrid_V(N,ZETA);
    dim3  dimBlock_V(M);    
      
    float t_theta;
    
    for (int j=0; j<=0; j=j+1)
    {                
        t_theta = (float)j*us_rate+initialAngle;
        t_theta = (t_theta + shiftAngle) *PI/180.0f;    
        t_theta = -t_theta;
        
		float sin_theta = sin(t_theta);
        float cos_theta = cos(t_theta);
        
        
        /************* forward *************/
        
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
                
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i); 
            forward_ray_driven_3d_kernel_correction_multiGPU<<<dimGrid,dimBlock >>>(d_volume_addr[i], d_proj_correction_portion_addr[i], d_proj_portion_addr[i], sin_theta, cos_theta, i, 0); 
        }        
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i); 
            cudaDeviceSynchronize();
        }
        
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i); 
            for (int k=0; k<N_source; k++)
            {
                cudaMemcpyAsync(h_proj_single+R*Z_prj*k+R*Z_prj/Number_of_Devices*i,d_proj_correction_portion_addr[i]+R*Z_prj/Number_of_Devices*k,size_proj_single_portion/N_source,cudaMemcpyDefault);
            }            
        }                
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i); 
            cudaDeviceSynchronize();
        }        
        
        
//         /**** for test use ****/
//         if (j==0)
//         {
//             SaveDeviceDataToFile(h_proj_single,R*Z_prj*N_source,"../projCheck.data");        
//             exit(0);
//         }          
        
        /************* backprj *************/
        
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i); 
            cudaMemset(d_volume_weightedSum_addr[i], 0, size_volume);
            cudaMemset(d_volume_length_addr[i], 0, size_volume);
        }              
        memset(h_volume_weightedLen, 0, size_volume);          
        
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i); 
            backprj_ray_driven_3d_kernel_correction_multiGPU<<<dimGrid_backprj,dimBlock_backprj >>>(d_volume_weightedSum_addr[i], d_volume_length_addr[i], d_proj_correction_portion_addr[i], sin_theta, cos_theta, i);             
        }        
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i); 
            cudaDeviceSynchronize();
        }
        
        memset(h_volume_weightedLen, 0, size_volume);
                
        for (int i=0; i<Number_of_Devices; i++)
        {
            cudaSetDevice(i); 
            kernel_add_proj<<<dimGrid_V,dimBlock_V>>>(h_volume_weightedLen, d_volume_weightedSum_addr[i]);
            cudaDeviceSynchronize(); 
        }                 
	}        
       
        
    double InnerProduct1 = 0.0f;
    double InnerProduct2 = 0.0f;    

    for (int i=0; i<R*Z_prj*N_source; i++)
         InnerProduct1 = InnerProduct1 + (double)h_proj_single[i]*(double)h_proj_single[i]; 

    printf("The 1st inner product = %f\n",InnerProduct1);
    
	for (int i=0; i<M*N*ZETA; i++)
        InnerProduct2 = InnerProduct2 + (double)h_volume[i] * (double)h_volume_weightedLen[i];   
    
    printf("The 2nd inner product = %f\n",InnerProduct2); 
            
    cudaFreeHost(h_volume_weightedLen);
    cudaFreeHost(h_proj_single);
    
    for (int i=0; i<Number_of_Devices; i++)
    {
        cudaSetDevice(i);  
        cudaFree(d_proj_portion_addr[i]);
        cudaFree(d_proj_correction_portion_addr[i]);
        cudaFree(d_volume_addr[i]);
        cudaFree(d_volume_weightedSum_addr[i]);
        cudaFree(d_volume_length_addr[i]);        
        if (i!=Default_GPU)
            cudaDeviceReset();
	}    
    
    cudaSetDevice(Default_GPU); 
    
}
