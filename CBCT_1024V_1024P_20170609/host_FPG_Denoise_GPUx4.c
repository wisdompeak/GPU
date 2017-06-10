////*******************************************/////
void FGP_denoise_GPUx4_exact(float *d_A, float lambda, int Niter_denoise)
{
    int SIZE = M*N*ZETA;             
    
    float *r_k = new float [SIZE];
	bzero(r_k, sizeof(float)*SIZE);
    float *s_k = new float [SIZE];
	bzero(s_k, sizeof(float)*SIZE);
    float *t_k = new float [SIZE];
	bzero(t_k, sizeof(float)*SIZE);    
        
    float *p_k = new float [SIZE];
	bzero(p_k, sizeof(float)*SIZE);
    float *q_k = new float [SIZE];
	bzero(q_k, sizeof(float)*SIZE);
    float *u_k = new float [SIZE];
	bzero(u_k, sizeof(float)*SIZE);
    
    float *p_k1 = new float [SIZE];
	bzero(p_k1, sizeof(float)*SIZE);
    float *q_k1 = new float [SIZE];
	bzero(q_k1, sizeof(float)*SIZE);
    float *u_k1 = new float [SIZE];
	bzero(u_k1, sizeof(float)*SIZE);    
    
    float *B = new float [SIZE];
	bzero(B, sizeof(float)*SIZE);            
    
    float T_k=1.0f; 
    float T_k1=1.0f;
    float coeff;
    
	size_t d_volume_size = sizeof(float)*SIZE;
         	
	float *d_r = NULL;
	cudaMalloc((void**)&d_r, d_volume_size);
    cudaMemset(d_r, 0, d_volume_size);
    
	float *d_s = NULL;
	cudaMalloc((void**)&d_s, d_volume_size);
    cudaMemset(d_s, 0, d_volume_size); 
    
	float *d_t = NULL;
	cudaMalloc((void**)&d_t, d_volume_size);
    cudaMemset(d_t, 0, d_volume_size); 
    
	float *d_volume = NULL;
	cudaMalloc((void**)&d_volume, d_volume_size);
    cudaMemset(d_volume, 0, d_volume_size); 
    
	dim3  dimblock_B(M);
	dim3  dimgrid_B(N,ZETA);        
    
    
	// time the kernels 
	cudaEvent_t start_event, stop_event; 
	float elapsed_time = 0.0f;   
	cudaEventCreate(&start_event); 
 	cudaEventCreate(&stop_event);  
    
    // Record the start time point
	cudaEventRecord(start_event,0);   

    
    for (int k=1; k<=Niter_denoise; k++)
    {

//         printf(" - have done FGP denoise %d out of %d times: \n",k,Niter);   
        
        memcpy(p_k1,p_k,sizeof(float)*SIZE);
        memcpy(q_k1,q_k,sizeof(float)*SIZE);
        memcpy(u_k1,u_k,sizeof(float)*SIZE);
        
        T_k=T_k1;           // T_k <- T_{k+1}                        
        T_k1=(1.0f+sqrt(1.0f+4.0f*T_k*T_k))/2.0f;  // update T_{k+1}
        coeff = (T_k-1)/(T_k1);
    
        cudaMemcpy(d_volume, d_A, d_volume_size, cudaMemcpyDeviceToDevice);
        
        F_operator_L_Pc_kernel<<<dimgrid_B, dimblock_B>>>(d_volume, d_r, d_s, d_t,lambda);  
        
        F_operator_LT_kernel<<<dimgrid_B, dimblock_B>>>(d_volume, d_r, d_s, d_t, 1.0f/(12.0f*lambda));      
       
        /****update p_k, q_k, u_k*****/
        F_Operator_Pp_kernel_p<<<dimgrid_B, dimblock_B>>>(d_volume, d_r, d_s, d_t);
        cudaMemcpy(p_k, d_volume, d_volume_size, cudaMemcpyDeviceToHost);  
    
        F_Operator_Pp_kernel_q<<<dimgrid_B, dimblock_B>>>(d_volume, d_r, d_s, d_t);
        cudaMemcpy(q_k, d_volume, d_volume_size, cudaMemcpyDeviceToHost);
    
        F_Operator_Pp_kernel_u<<<dimgrid_B, dimblock_B>>>(d_volume, d_r, d_s, d_t);
        cudaMemcpy(u_k, d_volume, d_volume_size, cudaMemcpyDeviceToHost);  
        
        /****update r_k, s_k, t_k*****/
        cudaMemcpy(d_volume, p_k1, d_volume_size, cudaMemcpyHostToDevice);     
        F_Operator_Pp_kernel_r<<<dimgrid_B, dimblock_B>>>(d_volume, d_r, d_s, d_t, coeff);
        cudaMemcpy(r_k, d_volume, d_volume_size, cudaMemcpyDeviceToHost);    
    
        (cudaMemcpy(d_volume, q_k1, d_volume_size, cudaMemcpyHostToDevice));         
        F_Operator_Pp_kernel_s<<<dimgrid_B, dimblock_B>>>(d_volume, d_r, d_s, d_t, coeff);
        cudaMemcpy(s_k, d_volume, d_volume_size, cudaMemcpyDeviceToHost);    
    
        cudaMemcpy(d_volume, u_k1, d_volume_size, cudaMemcpyHostToDevice);             
        F_Operator_Pp_kernel_t<<<dimgrid_B, dimblock_B>>>(d_volume, d_r, d_s, d_t, coeff);
        cudaMemcpy(t_k, d_volume, d_volume_size, cudaMemcpyDeviceToHost);    
        
        cudaMemcpy(d_r, r_k, d_volume_size, cudaMemcpyHostToDevice);    
        cudaMemcpy(d_s, s_k, d_volume_size, cudaMemcpyHostToDevice);    
        cudaMemcpy(d_t, t_k, d_volume_size, cudaMemcpyHostToDevice);    
                       
    }
    
    cudaMemcpy(d_r, p_k, d_volume_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, q_k, d_volume_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_t, u_k, d_volume_size, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_volume, d_A, d_volume_size, cudaMemcpyDeviceToDevice);    
    F_operator_L_Pc_kernel<<<dimgrid_B, dimblock_B>>>(d_volume, d_r, d_s, d_t,lambda);            
    cudaMemcpy(d_A, d_volume, d_volume_size, cudaMemcpyDeviceToDevice) ;    
    
    // Record the end time point
    cudaEventRecord(stop_event, 0);  
  	cudaEventSynchronize(stop_event);      
	cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
 	printf(" - TV denoise (FGP algorithm) %d times,  elapsed time = %.10f (ms) \n", Niter_denoise, elapsed_time);
	
    //Destroy the event
	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);
        
	cudaFree(d_r);
	cudaFree(d_s);
	cudaFree(d_t);
	cudaFree(d_volume);    
    
    delete []r_k;
    delete []s_k;
    delete []t_k;
    delete []p_k;
    delete []q_k;
    delete []u_k;
    delete []p_k1;
    delete []q_k1;
    delete []u_k1;     
    delete []B;
        
}



 ////*******************************************/////

void FGP_denoise_GPUx4_apprx(float *d_A, float lambda, int Niter_denoise)
{
    int SIZE = M*N*ZETA;               
        
    float *p_k = new float [SIZE];
	bzero(p_k, sizeof(float)*SIZE);
    float *q_k = new float [SIZE];
	bzero(q_k, sizeof(float)*SIZE);
    float *u_k = new float [SIZE];
	bzero(u_k, sizeof(float)*SIZE);
    
    float *p_k1 = new float [SIZE];
	bzero(p_k1, sizeof(float)*SIZE);
    float *q_k1 = new float [SIZE];
	bzero(q_k1, sizeof(float)*SIZE);
    float *u_k1 = new float [SIZE];
	bzero(u_k1, sizeof(float)*SIZE);    
    
    float *B = new float [SIZE];
	bzero(B, sizeof(float)*SIZE);            
    
    float T_k=1.0f; 
    float T_k1=1.0f;
    float coeff;
    
	size_t d_volume_size = sizeof(float)*SIZE;
         	
	float *d_r = NULL;
	cudaMalloc((void**)&d_r, d_volume_size);
    cudaMemset(d_r, 0, d_volume_size);
    
	float *d_s = NULL;
	cudaMalloc((void**)&d_s, d_volume_size);
    cudaMemset(d_s, 0, d_volume_size); 
    
	float *d_t = NULL;
	cudaMalloc((void**)&d_t, d_volume_size);
    cudaMemset(d_t, 0, d_volume_size); 
    
	float *d_volume = NULL;
	cudaMalloc((void**)&d_volume, d_volume_size);
    cudaMemset(d_volume, 0, d_volume_size); 
    
	dim3  dimblock_B(M);
	dim3  dimgrid_B(N,ZETA);        
    
    
	// time the kernels 
	cudaEvent_t start_event, stop_event; 
	float elapsed_time = 0.0f;   
	cudaEventCreate(&start_event); 
 	cudaEventCreate(&stop_event);  
    
    // Record the start time point
	cudaEventRecord(start_event,0);   

    
    for (int k=1; k<=Niter_denoise; k++)
    {
//         printf(" - have done FGP denoise %d out of %d times: \n",k,Niter);   
        
        memcpy(p_k1,p_k,sizeof(float)*SIZE);
        memcpy(q_k1,q_k,sizeof(float)*SIZE);
        memcpy(u_k1,u_k,sizeof(float)*SIZE);
        
        T_k=T_k1;           // T_k <- T_{k+1}                        
        T_k1=(1.0f+sqrt(1.0f+4.0f*T_k*T_k))/2.0f;  // update T_{k+1}
        coeff = (T_k-1)/(T_k1);
    
        cudaMemcpy(d_volume, d_A, d_volume_size, cudaMemcpyDeviceToDevice);
        
        F_operator_L_Pc_kernel<<<dimgrid_B, dimblock_B>>>(d_volume, d_r, d_s, d_t,lambda);  
        
        F_operator_LT_kernel<<<dimgrid_B, dimblock_B>>>(d_volume, d_r, d_s, d_t, 1.0f/(12.0f*lambda));      
       
        /****update p_k, q_k, u_k*****/
        F_Operator_Pp_kernel_p<<<dimgrid_B, dimblock_B>>>(d_volume, d_r, d_s, d_t);
        cudaMemcpy(p_k, d_volume, d_volume_size, cudaMemcpyDeviceToHost);  
    
        F_Operator_Pp_kernel_q<<<dimgrid_B, dimblock_B>>>(d_volume, d_r, d_s, d_t);
        cudaMemcpy(q_k, d_volume, d_volume_size, cudaMemcpyDeviceToHost);
    
        F_Operator_Pp_kernel_u<<<dimgrid_B, dimblock_B>>>(d_volume, d_r, d_s, d_t);
        cudaMemcpy(u_k, d_volume, d_volume_size, cudaMemcpyDeviceToHost);  
        
        /****update r_k, s_k, t_k*****/
        cudaMemcpy(d_volume, p_k1, d_volume_size, cudaMemcpyHostToDevice);     
        F_Operator_Pp_kernel_r<<<dimgrid_B, dimblock_B>>>(d_volume, d_r, d_s, d_t, coeff);
        cudaMemcpy(d_r, d_volume, d_volume_size, cudaMemcpyDeviceToDevice);    
    
        (cudaMemcpy(d_volume, q_k1, d_volume_size, cudaMemcpyHostToDevice));         
        F_Operator_Pp_kernel_s<<<dimgrid_B, dimblock_B>>>(d_volume, d_r, d_s, d_t, coeff);
        cudaMemcpy(d_s, d_volume, d_volume_size, cudaMemcpyDeviceToDevice);    
    
        cudaMemcpy(d_volume, u_k1, d_volume_size, cudaMemcpyHostToDevice);             
        F_Operator_Pp_kernel_t<<<dimgrid_B, dimblock_B>>>(d_volume, d_r, d_s, d_t, coeff);
        cudaMemcpy(d_t, d_volume, d_volume_size, cudaMemcpyDeviceToDevice);    
        
                       
    }
    
    cudaMemcpy(d_r, p_k, d_volume_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, q_k, d_volume_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_t, u_k, d_volume_size, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_volume, d_A, d_volume_size, cudaMemcpyDeviceToDevice);    
    F_operator_L_Pc_kernel<<<dimgrid_B, dimblock_B>>>(d_volume, d_r, d_s, d_t,lambda);            
    cudaMemcpy(d_A, d_volume, d_volume_size, cudaMemcpyDeviceToDevice) ;    
    
    // Record the end time point
    cudaEventRecord(stop_event, 0);  
  	cudaEventSynchronize(stop_event);      
	cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
 	printf(" - TV denoise (FGP algorithm) %d times,  elapsed time = %.10f (ms) \n", Niter_denoise, elapsed_time);
	
    //Destroy the event
	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);
        
	cudaFree(d_r);
	cudaFree(d_s);
	cudaFree(d_t);
	cudaFree(d_volume);    
    
    delete []p_k;
    delete []q_k;
    delete []u_k;
    delete []p_k1;
    delete []q_k1;
    delete []u_k1;     
    delete []B;
        
}






 ////*******************************************/////

void GP_denoise_GPUx4_fast(float *d_A, float lambda, int Niter_denoise)
{
    int SIZE = M*N*ZETA;                       
    
	size_t d_volume_size = sizeof(float)*SIZE;
         	
	float *d_r = NULL;
	cudaMalloc((void**)&d_r, d_volume_size);
    cudaMemset(d_r, 0, d_volume_size);
    
	float *d_s = NULL;
	cudaMalloc((void**)&d_s, d_volume_size);
    cudaMemset(d_s, 0, d_volume_size); 
    
	float *d_t = NULL;
	cudaMalloc((void**)&d_t, d_volume_size);
    cudaMemset(d_t, 0, d_volume_size); 
    
	float *d_volume = NULL;
	cudaMalloc((void**)&d_volume, d_volume_size);
    cudaMemset(d_volume, 0, d_volume_size); 
    
	dim3  dimblock_B(M);
	dim3  dimgrid_B(N,ZETA);        
    
    
	// time the kernels 
	cudaEvent_t start_event, stop_event; 
	float elapsed_time = 0.0f;   
	cudaEventCreate(&start_event); 
 	cudaEventCreate(&stop_event);  
    
    // Record the start time point
	cudaEventRecord(start_event,0);   

    
    for (int k=1; k<=Niter_denoise; k++)
    {
    
        cudaMemcpy(d_volume, d_A, d_volume_size, cudaMemcpyDeviceToDevice);
        
        F_operator_L_Pc_kernel<<<dimgrid_B, dimblock_B>>>(d_volume, d_r, d_s, d_t,lambda);  
        
        F_operator_LT_kernel<<<dimgrid_B, dimblock_B>>>(d_volume, d_r, d_s, d_t, 1.0f/(12.0f*lambda));      

        F_Operator_Pp_kernel_p<<<dimgrid_B, dimblock_B>>>(d_volume, d_r, d_s, d_t);
        cudaMemcpy(d_r, d_volume, d_volume_size, cudaMemcpyDeviceToDevice);    
        
        F_Operator_Pp_kernel_q<<<dimgrid_B, dimblock_B>>>(d_volume, d_r, d_s, d_t);
        cudaMemcpy(d_s, d_volume, d_volume_size, cudaMemcpyDeviceToDevice);    

        F_Operator_Pp_kernel_u<<<dimgrid_B, dimblock_B>>>(d_volume, d_r, d_s, d_t);
        cudaMemcpy(d_t, d_volume, d_volume_size, cudaMemcpyDeviceToDevice);    

    }
    
    cudaMemcpy(d_volume, d_A, d_volume_size, cudaMemcpyDeviceToDevice);    
    F_operator_L_Pc_kernel<<<dimgrid_B, dimblock_B>>>(d_volume, d_r, d_s, d_t,lambda);            
    cudaMemcpy(d_A, d_volume, d_volume_size, cudaMemcpyDeviceToDevice) ;    
    
    // Record the end time point
    cudaEventRecord(stop_event, 0);  
  	cudaEventSynchronize(stop_event);      
	cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
 	printf(" - TV denoise (FGP algorithm) %d times,  elapsed time = %.10f (ms) \n", Niter_denoise, elapsed_time);
	
    //Destroy the event
	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);
        
	cudaFree(d_r);
	cudaFree(d_s);
	cudaFree(d_t);
	cudaFree(d_volume);    
    
        
}



