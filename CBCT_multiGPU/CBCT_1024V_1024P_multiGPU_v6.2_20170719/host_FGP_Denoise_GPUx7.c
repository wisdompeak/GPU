__host__ void FGP_denoise_GPUx7_exact(float *h_f_noise, float lambda, int Niter_denoise)

{
	size_t d_volume_size = sizeof(float)*NO_VOXEL;
    
    // r 
	float *d_r = NULL;
	size_t d_r_array_size = sizeof(float)*(NO_Z)*(NO_Y)*(NO_X+1);
    (cudaMalloc((void**)&d_r, d_r_array_size));
    (cudaMemset(d_r, 0, d_r_array_size) );
	
    // s
	float *d_s = NULL;
	size_t d_s_array_size = sizeof(float)*(NO_Z)*(NO_Y+1)*(NO_X);
    (cudaMalloc((void**)&d_s, d_s_array_size));
    (cudaMemset(d_s, 0, d_s_array_size) );
    
    // t 
	float *d_t = NULL;
	size_t d_t_array_size = sizeof(float)*(NO_Z+1)*(NO_Y)*(NO_X);
    (cudaMalloc((void**)&d_t, d_t_array_size));
    (cudaMemset(d_t, 0, d_t_array_size) );
    
    // o 
	float *d_o = NULL;
	size_t d_o_array_size = sizeof(float)*(NO_Z)*(NO_Y)*(NO_X+1);
    (cudaMalloc((void**)&d_o, d_o_array_size));
    (cudaMemset(d_o, 0, d_o_array_size) );
	
    // p
	float *d_p = NULL;
	size_t d_p_array_size = sizeof(float)*(NO_Z)*(NO_Y+1)*(NO_X);
    (cudaMalloc((void**)&d_p, d_p_array_size));
    (cudaMemset(d_p, 0, d_p_array_size) );
    
    // q 
	float *d_q = NULL;
	size_t d_q_array_size = sizeof(float)*(NO_Z+1)*(NO_Y)*(NO_X);
    (cudaMalloc((void**)&d_q, d_q_array_size));
    (cudaMemset(d_q, 0, d_q_array_size) );
	
    // L
    float *d_L = NULL;
	size_t d_L_array_size = sizeof(float)*(NO_Z)*(NO_Y)*(NO_X);
    (cudaMalloc((void**)&d_L, d_L_array_size));
    (cudaMemset(d_L, 0, d_L_array_size) );
    
    // setup execution parameters for backprojection 
	dim3  dimblock_fgp(NO_X,1,1); 
	dim3  dimgrid_fgp(NO_Y,NO_Z,1);     

	// time the kernels 
	cudaEvent_t start_event, stop_event; 
	float elapsed_time = 0.0f; 
  
	cudaEventCreate(&start_event); 
 	cudaEventCreate(&stop_event);  
	cudaEventRecord(start_event,0);   
    
    float *d_f_noise = NULL;
	cudaMalloc((void**)&d_f_noise, d_volume_size);    
    cudaMemcpy(d_f_noise, h_f_noise, d_volume_size, cudaMemcpyHostToDevice);          

    float t_k_1 = 1.0f;
    float t_k; 
    
    
    for (int i= 0; i<Niter_denoise; i++)
    {
                 
        t_k = (1.0f + sqrt( 1.0f + 4.0f*t_k_1*t_k_1))/2.0f; 

        FGP_operator_L_Pc<<<dimgrid_fgp,dimblock_fgp>>>(d_f_noise, d_r, d_s, d_t, d_L, lambda ); 
        // d_L = Pc[ d_f_noise - labmda*L(d_r,d_s,d_t)]
        
        FGP_operator_LT_Pp<<<dimgrid_fgp,dimblock_fgp>>>(d_r, d_s, d_t, d_L, d_o, d_p, d_q, t_k_1, t_k, 1.0f/(lambda*12.0f) ); 
        // (d_o, d_p, d_q) = Pp[ (r,s,t) + 1/(12lambda)*L'(d_L) ]
        
        t_k_1 = t_k;  
    }

     //set x* = P_c[x_g - \alpha L(o,p,q)]
    FGP_operator_L_Pc<<<dimgrid_fgp,dimblock_fgp>>>(d_f_noise, d_o, d_p, d_q, d_L, lambda ); 

    
    cudaMemcpy(h_f_noise,d_L,d_volume_size,cudaMemcpyDefault);
    
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
	
    cudaFree(d_o);
	cudaFree(d_p);
	cudaFree(d_q);
	
    cudaFree(d_L);
    cudaFree(d_f_noise);

	    
}
