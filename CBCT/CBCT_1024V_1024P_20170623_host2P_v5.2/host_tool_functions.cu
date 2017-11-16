

double L2_norm_cpu(const float *a, const float *b, int SIZE)
{
        double sum=0;
        for (int i=0;i<SIZE;i++)
                sum = sum + (a[i]-b[i])*(a[i]-b[i]);
        //sum = sqrt(sum);
        return sum;
}


double L2_norm_gpu(float *d_a, float *d_b)  // h_b is pinned host memory
{
	
	double norm_result = 0.0f; 
    
    dim3  dimBlock_sub(R);
	dim3  dimGrid_sub(Z_prj,Nviews);     
	L2_norm_kernel<<<dimGrid_sub, dimBlock_sub>>>(d_a, d_b);  

    /* CPU summation */
// 	float *norm_result_temp = (float *)malloc(sizeof(float)*R*Z_prj*Nviews);  
// 	bzero(norm_result_temp, sizeof(float)*R*Z_prj*Nviews);         
// 	cudaMemcpy(norm_result_temp, d_a, d_proj_data_size, cudaMemcpyDeviceToHost);    
// 	for (int i=0;i<Z_prj*R*Nviews; i++)
// 		norm_result = norm_result + (double)norm_result_temp[i]; 
// 	free(norm_result_temp);     
// 	cout<<"    * L2 norm cpu test = "<<norm_result<<endl;
    
    /* GPU summation */
	float result_temp;    
    float *d_result = NULL;
    cudaMalloc((void**)&d_result, sizeof(float)*1);
	cudaMemset(d_result, 0, sizeof(float)*1);    
    getSum_kernel<<<1,1024,1024*sizeof(float)>>>(d_a,d_result,R*Z_prj*Nviews);      
    cudaMemcpy(&result_temp, d_result, sizeof(float)*1, cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    norm_result=(double)result_temp;
        
    return norm_result; 

}


double  TV_norm_gpu(float *d_volume)
{

	size_t size_volume = sizeof(float)*M*N*ZETA;    

	float *d_volume_tv = NULL;
    cudaMalloc((void**)&d_volume_tv, size_volume);
    cudaMemset(d_volume_tv, 0, size_volume);                
    
	dim3  dimblock_tv(M);
	dim3  dimgrid_tv(N,ZETA);

	TV_norm_kernel<<<dimgrid_tv, dimblock_tv>>>(d_volume_tv, d_volume);
        // Note: To calculate the tv matrix 
        // TV norm definition : See Amir Beck's paper: Fast Gradient-based algorithms for constrained total variation image denoising and deblurring problems   */
        
	double result = 0.0f;

//         /*CPU summation*/
//         float *volume_tv = (float *)malloc(sizeof(float)*M*N*ZETA); 
//         bzero(volume_tv, sizeof(float)*M*N*ZETA);   
//         cudaMemcpy(volume_tv, d_volume_tv, size_volume, cudaMemcpyDeviceToHost);
//         for (int i=0;i<M*N*ZETA;i++)
//             result = result + (double)volume_tv[i];
//         free(volume_tv);        
//         cout<<"    * Tv norm cpu test = "<<result<<endl;
        
        /* GPU summation */
	float *d_result = NULL;
	float result_temp;
	cudaMalloc((void**)&d_result, sizeof(float)*1);
	getSum_kernel<<<1,1024,1024*sizeof(float)>>>(d_volume_tv,d_result,M*N*ZETA);      
	cudaMemcpy(&result_temp, d_result, sizeof(float)*1, cudaMemcpyDeviceToHost);
	cudaFree(d_result);    
	result = (double)result_temp;
        
	cudaFree(d_volume_tv);
        
	return result; 
}



double  TV_norm_cpu(float *d_volume)
{
	float* d_TV = (float *)malloc(sizeof(float)*M*N*ZETA);
    
    #pragma omp parallel for
    for (int i=0; i<M; i++)
        for (int j=0; j<N; j++)
            for (int k=0; k<ZETA; k++)
    {
       int x = M*N*k+M*j+i;
       int a = M*N*k+M*j+(i+1);
       int b = M*N*k+M*(j+1)+i;
       int c = M*N*(k+1)+M*j+i;
        
        if ((i<M-1)&&(j<N-1)&&(k<ZETA-1))
            d_TV[x]=sqrt( (d_volume[x]-d_volume[a])*(d_volume[x]-d_volume[a])+(d_volume[x]-d_volume[b])*(d_volume[x]-d_volume[b])+(d_volume[x]-d_volume[c])*(d_volume[x]-d_volume[c]) );
        else if ((i==M-1)&&(j<N-1)&&(k<ZETA-1))
            d_TV[x]=sqrt( (d_volume[x]-d_volume[b])*(d_volume[x]-d_volume[b])+(d_volume[x]-d_volume[c])*(d_volume[x]-d_volume[c]) );
        else if ((i<M-1)&&(j==N-1)&&(k<ZETA-1))
            d_TV[x]=sqrt( (d_volume[x]-d_volume[a])*(d_volume[x]-d_volume[a])+(d_volume[x]-d_volume[c])*(d_volume[x]-d_volume[c]) );    
        else if ((i<M-1)&&(j<N-1)&&(k==ZETA-1))
            d_TV[x]=sqrt( (d_volume[x]-d_volume[a])*(d_volume[x]-d_volume[a])+(d_volume[x]-d_volume[b])*(d_volume[x]-d_volume[b]) );    
        else if ((i==M-1)&&(j==N-1)&&(k<ZETA-1))
            d_TV[x]=abs( d_volume[x]-d_volume[c]);
        else if ((i==M-1)&&(j<N-1)&&(k==ZETA-1))
            d_TV[x]=abs( d_volume[x]-d_volume[b]);
        else if ((i<M-1)&&(j==N-1)&&(k==ZETA-1))
            d_TV[x]=abs( d_volume[x]-d_volume[a]);       
    }            
    
    double result = 0.0f;
    for (int i=0; i<M*N*ZETA; i++)
        result+=d_TV[i];
    
    free(d_TV);
    
    return result;

}




void SaveDeviceDataToFile(float* d_data, int data_size, char filename[])
{
	float* data = (float *)malloc(sizeof(float)*data_size);
    cudaMemcpy(data,d_data,sizeof(float)*data_size,cudaMemcpyDefault);
    
    FILE *fp;
    if ( (fp = fopen(filename,"wb")) == NULL )
    {
        printf("can not open file to write data %s\n",filename);
        exit(0);
	}
	fwrite(data,sizeof(float)*data_size,1,fp);
    fclose(fp);   
    printf("Data saved in file %s \n",filename);    
}


//A function of check error, it will print out a message regarding the related information.
// Convenience function for checking CUDA runtime API results
// Can be wrapped around any runtime API call. No-op in release builds.

inline
cudaError_t checkCuda(cudaError_t result)
{
    #if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) 
    {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    #endif
    return result;
}


/*
void tv_gradient_calculate_3d_gpu_host(float *F_SART_temp, float *F_SART_gradient_tv, float epi_temp)
{
	
	//bzero(F_SART_gradient_tv,sizeof(float)*M*N*ZETA); 
	      
	// if (cudaSetDevice(2)!=cudaSuccess)
        //{
        //        std::cout<<"Error when initializing device6!"<<std::endl;
        //        exit(-1);
        //}

	float *d_volumn_f_sart = NULL;
        size_t d_volumn_f_sart_size = sizeof(float)*M*N*ZETA;
        cutilSafeCall(cudaMalloc((void**)&d_volumn_f_sart, d_volumn_f_sart_size));
        //cutilSafeCall(cudaMemset(d_volume, 0, d_volume_size) );
        cutilSafeCall(cudaMemcpy(d_volumn_f_sart, F_SART_temp, d_volumn_f_sart_size, cudaMemcpyHostToDevice) );

	float *d_volumn_f_sart_gradient_tv = NULL;
        size_t d_volumn_f_sart_gradient_tv_size = sizeof(float)*M*N*ZETA;
        cutilSafeCall(cudaMalloc((void**)&d_volumn_f_sart_gradient_tv, d_volumn_f_sart_gradient_tv_size));
        cutilSafeCall(cudaMemset(d_volumn_f_sart_gradient_tv, 0, d_volumn_f_sart_gradient_tv_size) );

	dim3  dimblock_tv_gradient(M-2);
        dim3  dimgrid_tv_gradient(N-2,ZETA-2);

	//calculate the tv matrix 
	tv_gradient_matrix_3d_kernel<<<dimgrid_tv_gradient, dimblock_tv_gradient>>>(d_volumn_f_sart_gradient_tv, d_volumn_f_sart, epi_temp);

	cutilCheckMsg("Kernel execution failed");
        cutilSafeCall(cudaMemcpy(F_SART_gradient_tv, d_volumn_f_sart_gradient_tv, d_volumn_f_sart_gradient_tv_size, cudaMemcpyDeviceToHost) );

        cutilSafeCall( cudaFree(d_volumn_f_sart));
        cutilSafeCall( cudaFree(d_volumn_f_sart_gradient_tv));
        cudaThreadExit();	
}



void backtracking_update_host(float *F_temp_update, float *F_temp, float *tv_gradient_matrix_temp, float alpha_k_temp)
{

	
	// if (cudaSetDevice(2)!=cudaSuccess)
        //{
        //        std::cout<<"Error when initializing device6!"<<std::endl;
        //        exit(-1);
        //}

	float *d_volumn_f_update = NULL;
        size_t d_volumn_f_update_size = sizeof(float)*M*N*ZETA;
        cutilSafeCall(cudaMalloc((void**)&d_volumn_f_update, d_volumn_f_update_size));
        cutilSafeCall(cudaMemcpy(d_volumn_f_update, F_temp_update, d_volumn_f_update_size, cudaMemcpyHostToDevice) );
	
	float *d_volumn_f = NULL;
        size_t d_volumn_f_size = sizeof(float)*M*N*ZETA;
        cutilSafeCall(cudaMalloc((void**)&d_volumn_f, d_volumn_f_size));
        cutilSafeCall(cudaMemcpy(d_volumn_f, F_temp, d_volumn_f_size, cudaMemcpyHostToDevice) );
	
	float *d_volumn_tv_gradient_matrix = NULL;
        size_t d_volumn_tv_gradient_matrix_size = sizeof(float)*M*N*ZETA;
        cutilSafeCall(cudaMalloc((void**)&d_volumn_tv_gradient_matrix, d_volumn_tv_gradient_matrix_size));
        cutilSafeCall(cudaMemcpy(d_volumn_tv_gradient_matrix, tv_gradient_matrix_temp, d_volumn_tv_gradient_matrix_size, cudaMemcpyHostToDevice) );
	
	dim3  dimblock_update(M);
        dim3  dimgrid_update(N,ZETA);

	//calculate the tv matrix 
	backtracking_update_kernel<<<dimgrid_update, dimblock_update>>>(d_volumn_f_update, d_volumn_f, d_volumn_tv_gradient_matrix, alpha_k_temp);

	cutilCheckMsg("Kernel execution failed");
        cutilSafeCall(cudaMemcpy(F_temp_update, d_volumn_f_update, d_volumn_f_size, cudaMemcpyDeviceToHost) );

        cutilSafeCall( cudaFree(d_volumn_f_update));
        cutilSafeCall( cudaFree(d_volumn_f));
        cutilSafeCall( cudaFree(d_volumn_tv_gradient_matrix));
        cudaThreadExit();	
}

float gradient_f_norm_host(float *tv_gradient_matrix_temp)
{

	
	// if (cudaSetDevice(2)!=cudaSuccess)
        //{
        //        std::cout<<"Error when initializing device6!"<<std::endl;
        //        exit(-1);
        //}

        float *d_volumn_f = NULL;
        size_t d_volumn_f_size = sizeof(float)*M*N*ZETA;
        cutilSafeCall(cudaMalloc((void**)&d_volumn_f, d_volumn_f_size));
        //cutilSafeCall(cudaMemset(d_volume, 0, d_volume_size) );
        cutilSafeCall(cudaMemcpy(d_volumn_f, tv_gradient_matrix_temp, d_volumn_f_size, cudaMemcpyHostToDevice) );

        float *d_volumn_df_l1 = NULL;
        size_t d_volumn_df_l1_size = sizeof(float)*N*ZETA;
        cutilSafeCall(cudaMalloc((void**)&d_volumn_df_l1, d_volumn_df_l1_size));
        cutilSafeCall(cudaMemset(d_volumn_df_l1, 0, d_volumn_df_l1_size) );

        float *d_volumn_df_l2 = NULL;
        size_t d_volumn_df_l2_size = sizeof(float)*ZETA;
        cutilSafeCall(cudaMalloc((void**)&d_volumn_df_l2, d_volumn_df_l2_size));
        cutilSafeCall(cudaMemset(d_volumn_df_l2, 0, d_volumn_df_l2_size) );

        float *d_volumn_df_sum = NULL;
        size_t d_volumn_df_sum_size = sizeof(float)*1;
        cutilSafeCall(cudaMalloc((void**)&d_volumn_df_sum, d_volumn_df_sum_size));
        cutilSafeCall(cudaMemset(d_volumn_df_sum, 0, d_volumn_df_sum_size) );

	float *norm_result_temp = (float *)malloc(sizeof(float)*1);  
	norm_result_temp[0] = 0.0f; 

	float norm_result = 0.0f; 

        dim3  dimblock_norm_l1(M,1,1);
        dim3  dimgrid_norm_l1(N,ZETA,1);

        dim3  dimblock_norm_l2(N,1,1);
        dim3  dimgrid_norm_l2(ZETA,1,1);

        dim3  dimblock_norm_sum(ZETA,1,1);
        dim3  dimgrid_norm_sum(1,1,1);

	//calculate the norm_2 
	reduce_norm_2_kernel_l1<<<dimgrid_norm_l1, dimblock_norm_l1, sizeof(float)*M>>>(d_volumn_f, d_volumn_df_l1, M*N*ZETA);
	reduce_norm_2_kernel_l2<<<dimgrid_norm_l2, dimblock_norm_l2, sizeof(float)*N>>>(d_volumn_df_l1, d_volumn_df_l2, N*ZETA);
	reduce_norm_2_kernel_end<<<dimgrid_norm_sum, dimblock_norm_sum, sizeof(float)*ZETA>>>(d_volumn_df_l2, d_volumn_df_sum, ZETA);
	
	cutilCheckMsg("Kernel execution failed");

        cutilSafeCall(cudaMemcpy(norm_result_temp, d_volumn_df_sum, d_volumn_df_sum_size, cudaMemcpyDeviceToHost) );
	norm_result = norm_result_temp[0]; 
        //printf("TV value calculation one time \n");
        //printf( "Processing time: %f (ms)\n", cutGetTimerValue(timer) );

        cutilSafeCall(cudaFree(d_volumn_f));

        cutilSafeCall( cudaFree(d_volumn_df_l1));
        cutilSafeCall( cudaFree(d_volumn_df_l2));
        cutilSafeCall( cudaFree(d_volumn_df_sum));

        cudaThreadExit();
	
	free(norm_result_temp); 
	return norm_result; 

}


void line_search_host(float *F_SART_temp, float *F_SART_TV_temp, float epi_temp)
{
	
	// if (cudaSetDevice(2)!=cudaSuccess)
        //{
        //        std::cout<<"Error when initializing device6!"<<std::endl;
        //        exit(-1);
        //}
	
	unsigned int timer = 0;
        cutilCheckError( cutCreateTimer( &timer));
        cutilCheckError( cutStartTimer( timer));

	float alpha_k = 1.0f;
        float rho = 0.5f;
        float c = 0.0001f;

	//float epi_temp = 1.0e-8f;
        float c_alpha_f_pk = 0.0f;
        float f_pk = 0.0f;

        //copy the f_sart to f_sart_tv
        //then start to do the line search 
	memcpy(F_SART_TV_temp, F_SART_temp, sizeof(float)*M*N*ZETA);

	float *tv_gradient_matrix =(float *)malloc(sizeof(float)*M*N*ZETA);
	bzero(tv_gradient_matrix,sizeof(float)*M*N*ZETA);

	
	//calculate the tv gradient matrix 
	tv_gradient_calculate_3d_gpu_host(F_SART_temp, tv_gradient_matrix, epi_temp);

	float tv_value_old;
	float tv_value_new;

	tv_value_old = tv_value_calculate_3d_gpu_host(F_SART_temp);
	backtracking_update_host(F_SART_TV_temp, F_SART_TV_temp, tv_gradient_matrix, alpha_k);		
	tv_value_new = tv_value_calculate_3d_gpu_host(F_SART_TV_temp);

	f_pk =-gradient_f_norm_host(tv_gradient_matrix); 

	c_alpha_f_pk = c*alpha_k*f_pk;

        while (tv_value_new > (tv_value_old + c_alpha_f_pk) )
        {
                alpha_k = alpha_k *rho;
                c_alpha_f_pk = c*alpha_k*f_pk;
		backtracking_update_host(F_SART_TV_temp, F_SART_temp, tv_gradient_matrix, alpha_k);	
                //for(int i=0;i<f_size;i++)
                //        f_sart_tv[i] = f_sart[i] - alpha_k*tv_gradient_matrix[i];
		tv_value_new = tv_value_calculate_3d_gpu_host(F_SART_TV_temp);
        }
	
        cutilCheckError(cutStopTimer(timer));
        printf("Line search one time \n");
        printf( "Processing time: %f (ms)\n", cutGetTimerValue(timer) );

	free(tv_gradient_matrix); 	
}

*/