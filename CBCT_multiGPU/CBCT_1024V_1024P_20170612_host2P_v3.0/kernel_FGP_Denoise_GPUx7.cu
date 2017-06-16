
__global__ void FGP_operator_L_Pc(float *d_f_noise, float *d_r, float *d_s, float *d_t, float *d_L, float alpha_tv)
{
    
    float temp_L; 
    float temp_mid_L; 
    
    int index_ijk = threadIdx.x + (blockIdx.x)*M + (blockIdx.y)*M*N; 
    int index_ijk_r = (threadIdx.x+1) + blockIdx.x*(M+1) + blockIdx.y*(M+1)*N;
    int index_ijk_s = threadIdx.x + (blockIdx.x+1)*M + (blockIdx.y)*M*(N+1); 
    int index_ijk_t = threadIdx.x + (blockIdx.x)*M + (blockIdx.y+1)*M*N;     
    
    // operator L 
    temp_L  = (d_r[index_ijk_r] - d_r[index_ijk_r-1]) + (d_s[index_ijk_s] - d_s[index_ijk_s - M]) +  (d_t[index_ijk_t]  - d_t[index_ijk_t-M*N]) ;       
    
    //positive constraint P_c
    temp_mid_L =  (d_f_noise[index_ijk] - alpha_tv * temp_L); 
//     if ( temp_mid_L < 0.0f )
//         d_L[index_ijk] = 0.0f;
//     else
        d_L[index_ijk] = temp_mid_L; 
    

}


__global__ void FGP_operator_LT_Pp(float *d_r, float *d_s, float *d_t, float *d_L, float *d_o_array, float *d_p_array, float *d_q_array, float t_k_1, float t_k, float alpha_12)
{
    
    int index_ijk; 
    int index_ijk_r;
    int index_ijk_s;
    int index_ijk_t; 
        
    float temp_r =0.0f; 
    float temp_s =0.0f;
    float temp_t =0.0f;
      
    float temp_rst =0.0f;   
    
    float temp_o_new = 0.0f;
    float temp_p_new = 0.0f;
    float temp_q_new = 0.0f;
    
    // (blockIdx.y  blockIdx,x, threadIdx.x) -----> (k,j,i)   
    
    index_ijk = threadIdx.x + (blockIdx.x)*M + (blockIdx.y)*M*N; 
    index_ijk_r = (threadIdx.x+1) + blockIdx.x*(M+1) + blockIdx.y*(M+1)*N;
    index_ijk_s = threadIdx.x + (blockIdx.x+1)*M + (blockIdx.y)*M*(N+1); 
    index_ijk_t = threadIdx.x + (blockIdx.x)*M + (blockIdx.y+1)*M*N; 

    temp_r = d_r[index_ijk_r]; 
    temp_s = d_s[index_ijk_s]; 
    temp_t = d_t[index_ijk_t]; 

    //__syncthreads(); 
    
    if (threadIdx.x < M-1)  // Do not update d_r at the extra layer boundary, leave it 0
        temp_r = temp_r + alpha_12*(d_L[index_ijk] - d_L[index_ijk + 1]);    
    
    if (blockIdx.x < N-1)  // Do not update d_s at the extra layer boundary, leave it 0
        temp_s = temp_s + alpha_12*(d_L[index_ijk] - d_L[index_ijk + M]) ;

    if (blockIdx.y < ZETA-1) // Do not update d_t at the extra layer boundary, leave it 0
        temp_t = temp_t + alpha_12*(d_L[index_ijk] - d_L[index_ijk + M*N]); 

    temp_rst = sqrt(temp_r * temp_r + temp_s * temp_s + temp_t * temp_t);  
   
    // Pp operator and update step together; 
    if (threadIdx.x < M-1) {
        temp_o_new = temp_r/fmax(1.0f, temp_rst) ;
        d_r[index_ijk_r] = temp_o_new + (t_k_1 - 1.0f)/t_k * (temp_o_new - d_o_array[index_ijk_r] );
        d_o_array[index_ijk_r] = temp_o_new;        
    }
    
    if (blockIdx.x < N-1) {
        temp_p_new = temp_s/fmax(1.0f, temp_rst); 
        d_s[index_ijk_s] = temp_p_new + (t_k_1- 1.0f)/t_k * (temp_p_new - d_p_array[index_ijk_s]);
        d_p_array[index_ijk_s] = temp_p_new;
    }
    
    if (blockIdx.y < ZETA-1) {
        temp_q_new = temp_t/fmax(1.0f, temp_rst); 
        d_t[index_ijk_t] = temp_q_new + (t_k_1 -1.0f)/t_k * (temp_q_new - d_q_array[index_ijk_t] );
        d_q_array[index_ijk_t] = temp_q_new; 
    }
  
}


