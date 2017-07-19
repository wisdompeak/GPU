__global__ void F_operator_L_Pc_kernel(float *d_B, float *d_R, float *d_S, float *d_T, float lambda)
{
    
    /* Note
     * dim3  dimGrid_backprj(N,ZETA);  =>
     *  blockIdx.x ranges 0~(N-1), index of pixels along y axis
     *  blockIdx.y ranges 0~(ZETA-1), index of pixels along z axis
     * dim3  dimBlock_backprj(M);  =>
     *  threadIdx.x ranges 0~(M-1), index of pixels along x axis
     */    
	
	unsigned int index_A = blockIdx.y* blockDim.x*gridDim.x + blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int index_r = blockIdx.y* blockDim.x*gridDim.x + blockIdx.x*blockDim.x + (threadIdx.x-1);
    unsigned int index_s = blockIdx.y* blockDim.x*gridDim.x + (blockIdx.x-1)*blockDim.x + threadIdx.x;
    unsigned int index_t = (blockIdx.y-1)* blockDim.x*gridDim.x + blockIdx.x*blockDim.x + threadIdx.x;

    float A1,A2,A3,B1,B2,B3,temp;
    
    if (threadIdx.x == M-1) 
        A1 = 0.0f;
    else 
        A1 = d_R[index_A];

    if (threadIdx.x == 0) 
        B1 = 0.0f;
    else 
        B1 = d_R[index_r];
    
    if (blockIdx.x == N-1) 
        A2 = 0.0f;
    else 
        A2 = d_S[index_A];
    
    if (blockIdx.x == 0) 
        B2 = 0.0f;
    else 
        B2 = d_S[index_s];
    
    if (blockIdx.y==ZETA-1)
        A3 = 0.0f;
    else
        A3 = d_T[index_A];         

    if (blockIdx.y==0)
        B3 = 0.0f;
    else
        B3 = d_T[index_t];
                        
    temp = A1-B1 + A2-B2 + A3-B3; 
        
//     if ((d_B[index_A]-lambda*temp[index_A])<0)
//         d_B[index_A]=0;
//     else
        d_B[index_A]=d_B[index_A]-lambda*temp;    
        
    __syncthreads();   
        
}


__global__ void F_operator_LT_kernel(float *d_A, float *d_r, float *d_s, float *d_t, float coeff)
{  
	
	unsigned int i = blockIdx.y* blockDim.x*gridDim.x + blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y* blockDim.x*gridDim.x + blockIdx.x*blockDim.x + (threadIdx.x+1);
    unsigned int k = blockIdx.y* blockDim.x*gridDim.x + (blockIdx.x+1)*blockDim.x + threadIdx.x;
    unsigned int l = (blockIdx.y+1)* blockDim.x*gridDim.x + blockIdx.x*blockDim.x + threadIdx.x;
        
    if (threadIdx.x<M-1)
        d_r[i]+= coeff*(d_A[i]-d_A[j]);
    
    if (blockIdx.x<N-1)
        d_s[i]+= coeff*(d_A[i]-d_A[k]);
    
    if (blockIdx.y<ZETA-1)
        d_t[i]+= coeff*(d_A[i]-d_A[l]);        
    
}


__global__ void F_Operator_Pp_kernel_p(float *d_P, float *d_r, float *d_s, float *d_t)
{  
	
	unsigned int i = blockIdx.y* blockDim.x*gridDim.x + blockIdx.x*blockDim.x + threadIdx.x;
    float temp;
        
	if ((threadIdx.x<M-1)&&(blockIdx.x<N-1)&&(blockIdx.y<ZETA-1))
        temp=d_r[i]/fmax(1,sqrt(d_r[i]*d_r[i]+d_s[i]*d_s[i]+d_t[i]*d_t[i]));
    else if ((threadIdx.x<M-1)&&(blockIdx.x==N-1)&&(blockIdx.y<ZETA-1))
        temp=d_r[i]/fmax(1,sqrt(d_r[i]*d_r[i]+d_t[i]*d_t[i]));    
    else if ((threadIdx.x<M-1)&&(blockIdx.x<N-1)&&(blockIdx.y==ZETA-1))
        temp=d_r[i]/fmax(1,sqrt(d_r[i]*d_r[i]+d_s[i]*d_s[i]));
    else if (threadIdx.x<M-1)
        temp=d_r[i]/fmax(1,abs(d_r[i]));
    
    d_P[i] = temp;
}

__global__ void F_Operator_Pp_kernel_q(float *d_Q, float *d_r, float *d_s, float *d_t)
{  
	
	unsigned int i = blockIdx.y* blockDim.x*gridDim.x + blockIdx.x*blockDim.x + threadIdx.x;
    float temp;
        
	if ((threadIdx.x<M-1)&&(blockIdx.x<N-1)&&(blockIdx.y<ZETA-1))
        temp=d_s[i]/fmax(1,sqrt(d_r[i]*d_r[i]+d_s[i]*d_s[i]+d_t[i]*d_t[i]));
    else if ((threadIdx.x==M-1)&&(blockIdx.x<N-1)&&(blockIdx.y<ZETA-1))
        temp=d_s[i]/fmax(1,sqrt(d_s[i]*d_s[i]+d_t[i]*d_t[i]));    
    else if ((threadIdx.x<M-1)&&(blockIdx.x<N-1)&&(blockIdx.y==ZETA-1))
        temp=d_s[i]/fmax(1,sqrt(d_s[i]*d_s[i]+d_r[i]*d_r[i]));
    else if (blockIdx.x<N-1)
        temp=d_s[i]/fmax(1,abs(d_s[i]));  
    
    d_Q[i] = temp;
}

__global__ void F_Operator_Pp_kernel_u(float *d_U, float *d_r, float *d_s, float *d_t)
{  
	
	unsigned int i = blockIdx.y* blockDim.x*gridDim.x + blockIdx.x*blockDim.x + threadIdx.x;
    float temp;
        
	if ((threadIdx.x<M-1)&&(blockIdx.x<N-1)&&(blockIdx.y<ZETA-1))
        temp=d_t[i]/fmax(1,sqrt(d_r[i]*d_r[i]+d_s[i]*d_s[i]+d_t[i]*d_t[i]));
    else if ((threadIdx.x==M-1)&&(blockIdx.x<N-1)&&(blockIdx.y<ZETA-1))
        temp=d_t[i]/fmax(1,sqrt(d_t[i]*d_t[i]+d_s[i]*d_s[i]));    
    else if ((threadIdx.x<M-1)&&(blockIdx.x==N-1)&&(blockIdx.y<ZETA-1))
        temp=d_t[i]/fmax(1,sqrt(d_t[i]*d_t[i]+d_r[i]*d_r[i]));
    else if (blockIdx.y<ZETA-1)
        temp=d_t[i]/fmax(1,abs(d_t[i]));     
    
    d_U[i] = temp;
    
}


__global__ void F_Operator_Pp_kernel_r(float *d_P, float *d_r, float *d_s, float *d_t, float coeff)
{  
	
	unsigned int i = blockIdx.y* blockDim.x*gridDim.x + blockIdx.x*blockDim.x + threadIdx.x;
    float temp;
        
	if ((threadIdx.x<M-1)&&(blockIdx.x<N-1)&&(blockIdx.y<ZETA-1))
        temp=d_r[i]/fmax(1,sqrt(d_r[i]*d_r[i]+d_s[i]*d_s[i]+d_t[i]*d_t[i]));
    else if ((threadIdx.x<M-1)&&(blockIdx.x==N-1)&&(blockIdx.y<ZETA-1))
        temp=d_r[i]/fmax(1,sqrt(d_r[i]*d_r[i]+d_t[i]*d_t[i]));    
    else if ((threadIdx.x<M-1)&&(blockIdx.x<N-1)&&(blockIdx.y==ZETA-1))
        temp=d_r[i]/fmax(1,sqrt(d_r[i]*d_r[i]+d_s[i]*d_s[i]));
    else if (threadIdx.x<M-1)
        temp=d_r[i]/fmax(1,abs(d_r[i]));
    
    d_P[i] = temp + coeff * (temp - d_P[i]);
}

__global__ void F_Operator_Pp_kernel_s(float *d_Q, float *d_r, float *d_s, float *d_t, float coeff)
{  
	
	unsigned int i = blockIdx.y* blockDim.x*gridDim.x + blockIdx.x*blockDim.x + threadIdx.x;
    float temp;
        
	if ((threadIdx.x<M-1)&&(blockIdx.x<N-1)&&(blockIdx.y<ZETA-1))
        temp=d_s[i]/fmax(1,sqrt(d_r[i]*d_r[i]+d_s[i]*d_s[i]+d_t[i]*d_t[i]));
    else if ((threadIdx.x==M-1)&&(blockIdx.x<N-1)&&(blockIdx.y<ZETA-1))
        temp=d_s[i]/fmax(1,sqrt(d_s[i]*d_s[i]+d_t[i]*d_t[i]));    
    else if ((threadIdx.x<M-1)&&(blockIdx.x<N-1)&&(blockIdx.y==ZETA-1))
        temp=d_s[i]/fmax(1,sqrt(d_s[i]*d_s[i]+d_r[i]*d_r[i]));
    else if (blockIdx.x<N-1)
        temp=d_s[i]/fmax(1,abs(d_s[i]));  
    
    d_Q[i] = temp + coeff * (temp - d_Q[i]);
}

__global__ void F_Operator_Pp_kernel_t(float *d_U, float *d_r, float *d_s, float *d_t, float coeff)
{  
	
	unsigned int i = blockIdx.y* blockDim.x*gridDim.x + blockIdx.x*blockDim.x + threadIdx.x;
    float temp;
        
	if ((threadIdx.x<M-1)&&(blockIdx.x<N-1)&&(blockIdx.y<ZETA-1))
        temp=d_t[i]/fmax(1,sqrt(d_r[i]*d_r[i]+d_s[i]*d_s[i]+d_t[i]*d_t[i]));
    else if ((threadIdx.x==M-1)&&(blockIdx.x<N-1)&&(blockIdx.y<ZETA-1))
        temp=d_t[i]/fmax(1,sqrt(d_t[i]*d_t[i]+d_s[i]*d_s[i]));    
    else if ((threadIdx.x<M-1)&&(blockIdx.x==N-1)&&(blockIdx.y<ZETA-1))
        temp=d_t[i]/fmax(1,sqrt(d_t[i]*d_t[i]+d_r[i]*d_r[i]));
    else if (blockIdx.y<ZETA-1)
        temp=d_t[i]/fmax(1,abs(d_t[i]));     
    
    d_U[i] = temp + coeff * (temp - d_U[i]);
    
}


/*****************************/

