void matgen(Matrix A)
{
    int n = A.width;
    int m = A.height;

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int k= i*n + j;
//             A.elements[k] = (float)rand() / RAND_MAX + (float)rand() / (RAND_MAX * RAND_MAX);
            A.elements[k] = (float)rand() / RAND_MAX ;         
//             A.elements[k] = rand() % 10;

        }
    }
}

void MatMul_GPU(const Matrix A, const Matrix B, Matrix &C)
{
    
    Matrix d_A;
    d_A.width = A.width;
    d_A.height = A.height;
    size_t mat_size = sizeof (float)*A.width*A.height;
    cudaMalloc(&d_A.elements, mat_size);
    cudaMemcpy(d_A.elements, A.elements, mat_size, cudaMemcpyHostToDevice);
    
    Matrix d_B;
    d_B.width = B.width;
    d_B.height = B.height;
    mat_size = sizeof (float)*B.width*B.height;
    cudaMalloc(&d_B.elements, mat_size);
    cudaMemcpy(d_B.elements, B.elements, mat_size, cudaMemcpyHostToDevice);    
    
    C.height=A.height;
    C.width=B.width;
    C.elements = new float [C.height*C.width];
            
    Matrix d_C;
    d_C.width = B.width;
    d_C.height = B.height;
    mat_size = sizeof (float)*C.width*C.height;
    cudaMalloc(&d_C.elements, mat_size);
    cudaMemset(d_C.elements, 1.0f, mat_size);

    clock_t* d_time;
    cudaMalloc((void**)&d_time, sizeof(clock_t)* BLOCK_NUM * 2);    
    
    cudaEvent_t start,stop;
    cudaEventCreate(&start);   
    cudaEventCreate(&stop);       
    cudaEventRecord(start,0);    //start GPU timing

    dim3 dimBlock(THREAD_NUM,1);
    dim3 dimGrid(N/THREAD_NUM, N);    
    kernel_MatMul_GPU<< <dimGrid,dimBlock>> >(d_A,d_B,d_C,d_time);
    
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);   //end GPU timing
    cudaEventSynchronize(stop); // blocks CPU execution until the specified event is recorded.
    float time_elapsed;
    cudaEventElapsedTime(&time_elapsed, start, stop);  //  this value has a resolution of approximately one half microsecond.
    cudaEventDestroy(start);
    cudaEventDestroy(stop);  
    printf("GPU kernel Timing (cudaEventRecord): %f (ms)\n", time_elapsed );    
    
    cudaMemcpy(C.elements, d_C.elements, mat_size, cudaMemcpyDeviceToHost);

    clock_t time_use[BLOCK_NUM * 2];
    cudaMemcpy(&time_use, d_time, sizeof(clock_t)* BLOCK_NUM * 2, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
    cudaFree(d_time);
    

    
   
    //采取新的计时策略 把每个 block 最早的开始时间，和最晚的结束时间相减，取得总运行时间
    clock_t min_start, max_end;

    min_start = time_use[0];
    max_end = time_use[BLOCK_NUM];

    for (int i = 1; i < BLOCK_NUM; i++) {
        if (time_use[i] < min_start)
            min_start = time_use[i];
        if (time_use[i + BLOCK_NUM] > max_end )
            max_end = time_use[i + BLOCK_NUM];
    }
    
    printf("GPU kernel d_time elapsed : %f ms\n", (max_end - min_start)*1.0f/ClockRate);
    printf("GPU block 1 d_time elapsed : %f ms\n", (time_use[0+BLOCK_NUM] - time_use[0])*1.0f/ClockRate);
    printf("GPU block 2 d_time elapsed : %f ms\n", (time_use[1+BLOCK_NUM] - time_use[1])*1.0f/ClockRate);
    printf("GPU block 1&2 d_time delay : %f ms\n", (time_use[1] - time_use[0])*1.0f/ClockRate);
       
}


/***************************************************************************/

void MatMul_CPU(const Matrix A, const Matrix B, Matrix &C)
{
    
    C.height=A.height;
    C.width=B.width;
    
    C.elements = new float [C.height*C.width];
    
    //CPU矩阵乘法，存入矩阵d
    for (int i = 0; i < A.height; i++)
    {
        for (int j = 0; j < B.width; j++)
        { 
            double t = 0;

            for (int k = 0; k < A.width; k++)
            { 
                t += A.elements[i * A.width + k] * B.elements[k * B.width + j]; 
            } 
            C.elements[i * B.width + j] = t; 

        } 
    }
}