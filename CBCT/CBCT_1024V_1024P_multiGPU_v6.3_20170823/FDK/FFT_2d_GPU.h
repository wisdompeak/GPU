// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_functions.h>
#include <helper_cuda.h>

// Complex data type
typedef float2 Complex;
static __device__ __host__ inline Complex ComplexAdd(Complex, Complex);
static __device__ __host__ inline Complex ComplexScale(Complex, float);
static __device__ __host__ inline Complex ComplexMul(Complex, Complex);
static __global__ void ComplexPointwiseMulAndScale(Complex *, const Complex *, float);

// Filtering functions
void Convolve(const Complex *, int, const Complex *, int, Complex *);

// Padding functions
void PadData_2D(const Complex *, Complex **, int, int, int);

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void FFT_2d_Convolve_GPU(float *g, Complex *filter, int pad_size);

// The filter size is assumed to be a number smaller than the signal size
#define NX        M
#define NY        N
#define PAD_SIZE    (2048)
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

void FilterDesign2D(Complex *filter_freq, int pad_size)
{

    int mem_size = sizeof(Complex) * pad_size * pad_size;    
    
    /****** Filter Design 1: spatial domain *************/
                
//     // Allocate host memory for the filter
//     Complex *h_filter_kernel = (Complex *)malloc(sizeof(Complex) * FILTER_KERNEL_SIZE*FILTER_KERNEL_SIZE);
//     
//     // Initialize the memory for the filter
//     for (int i = 0; i < FILTER_KERNEL_SIZE; i++)
//         for (int j = 0; j < FILTER_KERNEL_SIZE; j++)        
//         {
//             if ((i==FILTER_KERNEL_SIZE/2)&&(j==FILTER_KERNEL_SIZE/2))
//             {
// //                 h_filter_kernel[i*FILTER_KERNEL_SIZE+j].x = rand() / (float)RAND_MAX;
//                 h_filter_kernel[i*FILTER_KERNEL_SIZE+j].x = 1;
//                 h_filter_kernel[i*FILTER_KERNEL_SIZE+j].y = 0;
//             }
//             else
//             {
//                 h_filter_kernel[i*FILTER_KERNEL_SIZE+j].x = 0;
//                 h_filter_kernel[i*FILTER_KERNEL_SIZE+j].y = 0;                
//             }
//         }
// 
//     // Pad filter kernel    
//     Complex *h_padded_filter_kernel;
//     PadData_2D(h_filter_kernel, &h_padded_filter_kernel, FILTER_KERNEL_SIZE, FILTER_KERNEL_SIZE, pad_size);    
//     
//     // Allocate device memory for filter kernel
//     Complex *d_filter_kernel;
//     checkCudaErrors(cudaMalloc((void **)&d_filter_kernel, mem_size));    
//     
//     // Copy host memory to device
//     checkCudaErrors(cudaMemcpy(d_filter_kernel, h_padded_filter_kernel, mem_size, cudaMemcpyHostToDevice));
// 
//     // CUFFT plan
//     cufftHandle plan;    
//     if  (cufftPlan2d(&plan, pad_size, pad_size, CUFFT_C2C) != CUFFT_SUCCESS)
//     {
//         fprintf(stderr, "CUFFT error: Plan creation failed");
//         return;
//     } 
//     
//     // Transform kernel
//     printf("Transforming filter cufftExecC2C\n");
//     if  (cufftExecC2C(plan, d_filter_kernel, d_filter_kernel, CUFFT_FORWARD) != CUFFT_SUCCESS)
//     {
//         fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
//         return; 
//     }
//     
//     // Copy device memory to host
//     checkCudaErrors(cudaMemcpy(filter_freq, d_filter_kernel, mem_size, cudaMemcpyDeviceToHost));
//     
//     //Destroy CUFFT context
//     checkCudaErrors(cufftDestroy(plan));
//     
//     free(h_filter_kernel);
//     free(h_padded_filter_kernel);
//     checkCudaErrors(cudaFree(d_filter_kernel));    
    
    /****** Filter Design 2: freq domain *************/

//     // Allocate host memory for the filter
//     Complex *h_filter_kernel = (Complex *)malloc(sizeof(Complex) * pad_size * pad_size);
//     
//     int p,q,k;
//     float Mag = 3;
//     float r;
//     
//     // Initialize the memory for the filter
//     for (int i = 0; i < pad_size; i++)
//         for (int j = 0; j < pad_size; j++)
//         {
//                 if (i>pad_size/2) 
//                     p=i-pad_size;
//                 else 
//                     p=i;
//                 if (j>pad_size/2) 
//                     q=j-pad_size;
//                 else 
//                     q=j;
//                 
//                 r = sqrt(p*p+q*q);
//                 
//                 if (r<50)
//                 {
//                     k=1.0f;
//                 }
//                 else if (r<300)
//                 {                        
//                     k=sin((r-50)/(300-50)*PI/2)*Mag+1;
//                 }
//                 else if (r<400)
//                 {                        
//                     k=Mag+1;
//                 }
//                 else if (r<600)
//                 {
//                     k=sin(PI/2-(r-400)/(600-400)*PI/2)*Mag+1;      
//                 }
//                 else 
//                 {
//                     k=1.0f;
//                 }   
//                 
//                 h_filter_kernel[i*pad_size+j].x=1.0/k;
//                 h_filter_kernel[i*pad_size+j].y=0.0f;
//         }  
//     
//     FILE *fp;
//     char fn[120];        
//     
//     // Pad filter kernel : Not necessary here   
//     Complex *h_padded_filter_kernel = (Complex *)malloc(sizeof(Complex) * pad_size*pad_size);
//     PadData_2D(h_filter_kernel, &h_padded_filter_kernel, pad_size, pad_size, pad_size);            
//     
//     memcpy(filter_freq, h_padded_filter_kernel, mem_size);
//     
//     free(h_filter_kernel);
//     free(h_padded_filter_kernel);
    
    /****** Filter Design 3: Import from Outside *************/
    
    // Allocate host memory for the filter
    Complex *h_filter_kernel = (Complex *)malloc(sizeof(Complex) * pad_size * pad_size);

    FILE *fp;
    char fn[120];    
    
    sprintf(fn,"../FreqTF.dat");    
    if ( (fp = fopen(fn,"rb")) == NULL )
    {
        printf("can not open projection file: %s  \n",fn);
        exit(0);
    }
    fread(h_filter_kernel,sizeof(Complex)*pad_size*pad_size,1,fp);
    fclose(fp);
    
    for (int i=0; i<pad_size*pad_size; i++)
        h_filter_kernel[i].x=1/h_filter_kernel[i].x;
    
//     printf("Test %f,%f\n",h_filter_kernel[1820000].x,h_filter_kernel[1820000].y);
    
    memcpy(filter_freq, h_filter_kernel, mem_size);    
    
    free(h_filter_kernel);
    
}


int Image2Dfilter_GPU(float *g_int, float *g_out)
{
       
    int pad_size = PAD_SIZE;
    
    memcpy(g_out, g_int, sizeof(float)*M*N);
            
    Complex *filter = (Complex *)malloc(sizeof(Complex) * pad_size*pad_size);
    FilterDesign2D(filter, pad_size); // load Filter
    
    FFT_2d_Convolve_GPU(g_out, filter, pad_size);
    
    free(filter);
        
}


////////////////////////////////////////////////////////////////////////////////
void FFT_2d_Convolve_GPU(float *g, Complex *h_filter_kernel, int pad_size)
{
    int mem_size = sizeof(Complex) * pad_size * pad_size;    
    
    printf("    - GPU-based 2d-FFT is starting...\n");

    if (cudaSetDevice(GPUNumber)!=cudaSuccess)
        {
                std::cout<<"Error when initializing device!"<<std::endl;
                exit(-1);
        }
    
    /****** Signal Processing *************/
    
    // Allocate host memory for the signal
    Complex *h_signal = (Complex *)malloc(sizeof(Complex) * NX * NY);

    // Initialize the memory for the signal
    for (unsigned int i = 0; i < NX * NY; i++)
    {
        h_signal[i].x = g[i];
        h_signal[i].y = 0;
    }
        
    // Pad signal 
    Complex *h_padded_signal= (Complex *)malloc(sizeof(Complex) * pad_size*pad_size);
    PadData_2D(h_signal, &h_padded_signal, NX, NY, pad_size);
    
    // Allocate device memory for signal
    Complex *d_signal;
    checkCudaErrors(cudaMalloc((void **)&d_signal, mem_size));
    // Copy host memory to device
    checkCudaErrors(cudaMemcpy(d_signal, h_padded_signal, mem_size, cudaMemcpyHostToDevice));
        
    // CUFFT plan
    cufftHandle plan;    
    if  (cufftPlan2d(&plan, pad_size, pad_size, CUFFT_C2C) != CUFFT_SUCCESS)
    {
        fprintf(stderr, "CUFFT error: Plan creation failed");
        return;
    } 
    
    // Transform signal
//     printf("Transforming signal cufftExecC2C\n");    
    if  (cufftExecC2C(plan, d_signal, d_signal, CUFFT_FORWARD) != CUFFT_SUCCESS)
    {
        fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
        return; 
    }
    
    /*************** Load Filter *****************/
    
    // Allocate device memory for filter kernel
    Complex *d_filter_kernel;
    checkCudaErrors(cudaMalloc((void **)&d_filter_kernel, mem_size));    
    
    // Copy host memory to device
    checkCudaErrors(cudaMemcpy(d_filter_kernel, h_filter_kernel, mem_size, cudaMemcpyHostToDevice));    
    
    /********* Multiply the coefficients ****************/
    
    dim3  dimBlock(pad_size/2,1);
    dim3  dimGrid(2,pad_size);
    
    // Multiply the coefficients together and normalize the result
//     printf("Launching ComplexPointwiseMulAndScale<<< >>>\n");
    ComplexPointwiseMulAndScale<<<dimGrid,dimBlock>>>(d_signal, d_filter_kernel, 1.0f);

    // Check if kernel execution generated and error
    getLastCudaError("Kernel execution failed [ ComplexPointwiseMulAndScale ]");

    // Transform signal back
//     printf("Transforming signal back cufftExecC2C\n");
    if (cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_INVERSE) != CUFFT_SUCCESS)
    {
        fprintf(stderr, "CUFFT error: ExecC2C Inverse failed");
        return; 
    }
    
    // Copy device memory to host
    Complex *h_convolved_signal = (Complex *)malloc(sizeof(Complex) * pad_size * pad_size);    
    checkCudaErrors(cudaMemcpy(h_convolved_signal, d_signal, mem_size, cudaMemcpyDeviceToHost));

    for (int i=0; i<NY; i++)
        for (int j=0; j<NX; j++)
            g[i*NX+j]=h_convolved_signal[(i+(pad_size-NY)/2)*pad_size+j+(pad_size-NX)/2].x/(pad_size*pad_size); // Definition of inverse DFT
        

    /************************/
    
    if (cudaThreadSynchronize() != cudaSuccess)
    {
        fprintf(stderr, "Cuda error: Failed to synchronize\n");
        return;     
    }     
    
    //Destroy CUFFT context
    checkCudaErrors(cufftDestroy(plan));

    // cleanup memory
    free(h_signal);
    free(h_padded_signal);
    free(h_convolved_signal);    
    checkCudaErrors(cudaFree(d_signal));
    checkCudaErrors(cudaFree(d_filter_kernel));

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
    
}


/*** The following functions have been defined in FFT_1_GPU.h ****/

////////////////////////////////////////////////////////////////////////////////
// Pad data
////////////////////////////////////////////////////////////////////////////////
void PadData_2D(const Complex *signal, Complex **padded_signal, int N_X, int N_Y, int pad_size)
{

    Complex *new_data = (Complex *)malloc(sizeof(Complex) * pad_size*pad_size);
    memset(new_data, 0, sizeof(Complex)*pad_size*pad_size);
    
    for (int i=0; i<N_Y; i++)
    {
        memcpy(new_data+(i+(pad_size-N_Y)/2)*pad_size + (pad_size-N_X)/2, signal+i*N_X, sizeof(Complex)*N_X);
    }
    *padded_signal = new_data;

}

/***********The following functions have been defined in FFT_1d_GPU.h*************/

////////////////////////////////////////////////////////////////////////////////
// Filtering operations
////////////////////////////////////////////////////////////////////////////////

// // Computes convolution on the host
// void Convolve(const Complex *signal, int signal_size,
//               const Complex *filter_kernel, int filter_kernel_size,
//               Complex *filtered_signal)
// {
//     int minRadius = filter_kernel_size / 2;
//     int maxRadius = filter_kernel_size - minRadius;
// 
//     // Loop over output element indices
//     for (int i = 0; i < signal_size; ++i)
//     {
//         filtered_signal[i].x = filtered_signal[i].y = 0;
// 
//         // Loop over convolution indices
//         for (int j = - maxRadius + 1; j <= minRadius; ++j)
//         {
//             int k = i + j;
// 
//             if (k >= 0 && k < signal_size)
//             {
//                 filtered_signal[i] = ComplexAdd(filtered_signal[i], ComplexMul(signal[k], filter_kernel[minRadius - j]));
//             }
//         }
//     }
// }

////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Complex addition
static __device__ __host__ inline Complex ComplexAdd(Complex a, Complex b)
{
    Complex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

// Complex scale
static __device__ __host__ inline Complex ComplexScale(Complex a, float s)
{
    Complex c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}

// Complex multiplication
static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b)
{
    Complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

// Complex pointwise multiplication
static __global__ void ComplexPointwiseMulAndScale(Complex *a, const Complex *b, float scale)
{
    int i = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

    a[i] = ComplexScale(ComplexMul(a[i], b[i]), scale);
}
