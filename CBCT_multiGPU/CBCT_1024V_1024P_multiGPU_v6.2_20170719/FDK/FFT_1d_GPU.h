// includes, project
// #include <cuda_runtime.h>
#include <cufft.h>
// #include <helper_functions.h>
// #include <helper_cuda.h>

// Complex data type
typedef float2 Complex;
static __device__ __host__ inline Complex ComplexAdd(Complex, Complex);
static __device__ __host__ inline Complex ComplexScale(Complex, float);
static __device__ __host__ inline Complex ComplexMul(Complex, Complex);
static __global__ void ComplexPointwiseMulAndScale(Complex *, const Complex *, float);

// Filtering functions
void Convolve(const Complex *, int, const Complex *, int, Complex *);

// Padding functions
void PadData_1D(const Complex *, Complex **, int, int, int);

// The filter size is assumed to be a number smaller than the signal size
#define SIGNAL_SIZE        (R)                // number of elements per row
#define BATCH              (Z_prj*1)   // One batch cannot contain too many

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

void FFT_1d_Convolve_GPU(float *g, Complex *h_filter_kernel, int pad_size)
{
    
    int mem_size = sizeof(Complex) * pad_size * BATCH;    
    
//     printf("    - GPU-based 1d-FFT is starting...\n");

    if (cudaSetDevice(Default_GPU)!=cudaSuccess)
        {
                std::cout<<"Error when initializing device!"<<std::endl;
                exit(-1);
        }
    
    
    
    /****** Signal Processing *************/
    
    // Allocate host memory for the signal
    Complex *h_signal = (Complex *)malloc(sizeof(Complex) * SIGNAL_SIZE * BATCH);

    // Initialize the memory for the signal
    for (unsigned int i = 0; i < SIGNAL_SIZE * BATCH; ++i)
    {
        h_signal[i].x = g[i];
        h_signal[i].y = 0;
    }
    
    // Pad signal 
    Complex *h_padded_signal;
    PadData_1D(h_signal, &h_padded_signal, SIGNAL_SIZE, pad_size, BATCH);
    
    // Allocate device memory for signal
    Complex *d_signal;
    cudaMalloc((void **)&d_signal, mem_size);
    // Copy host memory to device
    cudaMemcpy(d_signal, h_padded_signal, mem_size, cudaMemcpyHostToDevice);
    
    // CUFFT plan
    cufftHandle plan;        
    
    if  (cufftPlan1d(&plan, pad_size, CUFFT_C2C, BATCH) != CUFFT_SUCCESS)
    {
        fprintf(stderr, "CUFFT error: Plan creation failed\n");
        return;
    } 
        
    // Transform signal
//     printf("Transforming signal cufftExecC2C\n");    
    if  (cufftExecC2C(plan, d_signal, d_signal, CUFFT_FORWARD) != CUFFT_SUCCESS)
    {
        fprintf(stderr, "CUFFT error: ExecC2C Forward failed\n");
        return; 
    }
    
    
    /****** Filter Design 2: freq domain *************/
    
    // Pad filter kernel    
    Complex *h_padded_filter_kernel = (Complex *)malloc(sizeof(Complex) * pad_size * BATCH);
    for (int i=0; i<BATCH; i++)
    {
        memcpy(h_padded_filter_kernel+i*pad_size, h_filter_kernel, pad_size * sizeof(Complex));
    }
    
    
    // Allocate device memory for filter kernel
    Complex *d_filter_kernel;
    cudaMalloc((void **)&d_filter_kernel, mem_size);    
    
    // Copy host memory to device
    cudaMemcpy(d_filter_kernel, h_padded_filter_kernel, mem_size, cudaMemcpyHostToDevice);    
    

    /********* Multiply the coefficients ****************/
    
    dim3  dimBlock(pad_size/2,1);
    dim3  dimGrid(2,BATCH);
    
    // Multiply the coefficients together and normalize the result
//     printf("Launching ComplexPointwiseMulAndScale<<< >>>\n");
    ComplexPointwiseMulAndScale<<<dimGrid,dimBlock>>>(d_signal, d_filter_kernel, 1.0f);


    // Transform signal back
//     printf("Transforming signal back cufftExecC2C\n");
    if (cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_INVERSE) != CUFFT_SUCCESS)
    {
        fprintf(stderr, "CUFFT error: ExecC2C Inverse failed");
        return; 
    }
    
    // Copy device memory to host
    Complex *h_convolved_signal = h_padded_signal;
    cudaMemcpy(h_convolved_signal, d_signal, mem_size, cudaMemcpyDeviceToHost);

    for (int i=0; i<BATCH; i++)
        for (int j=0; j<SIGNAL_SIZE; j++)
            g[i*SIGNAL_SIZE+j]=h_convolved_signal[i*pad_size+j].x/pad_size; // Definition of inverse DFT
    
    /************************/
    
    //Destroy CUFFT context
    cufftDestroy(plan);
    cudaDeviceSynchronize();

    // cleanup memory
    free(h_signal);
    free(h_padded_signal);
    free(h_padded_filter_kernel);
    cudaFree(d_signal);
    cudaFree(d_filter_kernel);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits

//     cudaDeviceReset();
}

// Pad data
void PadData_1D(const Complex *signal, Complex **padded_signal, int signal_size, int pad_size, int batch)
{

    Complex *new_data = (Complex *)malloc(sizeof(Complex) * pad_size * batch);
    memset(new_data, 0, sizeof(Complex) * pad_size * batch);
    
    for (int i=0; i<BATCH; i++)
    {
        memcpy(new_data + i*pad_size, signal + i*signal_size, signal_size * sizeof(Complex));
        memset(new_data + i*pad_size + signal_size,      0, (pad_size - signal_size) * sizeof(Complex));
    }
    *padded_signal = new_data;

}

////////////////////////////////////////////////////////////////////////////////
// Filtering operations
////////////////////////////////////////////////////////////////////////////////

// Computes convolution on the host
void Convolve(const Complex *signal, int signal_size,
              const Complex *filter_kernel, int filter_kernel_size,
              Complex *filtered_signal)
{
    int minRadius = filter_kernel_size / 2;
    int maxRadius = filter_kernel_size - minRadius;

    // Loop over output element indices
    for (int i = 0; i < signal_size; ++i)
    {
        filtered_signal[i].x = filtered_signal[i].y = 0;

        // Loop over convolution indices
        for (int j = - maxRadius + 1; j <= minRadius; ++j)
        {
            int k = i + j;

            if (k >= 0 && k < signal_size)
            {
                filtered_signal[i] = ComplexAdd(filtered_signal[i], ComplexMul(signal[k], filter_kernel[minRadius - j]));
            }
        }
    }
}

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



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


void RampFiltering_GPU(float *g_in, float *g_out)
{       
    int pad_size = R*2; // the size after padding
    int k;
    
    float F_max = 1.0/2.0/(Detector_pixel_x*DSO/(DSO-DOD));	
            // For FDK, the pixel size should be the one for the contact plane
    float F_step = F_max/(pad_size/2); 
                    
    float *g = (float *)malloc(sizeof(float) * R*BATCH);
    
    /************ Ramp Filter **************/   
    
    Complex *filter = (Complex *)malloc(sizeof(Complex) * pad_size);

    for (int i = 0; i < pad_size; i++)
    {
        if (i>=pad_size/2)
            k=i-pad_size;
        else
            k=i;
        
//         if (abs(i-pad_size/2)<pad_size/8)   // filter out high frequency
//             k=0;
        
        filter[i].x = abs(k)*F_step;
        filter[i].x *= 0.5f;  // For FDK only
        filter[i].y = 0;
    }  
    
    /************ End of Filter design **************/
    
    for (int k=0; k<Nviews; k++)
    {
        for (int i=0; i<R*BATCH; i++)
            g[i]=g_in[R*BATCH*k+i];
        
        FFT_1d_Convolve_GPU(g, filter, pad_size); 
        
        for (int i=0; i<R*BATCH; i++)
            g_out[R*BATCH*k+i]=g[i];        
    }
    
    free(filter);
    free(g);
    
}


