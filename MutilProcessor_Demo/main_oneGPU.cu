#include <cmath>
#include <fstream>
#include <time.h>
#include <iostream>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <sys/time.h>


using namespace std;

// OpenMP 
#include <omp.h>
#define Max_num_device 	(1)
#define threads_x 		(512)
#define Block_x 		(512)
#define Block_y 		(1024)
#define Repeat_time 	(1)

#define Num_elem (Block_x*Block_y*threads_x)

#include "kernels.cu"
#include "hosts.cu"


int main(int argc, char **argv)
{
   cout<<"Hello"<<endl;
    
    size_t single_gpu_chunk_size = sizeof(float)*Num_elem/Max_num_device;

    // must be page-locked memory in order to achieve concurrency
    float* h_recon=NULL;
    float* h_proj=NULL;
    cudaHostAlloc((void**)&h_recon, sizeof(float)*Num_elem, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_proj, sizeof(float)*Num_elem, cudaHostAllocDefault);    
            
   /***********************/
   cudaDeviceProp deviceProp;
   for (int i=0; i<Max_num_device; i++)
   {       
       cudaGetDeviceProperties(&deviceProp,i);
       printf("Device %d has compute capability %d.%d.\n",i,deviceProp.major,deviceProp.minor);
   }
   cout<<"deviceProp.concurrentKernels "<<deviceProp.concurrentKernels<<endl;
   cout<<"deviceProp.asyncEngineCount "<<deviceProp.asyncEngineCount<<endl;
     
   
   /***********************/
   
   
   float * d_recon_addr[Max_num_device];
   float * d_proj_addr[Max_num_device];
   
	dim3  dimblock(threads_x,1);
    dim3  dimgrid(Block_x,Block_y/Max_num_device,1);  
    
    cudaEvent_t kernel_start_event[Max_num_device];
	cudaEvent_t kernel_stop_event[Max_num_device];
    float kernel_time[Max_num_device];
    
   for (int i=0; i<Max_num_device; i++)
   {
       cudaSetDevice(i);        
       cudaEventCreate(&kernel_start_event[i]);
       cudaEventCreate(&kernel_stop_event[i]);
   }    

    
   for (int i=0; i<Num_elem/2; i++)
       h_recon[i]=1.0f;
   for (int i=Num_elem/2; i<Num_elem; i++)
       h_recon[i]=2.0f;   
   cout<<h_recon[0]<<endl;
   cout<<h_recon[Num_elem/Max_num_device]<<endl;    
   
   /***************GPU initizalization and Loading*********************/
    
    //Set Timer 1
    struct timeval t1,t2;
    gettimeofday(&t1,NULL);  
   
   cout<<"GPU initializating..."<<endl;   
   
   for (int i=0; i<Max_num_device; i++)
   {
       cudaSetDevice(i);               
       cudaMalloc((void**)&d_recon_addr[i], single_gpu_chunk_size);  //blocking actions
       cudaMalloc((void**)&d_proj_addr[i], single_gpu_chunk_size); 
       cudaMemcpyAsync(d_recon_addr[i],h_recon+Num_elem/Max_num_device*i,single_gpu_chunk_size,cudaMemcpyDefault);                    
   }   
   
   cudaDeviceSynchronize();
            
    // End timer
    gettimeofday(&t2,NULL);
    printf("Init timing (gettimeofday): %f (s)\n", (t2.tv_sec-t1.tv_sec + (t2.tv_usec-t1.tv_usec)/1000000.0));
   
   cout<<"initializating done..."<<endl;
   
   /************* Forward *********************/
      
   gettimeofday(&t1,NULL); 
   
   for (int i=0; i<Max_num_device; i++)
   {       
        cudaSetDevice(i);        
        cudaEventRecord(kernel_start_event[i],0);        
        kernel_forward_projection<<<dimgrid, dimblock>>>(d_recon_addr[i], d_proj_addr[i]);  
   }   
      

/***    AddProjection(d_proj_addr);  ***/
//    {
//     cudaSetDevice(0);                   
//     for (int i=1; i<Max_num_device; i++)
//     {
//         cudaDeviceEnablePeerAccess(i,0);
//         kernel_add<<<dimgrid, dimblock>>>(d_proj_addr[0],d_proj_addr[i]);
//         cudaMemcpyAsync(d_proj_addr[i],d_proj_addr[0],single_gpu_chunk_size,cudaMemcpyDefault);        
//     }          
//    }

/***    Backprojection  ***/
   
   for (int i=0; i<Max_num_device; i++)
   {            
        cudaSetDevice(i);        
        kernel_back_projection<<<dimgrid, dimblock>>>(d_recon_addr[i], d_proj_addr[i]);  
   } 
      
   for (int i=0; i<Max_num_device; i++)
   {
       cudaSetDevice(i);     
       cudaEventRecord(kernel_stop_event[i],0);
       cudaEventSynchronize(kernel_stop_event[i]);
   }
    
   for (int i=0; i<Max_num_device; i++)
   {
       cudaSetDevice(i);     
       cudaEventElapsedTime(&kernel_time[i], kernel_start_event[i], kernel_stop_event[i]);
       cudaEventDestroy(kernel_start_event[i]);
       cudaEventDestroy(kernel_stop_event[i]); 
       cout<<"GPU timing No."<<i<<" : "<<kernel_time[i]<<endl;
   }       
   
   cudaDeviceSynchronize();
    // End timer
    gettimeofday(&t2,NULL);
    printf("Compute timing (gettimeofday): %f (s)\n", (t2.tv_sec-t1.tv_sec + (t2.tv_usec-t1.tv_usec)/1000000.0));
   
   
    /****************dump out********************/
   gettimeofday(&t1,NULL); 
   
   for (int i=0; i<Max_num_device; i++)
   {
       cudaSetDevice(i);
       cudaMemcpyAsync(h_recon+Num_elem/Max_num_device*i,d_recon_addr[i],single_gpu_chunk_size,cudaMemcpyDefault);
   }
    
   cudaDeviceSynchronize();
   
   gettimeofday(&t2,NULL);
   printf("dump timing (gettimeofday): %f (s)\n", (t2.tv_sec-t1.tv_sec + (t2.tv_usec-t1.tv_usec)/1000000.0));

   /*************************/
    int shift=100024;
   cout<<h_recon[0+shift]<<endl;
   cout<<h_recon[Num_elem/2+shift]<<endl;
   
   for (int i=0; i<Max_num_device; i++)
   {
       cudaSetDevice(i);     
       cudaFree(d_recon_addr[i]);
       cudaFree(d_proj_addr[i]);
   }  
   
   cudaFree(h_recon);
   cudaFree(h_proj);
   
	return 0; 
}
