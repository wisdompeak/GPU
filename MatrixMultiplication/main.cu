// #include <cutil_inline.h>
#include <cmath>
#include <fstream>
#include <time.h>
#include <iostream>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

using namespace std;

#define GPU_Device 2
#define THREAD_NUM 1024
#define N 2048
#define BLOCK_NUM N*N/THREAD_NUM

int ClockRate;

typedef struct
{
    int width;
    int height;
    float *elements;
}Matrix;

#include "InitGPU.h"
#include "Kernel_Matrix.cu"
#include "host_Matrix.cpp"



int main() 
{

    //CUDA 初始化
    if (!InitCUDA()) 
    {
        return 0;
    }
       
    Matrix A;
    A.width = N;
    A.height = N;
    A.elements = new float [A.width*A.height];
    bzero(A.elements, sizeof(int)*A.width*A.height); 
    
    Matrix B;
    B.width = N;
    B.height = N;
    B.elements = new float [B.width*B.height];  
    bzero(B.elements, sizeof(int)*B.width*B.height); 
        
    //设置随机数种子
    srand(0);
    
    matgen(A);
    matgen(B);    

    /*****CPU computing ***/
    struct timeval t1,t2;
    gettimeofday(&t1,NULL);   
    Matrix C1;        
    MatMul_CPU(A,B,C1);
    gettimeofday(&t2,NULL);   
    printf("CPU whole computing (gettimeofday): %f (ms)\n", (t2.tv_sec-t1.tv_sec + (t2.tv_usec-t1.tv_usec)/1000000.0)*1000.0);
    
    /*****GPU computing ***/    
    gettimeofday(&t1,NULL);   
    Matrix C2;    
    MatMul_GPU(A,B,C2);
    gettimeofday(&t2,NULL);   
    printf("GPU whole computing (gettimeofday): %f (ms)\n", (t2.tv_sec-t1.tv_sec + (t2.tv_usec-t1.tv_usec)/1000000.0)*1000.0);
    
//     for (int i=0; i<A.height; i++)
//     {
//         for (int j=0; j<A.width; j++)
//         {
//             cout<<A.elements[i*A.width+j]<<" ";
//         }
//         cout<<endl;
//     }
//     
//     for (int i=0; i<B.height; i++)
//     {
//         for (int j=0; j<B.width; j++)
//         {
//             cout<<B.elements[i*B.width+j]<<" ";
//         }
//         cout<<endl;
//     } 
//     
//     for (int i=0; i<C2.height; i++)
//     {
//         for (int j=0; j<C2.width; j++)
//         {
//             cout<<C2.elements[i*C2.width+j]<<" ";
//         }
//         cout<<endl;
//     }     
    
    float max_err = 0;
    float average_err = 0;     
    
    for (int i=0; i<C2.height; i++)
    {
        for (int j=0; j<C2.width; j++)
        {
            max_err = max(max_err, abs(C2.elements[i*C2.width+j]-C1.elements[i*C1.width+j]));
            average_err += (C2.elements[i*C2.width+j]-C1.elements[i*C1.width+j])*(C2.elements[i*C2.width+j]-C1.elements[i*C1.width+j]);
        }        
    }    
    cout<<"max_error = "<<max_err<<endl;
    cout<<"avg_error = "<<sqrt(average_err)/N<<endl;
    
    delete [](A.elements);
    delete [](B.elements);
    delete [](C1.elements);
    delete [](C2.elements);
    
}