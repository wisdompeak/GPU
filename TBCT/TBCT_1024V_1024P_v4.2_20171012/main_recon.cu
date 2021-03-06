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

#include <omp.h>
using namespace std;

#define Default_GPU (0)

#define Number_of_Devices (4)  // 1,2,3,4
/* if Number_of_Devices is 3, please modify your volume and projection size
 * so that the size could be a multiply of 3
 */ 
        
#define pixelDriven (0)     
    // 1: pixel-driven backprjection;
    // 0: voxel-driven backprjection;

#define MAX_infi (1.0/0.0)
      
#define FISTA (1)
    // -2: execution configuration test, for CUDA 7.0 and above
    // -1: joint operator test
    // 0: SART + denoise
    // 1: SART + FISTA
    // 2: FDK

#define DenoiseOption (1)
    // 1: FGP_denoise_GPUx7_exact, requiring x7 volume size, fast
    // 2: FGP_denoise_GPUx4_exact, requiring x4 volume size, slowest
    // 3: FGP_denoise_GPUx4_apprx, requiring x4 volume size, slower
    // 4: GP_denoise_GPUx4_fast, requiring x4 volume size, fast, slow in convergence
                             
const int Iter_num = 21; 

const float Lip_con = 32.0f;          
const float lambda_TV = 0.00f;         //regularization parameter for the tv norm
const float lambda_L1 = 0.00f;         //regularization parameter for the l1 norm

// Distances
const float DSO = 0.2f;
const float DOD = -0.2f;

/* Parameters for reconstructed volume
 * You need to consider the GPU memory capcity
 * Also, double check the GPU block configuration
 */
const int M = 512;      // reconstruction volume x range       maximum: 1024
const int N = 512;      // reconstruction volume y range       maximum: 1024
const int ZETA = 512;   // reconstruction volume z range       maximum: 1024 

const int NO_X = M;
const int NO_Y = N;
const int NO_Z = ZETA;
const int NO_VOXEL = NO_X*NO_Y*NO_Z; 

const float volumn_x = 1.0e-4 ; // volume voxel physical size (in m)
const float inv_volumn_x = 1.0/volumn_x; 
const int M_Offset = 0;
const float boundary_voxel_x  = -volumn_x*(float(M)/2.0f+M_Offset);

const float volumn_y = volumn_x ;
const float inv_volumn_y = 1.0/volumn_y; 
const float boundary_voxel_y  = -volumn_y*(float(N)/2.0f);

const float volumn_z = volumn_x ;
const float inv_volumn_z = 1.0/volumn_z; 
const float boundary_voxel_z = -volumn_z*(float(ZETA)/2.0f); 
const float boundary_voxel_z_upper = -volumn_z*(float(ZETA)/2.0f)+ZETA*volumn_z; 

// parameters for half detector offset
const float Offset =  0;

const float PI = 3.141592653589793f; 

#define K (64)  // the factor that adjust the number of sources and projection height
        
// parameters for Detector
const float Detector_pixel_x = 1.2e-4; // all in m
const float inv_Detector_pixel_x = 1.0f/Detector_pixel_x; // all in m
const int R = 1024;         // detector width (in pixel)
const int Z_prj = 1024/K;       // detector height (in pixel)

const float Detector_Ymin = -Detector_pixel_x*((R)/2.0f - 0.5f) + Offset; 
const float Detector_Zmin = -Detector_pixel_x*((Z_prj)/2.0f - 0.5f); 
const float Detector_Zmax = +Detector_pixel_x*((Z_prj)/2.0f - 0.5f); 

// parameters for Source  :  automatic source distribution
#if (K==1)
const int N_source = 1;    // number of sources
const float Source_z_max = 0.0f;
const float Source_z_min = 0.0f;
const float Source_interval = 0.0f;
const float Source_y = Offset;
const float Source_x = DSO;
#else
const int N_source = K;    
const float Source_z_max = (boundary_voxel_z_upper-Detector_Zmax)*(DSO-DOD)/DSO+Detector_Zmax;
const float Source_z_min = (boundary_voxel_z-Detector_Zmin)*(DSO-DOD)/DSO+Detector_Zmin;
const float Source_interval = (Source_z_max-Source_z_min)/(K-1);   // source spacing
const float Source_y = Offset;
const float Source_x = DSO;
#endif


// parameters for Acqusition
const int Nviews = 360; 
const float us_rate = 1.00f; 
const float initialAngle= 0.00f ;
const float shiftAngle= 0.0f;

#include "InitGPU.h"
#include "kernel_tool_functions.cu"
#include "host_tool_functions.cu"

#if (pixelDriven==1 && Number_of_Devices==1)
    #include "pixel/kernel_IterativeRecon_TBCT_pixel.cu"
    #include "pixel/host_IterativeRecon_TBCT_pixel.c"
#elif (pixelDriven==1 && Number_of_Devices>1)
    #include "pixel/kernel_IterativeRecon_TBCT_pixel_multiGPU.cu"
    #include "pixel/host_IterativeRecon_TBCT_pixel_multiGPU.c"
#elif (pixelDriven==0 && Number_of_Devices==1)
    #include "voxel/kernel_IterativeRecon_TBCT_voxel.cu"
    #include "voxel/host_IterativeRecon_TBCT_voxel.c"
#elif (pixelDriven==0 && Number_of_Devices>1)
    #include "voxel/kernel_IterativeRecon_TBCT_voxel_multiGPU.cu"
    #include "voxel/host_IterativeRecon_TBCT_voxel_multiGPU.c"
#endif

#include "FGP/host_FGP_Denoise_CPU.h"

#include "FGP/kernel_FGP_Denoise_GPUx4.cu"
#include "FGP/host_FPG_Denoise_GPUx4.c"

#include "FGP/kernel_FGP_Denoise_GPUx7.cu"
#include "FGP/host_FGP_Denoise_GPUx7.c"

#include "FDK/host_FDK.cpp"
// #include "FDK/EqualWeighting/host_FDK.cpp"
// #include "FDK/OneRay/host_FDK.cpp"

main(int argc, char ** argv)
{
    
    // print CUDA information
    if (!InitCUDA()) 
    {
        return 0;
    }    
    
    /* ************* User defined parameters ************/
    
    char directory[]="/home/huifeng/TBCT/PigHead/";
    char objectName[]="SLphantom_64_noise_05";
    char outputFolder[]="/Reconstructed_images_SL/";   
        
	int Niter_denoise = 20;         //iter number for denoising problem

    /*********** other declared variables ************/
    
    float step_size = 2.0f/Lip_con;
    float lambda_denoise_TV = 2.0f*lambda_TV/Lip_con;
           
	double data_fidelity;
	double tv_value = 0.0f;
	double object_function_value_xk;       
    double *object_function_array = new double [Iter_num*3];
	bzero(object_function_array, sizeof(double)*Iter_num*3);  

    FILE *fp;		    
    char filename[200];
    char fn[200];
    int VIEW = Nviews;    
    float endAngle = initialAngle + (VIEW - 1)*us_rate;      
    
    /****************  CPU memory allocation  *****************/
    
	size_t size_proj_data = sizeof(float)*R*Z_prj*N_source*Nviews;
	size_t size_volume = sizeof(float)*M*N*ZETA;    
    
    // for 3D reconstructed volume
	float *F_Y_k = new float [M*N*ZETA];    // Y(k)
	bzero(F_Y_k, sizeof(float)*M*N*ZETA);

    float *F_X_k_1 = new float [M*N*ZETA];  // X(k-1)
    bzero(F_X_k_1, sizeof(float)*M*N*ZETA);    
    
    /*** page-locked memory ***/          

    float *h_proj_data;
    cudaHostAlloc((void**)&h_proj_data, size_proj_data, cudaHostAllocDefault); 
            
    float *F_recon;
    cudaHostAlloc((void**)&F_recon, size_volume, cudaHostAllocDefault);
    
        
	/********** Read Projections **************/

//     printf("Read projection files ...\n");
//     
// 	for (int j=0;j<Nviews;j++)
// 	{
//         fileAngle = float(j*us_rate + initialAngle);        
//     
//         strcpy(filename,directory);
//  		sprintf(fn,"/FinalData/proj_%.2f.bin", fileAngle);
//         strcat(filename,fn);
//         cout<<fn<<endl;
//         if ( (fp = fopen(filename,"rb")) == NULL )
//         {
//         	printf("Can not open projection files for main function \n");
//             printf("%s\n",filename);                    
//             exit(0);
//         }
//         fseek(fp,sizeof(float)*R*(int(2048/2-Z_prj/2)),0); // If you want to read part of the projections
//         fread(h_proj_data + j*R*Z_prj*N_source, sizeof(float)*R*Z_prj*N_source,1,fp);  // stack all projections together
//         fclose(fp);
// 	}       
    
    
        strcpy(filename,directory);
 		sprintf(fn,"GeneratedNoisyProjection_05.dat");
        strcat(filename,fn);
        cout<<fn<<endl;
        if ( (fp = fopen(filename,"rb")) == NULL )
        {
        	printf("Can not open projection files for main function \n");
            printf("%s\n",filename);                    
            exit(0);
        }        
        fread(h_proj_data, sizeof(float)*R*Z_prj*N_source*Nviews,1,fp);  // stack all projections together
        fclose(fp);    

    
    /**************** Inverse Crime Studies ********************************/
    
//     // load volumetric image
//     
//         strcpy(filename,directory);
//  		sprintf(fn,"SLphantom3d_512.dat");        
//         strcat(filename,fn);
//         cout<<"Loading "<<fn<<endl;
//         if ( (fp = fopen(filename,"rb")) == NULL )
//                 {
//                     printf("Can not load volumetric image \n");
//                     printf("%s\n",filename);                    
//                     exit(0);
//                 }
//         fread(F_recon, sizeof(float)*M*N*ZETA,1,fp);
//         fclose(fp);             
//         cout<<"Load Phantom Sucessfully!"<<endl;
//         
//         if (FISTA!=-1)
//         {
//             Forward_3D_ray_driven_siddon_TBCT(F_recon,h_proj_data);  
//             bzero(F_recon,sizeof(float)*M*N*ZETA);
//             SaveDeviceDataToFile(h_proj_data,R*Z_prj*N_source*Nviews,"../GeneratedProjection.dat");
//         }
            
        
	/****************Iteration Reconstruction******************************/
	

    float t_k;
    float t_k_1=1.0f;    
            
    
    //Set Timer 1
    struct timeval t1,t2,t0;
    gettimeofday(&t1,NULL);
    gettimeofday(&t0,NULL);
    
    if (FISTA==2)
    {
        Reconstruction_FDK(F_recon,h_proj_data);
        SaveDeviceDataToFile(F_recon,M*N*ZETA,"../ReconstructionFDK.dat");                           
        goto endProgram;  
    } 
    
    
	for (int k=1;k<=Iter_num;k++)
	{	
//         if (FISTA==-2)  // "occupancy calculator", check the best execution configuration. Refer to the program guide
//         {
//             int numBlocks;       // Occupancy in terms of active blocks
//             int blockSize = 256;
//             int activeWarps;
//             int maxWarps;
// 
//             cudaDeviceProp prop;
//             cudaGetDeviceProperties(&prop, Default_GPU);
//             
//             cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,backprj_ray_driven_3d_kernel,blockSize,0);
//             activeWarps = numBlocks * blockSize / prop.warpSize;
//             maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
//             std::cout << "Occupancy: " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;      
//             std::cout << "MaxActiveBlocksPerMultiprocessor: " << numBlocks << std::endl;      
//             goto endProgram;   
//         }
        
        if (FISTA==-1)  // check matched joint operator;
        {
            /*Note: You need to first uncomment the phantom loading code to initialize a valid F_recon*/            
            CheckMatchedJointOperator(F_recon);  
            goto endProgram;   
        }
                
        if (FISTA==0)
        {
        
            printf("Undergoing SART updating...  relaxation = %f\n", step_size);					            
        
            Reconstruction_3D_ray_driven_TBCT(F_recon, h_proj_data, step_size); 
                            
            if (lambda_TV>0.0f)                    
            {
                printf("Undergoing TV regularization ...\n");                                
                                
                switch(DenoiseOption) // Denoise options
                {
                    case 1 : FGP_denoise_GPUx7_exact(F_recon, lambda_denoise_TV, Niter_denoise); 
                             break;     // require x7 volume size, fast
                    case 2 : FGP_denoise_GPUx4_exact(F_recon, lambda_denoise_TV, Niter_denoise); 
                             break;     // require x4 volume size, slowest
                    case 3 : FGP_denoise_GPUx4_apprx(F_recon, lambda_denoise_TV, Niter_denoise); 
                             break;     // require x4 volume size, slower
                    case 4 : GP_denoise_GPUx4_fast(F_recon, lambda_denoise_TV, Niter_denoise); 
                             break;     // require x4 volume size, fast, slow in convergence
                }

                std::cout<<"TV regularization finished.\n";
            }
                        
        }
        
        
        if (FISTA==1)
        {
            printf("Undergoing SART updating...  relaxation = %f\n", step_size);					            
                    
            #pragma omp parallel for            
            for (int i=0;i<NO_VOXEL;i++)
            	F_recon[i] = F_Y_k[i];            

            Reconstruction_3D_ray_driven_TBCT(F_recon, h_proj_data, step_size); 
                        
            if (lambda_TV>0.0f)                    
            {
                
                printf("Undergoing TV regularization ...\n");          
                switch(DenoiseOption) // Denoise options
                {
                    case 1 : FGP_denoise_GPUx7_exact(F_recon, lambda_denoise_TV, Niter_denoise); 
                             break;     // require x7 volume size, fast
                    case 2 : FGP_denoise_GPUx4_exact(F_recon, lambda_denoise_TV, Niter_denoise); 
                             break;     // require x4 volume size, slowest
                    case 3 : FGP_denoise_GPUx4_apprx(F_recon, lambda_denoise_TV, Niter_denoise); 
                             break;     // require x4 volume size, slower
                    case 4 : GP_denoise_GPUx4_fast(F_recon, lambda_denoise_TV, Niter_denoise); 
                             break;     // require x4 volume size, fast, slow in convergence
                }                
                std::cout<<"TV regularization finished.\n";
            }
                     
            t_k = (1.0f + sqrt(1.0f + 4.0f*t_k_1*t_k_1) )/2.0f;        
                        // Note: t(k) = [1+sqrt(1+4*t(k-1)^2)]/2
            
            #pragma omp parallel for            
            for (int i=0;i<NO_VOXEL;i++)
            {
            	F_Y_k[i] = F_recon[i] + (t_k_1 -1.0f)/t_k * (F_recon[i] - F_X_k_1[i]);
                    // Note: y(k) = x(k) + [t(k-1) -1]/t(k) * [x(k)-x(k-1)]
                F_X_k_1[i] = F_recon[i];
                    // Note: Update x(k-1):   x(k-1) <- x(k)
            }
                        
            t_k_1 = t_k;
                        // Note: Update t(k-1):   t(k-1) <- t(k)                                                
        }
        
        
        #pragma omp parallel sections
        {
            
            /*****************Calculating Obj Func Value ********************/
            
          #pragma omp section
          {
            std::cout<<"Calculating Object Func Value ...\n";            
                //Note: object function value || Ax - b ||_2 + 2*lambda_TV*||f||_tvnorm  + lambda_L1*||\phi f ||_L1 ;
                        
            /*** data fidelity ****/
            cudaSetDevice(Default_GPU);
            float *d_proj_forward = NULL;
            cudaMalloc((void**)&d_proj_forward, size_proj_data);
            
            std::cout<<" - calculating data fidelity ... \n";	                   
            Forward_3D_ray_driven_siddon_TBCT(F_recon, d_proj_forward);               
            data_fidelity = L2_norm_gpu(d_proj_forward, h_proj_data);                        
           	std::cout<<"    * L2 Norm="<<data_fidelity<<endl;              
            cudaFree(d_proj_forward);

            /*** TV norm ****/
            float *d_recon;
            cudaMalloc((void**)&d_recon, size_volume);                            
            cudaMemcpyAsync(d_recon,F_recon,size_volume,cudaMemcpyDefault);            
            
            std::cout<<" - calculating TV norm ... \n";
            tv_value = TV_norm_gpu(d_recon);                        
            std::cout<<"    * TV value="<<tv_value<<endl;              
            cudaFree(d_recon);
            
            /***** obj function ******/
            object_function_value_xk = data_fidelity + 2.0f*lambda_TV*tv_value;
                //Note: object_function_value_xk = data_fidelity + 2.0f*lambda_TV*tv_value + 1.0f*lambda_L1*l1_value;
            object_function_array[k*3-3] = tv_value;
            object_function_array[k*3-2] = data_fidelity;
            object_function_array[k*3-1] = object_function_value_xk;
           	std::cout<<"Object function value for x(k) = "<< tv_value <<" + "<< data_fidelity <<" = "<<object_function_value_xk <<std::endl; 
                                    
         }
            
            /***************** Saving ********************/            
            
         #pragma omp section
         {                
                
            strcpy(filename,directory);
            sprintf(fn,"%s/%s_%d_%d_%d_%.0fum_new_view_%d_(%.0f,%.0f)_TV_%.2f_L1_%.2f_Lip_%.2f.recon",outputFolder, objectName, M,N,ZETA, volumn_x*1000000, Nviews, initialAngle, endAngle, lambda_TV, lambda_L1, Lip_con);
            strcat(filename,fn);        	
        	if ( (fp = fopen(filename,"wb")) == NULL )
        	{
                	printf("can not open file to write the intermediate reconstructed image \n");
                    printf("%s\n",filename);
                	exit(0);
        	}
        	fwrite(F_recon,sizeof(float)*M*N*ZETA,1,fp);
        	fclose(fp);      
                    
            if (k%10==1)
            {
                strcpy(filename,directory);
                sprintf(fn,"%s/%s_%d_%d_%d_%.0fum_iterative_%d_view_%d_(%.0f,%.0f)_TV_%.2f_L1_%.2f_Lip_%.2f.recon",outputFolder, objectName, M,N,ZETA, volumn_x*1000000, k, Nviews, initialAngle, endAngle, lambda_TV, lambda_L1, Lip_con);
                strcat(filename,fn);                
        		if ( (fp = fopen(filename,"wb")) == NULL )
        		{
                		printf("can not open file to write the reconstr5ucted image \n");
                        printf("%s\n",filename);
                		exit(0);
        		}
        		fwrite(F_recon,sizeof(float)*M*N*ZETA,1,fp);
        		fclose(fp);
            }
            
          }
        }
        
        strcpy(filename,directory);
        sprintf(fn,"%s/object_func_%s_view_%d_(%.0f,%.0f)_TV_%.2f_Lip_%.2f.bin",outputFolder, objectName, Nviews, initialAngle, endAngle, lambda_TV, Lip_con);
        strcat(filename,fn);                 
        if ( (fp = fopen(filename,"wb")) == NULL )
        {
            printf("can not open file to write the tv_value_file \n");
            printf("%s\n",filename);
            exit(0);
        }
        fwrite(object_function_array,sizeof(double)*k*3,1,fp);
        fclose(fp);        
        
        std::cout<<"Have done "<< k <<" iteration(s)"<<std::endl;        
        
        gettimeofday(&t2,NULL);
        printf("Time ellapsed for this iteration: %f (s)\n\n\n", (t2.tv_sec-t1.tv_sec + (t2.tv_usec-t1.tv_usec)/1000000.0));
        t1=t2;          
	}
    

	endProgram: ;

    gettimeofday(&t2,NULL);
    printf("Whole computing (gettimeofday): %f (s)\n\n\n", (t2.tv_sec-t0.tv_sec + (t2.tv_usec-t0.tv_usec)/1000000.0));
    
    cudaFreeHost(h_proj_data);
    cudaFreeHost(F_recon);
            
	delete []F_Y_k;
    delete []F_X_k_1;
    delete []object_function_array;

	return 0;
}
