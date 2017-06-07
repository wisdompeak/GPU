#include <cutil_inline.h>
#include <cmath>
#include <fstream>
#include <time.h>
#include <iostream>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

using namespace std;

const int M = 256;      // reconstruction volume x range       maxium: 1024
const int N = 256;      // reconstruction volume y range       maxium: 1024
const int ZETA = 256;    // reconstruction volume z range       maxium: 1024 Consider GPU Memory

const int NO_X = M;
const int NO_Y = N;
const int NO_Z = ZETA;
const int NO_VOXEL = NO_X*NO_Y*NO_Z; 

const int R = 300;         // detector width
const int Z_prj = 4;
const int N_source = 75;    
const float Source_interval = 4e-3; // all in m
const float Detector_pixel_x = 2.54e-3; // all in m

const float DSO = 1.5f; 
const float DOD = -2.0f; 

const float PI = 3.141592653589793f; 

// parameters for half detector offset
const float Offset =  0;

const float volumn_x = 1.0e-3 ;
const float inv_volumn_x = 1.0/volumn_x; 
const int M_Offset = 0;
const float boundary_voxel_x  = -volumn_x*(float(M)/2.0f+M_Offset);

const float volumn_y = volumn_x ;
const float inv_volumn_y = 1.0/volumn_y; 
const float boundary_voxel_y  = -volumn_y*(float(N)/2.0f);

const float volumn_z = volumn_x ;
const float inv_volumn_z = 1.0/volumn_z; 
const float boundary_voxel_z = -volumn_z*(float(ZETA)/2.0f); 

// parameters for source 1 
const float Source_z_min = -Source_interval*(float(N_source)/2.0f - 0.5f);
const float Source_y = Offset;
const float Source_x = DSO;

// parameters for Detecotr 1
const float Detector_Ymin = -Detector_pixel_x*(float(R)/2.0f - 0.5f) + Offset; 
const float Detector_Zmin = -Detector_pixel_x*(float(Z_prj)/2.0f - 0.5f); 

const int CT_TOMO = 0; // 0: for CT and conventional Tomo; 1: for half detector TOMO

const int Nviews = 360; 
const float us_rate = 1.00f; 
const float initialAngle= 0.00f ;
const float shiftAngle= 0.0f;

const int GPUNumber = 0;

const float MAX_infi = 1e8;


#include "kernel_IterativeRecon_TBCT.cu"
#include "kernel_Denoise.cu"
#include "host_IterativeRecon_TBCT.c"
#include "host_Denoise_GPU.c"
#include "host_Denoise_CPU.h"

 main(int argc, char ** argv)
{
    cout<<"Hello"<<endl;

/* ******************* Some parameters *********************************************/

    FILE *fp;		    
    char directory[]="/home/huifeng/TBCT/";
    char filename[200];
    char fn[200];
	
	float *F_phantom = new float [M*N*ZETA];
	bzero(F_phantom, sizeof(float)*M*N*ZETA);

	float *G_Prj_Generated = new float [N_source*R*Z_prj*Nviews];
	bzero(G_Prj_Generated, sizeof(float)*N_source*R*Z_prj*Nviews);
	
    /**************** Read Object ***********/
	
	{
        strcpy(filename,directory);        
 		sprintf(fn,"GenerateSLphantom/SLphantom3d_256.dat");
        strcat(filename,fn);        
       	if ( (fp = fopen(filename,"rb")) == NULL )
        {
        	printf("can not open phantom files for main function \n");
           	exit(0);
        }
       	fread(F_phantom, sizeof(float)*M*N*ZETA,1,fp);
       	fclose(fp);
	}

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
        if (cutCheckCmdLineFlag(argc, (const char**)argv, "device"))
                cutilDeviceInit(argc, argv);
        else
                cudaSetDevice( cutGetMaxGflopsDeviceId() );

    /***********produce the forward ******/
    
    printf("Initialising ...\n");
           
	Forward_3D_ray_driven_siddon(F_phantom, G_Prj_Generated);
    
    printf("%d projections have been generated.\n",Nviews);        
    printf("Saving...\n");
    
    // *** Saving  ******//
    
//         strcpy(filename,directory);                
//        	sprintf(fn,"/Projections_HalfDetector/Projection_data_SheppLogan_half_detector.bin");
//         strcat(filename,fn);        
//         if ( (fp = fopen(filename,"wb")) == NULL )
//         {
//                 printf("can not open file to write whole prjections 512 \n");
//                 exit(0);
//         }
//         fwrite(G_Prj_Generated,sizeof(float)*R*Z_prj*Nviews,1,fp);	
//         fclose(fp);    
    

        
    for (int j=0; j<Nviews; j=j+1)
    {       
        strcpy(filename,directory);                        
		sprintf(fn,"/NumericalForwardProjection/phi_%.2f.proj",float(j*us_rate));
        strcat(filename,fn);        
        if ( (fp = fopen(filename,"wb")) == NULL )
        {
                printf("can not open file to write seperate projections \n");
                exit(0);
        } 
        fwrite(G_Prj_Generated+R*Z_prj*N_source*j,sizeof(float)*R*Z_prj*N_source,1,fp);	
        fclose(fp);	        
    }
                
	delete []F_phantom;
    delete []G_Prj_Generated;
	
	return 0;
}