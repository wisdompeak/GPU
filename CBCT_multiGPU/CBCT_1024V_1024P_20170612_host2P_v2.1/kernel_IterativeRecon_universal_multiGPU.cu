__global__ void kernel_add_proj(float *d_a, float *d_b)
{
	int idx = blockDim.x * gridDim.x * blockIdx.y +  blockDim.x * blockIdx.x + threadIdx.x; 
    d_a[idx]=d_a[idx]+d_b[idx];
}

__global__ void kernel_divide_proj(float *h_proj_correction, float *h_proj_data, float *h_proj_sumLen, float *h_proj_weightedLen)
{
	int idx = blockDim.x * gridDim.x * blockIdx.y +  blockDim.x * blockIdx.x + threadIdx.x; 
    
    float temp = h_proj_sumLen[idx];
    
    if ( temp < volumn_z*1e-6)
        h_proj_correction[idx] = 0;
    else
    {        
        h_proj_correction[idx] = (h_proj_data[idx] - h_proj_weightedLen[idx]) / temp *Number_of_Devices;
    }
}


__global__ void forward_ray_driven_3d_kernel_correction_multiGPU(float *d_f , float *d_proj_correction, float *d_proj_data, float sin_theta, float cos_theta, int subPrjIdx, int command)

{
	// d_f: 3D object array;    d_f[i,j,k] = d_f [k*M*N+j*M+i]; 
    // d_proj_data: 2D projection acquired at the angle of t_theta
	// d_proj_correction: 2D projection correction,  (output of this function. i.e. c(i) in the paper)    
        
    int Detector_x_idx = threadIdx.x + blockDim.x * blockIdx.x;
    int Detector_z_idx = blockIdx.y;
    
    int proj_pixel_index = Detector_z_idx * R + Detector_x_idx;
    
	// Source position (X2): coordinate in (x,y,z) system . 
    float vertex_x2_x = Source_x * cos_theta - Source_y * sin_theta;
    float vertex_x2_y = Source_x * sin_theta + Source_y * cos_theta;
	float vertex_x2_z;
    if (FBCT)
        vertex_x2_z = Detector_Zmin + (Z_prj/Number_of_Devices*subPrjIdx+Detector_z_idx) * Detector_pixel_x;  // FBCT geometry
    else
        vertex_x2_z = Source_z;  // CBCT geometry
    
    // Detector element center positions (X1): Coordinate in (x,y,z) system --- 
	float vertex_x1_x = DOD * cos_theta - (Detector_Ymin +  Detector_x_idx * Detector_pixel_x) * sin_theta;
    float vertex_x1_y = DOD * sin_theta + (Detector_Ymin +  Detector_x_idx * Detector_pixel_x) * cos_theta;
    float vertex_x1_z = Detector_Zmin + (Z_prj/Number_of_Devices*subPrjIdx+Detector_z_idx) * Detector_pixel_x;
        //  Notice: in this system, vertex_x1_x < 0 < vertex_x2_x    
            
    float inv_x_diff = 1.0f / (vertex_x2_x - vertex_x1_x);
    float inv_y_diff = 1.0f / (vertex_x2_y - vertex_x1_y);
    float inv_z_diff = 1.0f / (vertex_x2_z - vertex_x1_z);    
    
    /*****************************************/
                
    float alpha_x_min= 0.0f, alpha_y_min= 0.0f, alpha_z_min= 0.0f;
    float alpha_x_max= 0.0f, alpha_y_max= 0.0f, alpha_z_max= 0.0f;
    float alpha_min= 0.0f, alpha_max= 0.0f;
        
	int i_min=0, j_min=0, k_min=0;
	int i_max=0, j_max=0, k_max=0;
    int i=0, j=0, k=0;
    int voxel_i=0, voxel_j=0, voxel_k=0;
    
	float alpha_x=0.0f, alpha_y=0.0f, alpha_z=0.0f;  
    float one_ray_sum = 0.0f;
    float one_ray_length = 0.0f; 

    float alpha_c= 0.0f;
    float d_x1_x2= 0.0f;

	int N_total_sec=0; 
    
    int next_alpha_index;

            
	/**** Step 1 :find out alpha_min, alpha_max ********/

    if ( (vertex_x1_x == vertex_x2_x) || (vertex_x1_y == vertex_x2_y)  )  //Note: You may rotate the angle to avoid this happening
    {    
        d_proj_correction[proj_pixel_index] = 0.0f ;
//         printf("Vertical or Horizontal line occurs! Detector_x_idx:%d, Detector_z_idx:%d/n", Detector_x_idx,Detector_z_idx);
//         assert(0);
    }    

    else //  if ( (vertex_x1_x != vertex_x2_x) && (vertex_x1_y != vertex_x2_y) )
    {

        alpha_min = (boundary_voxel_x + volumn_x*0 - vertex_x1_x )* inv_x_diff; //(9)
        alpha_max = (boundary_voxel_x + volumn_x*M - vertex_x1_x )* inv_x_diff;
        // Notice: it is still unsure here which one is the parametric value of the first intersection point of the ray with the x-plane
        // It depends on whether source or detector lies on the left side of the reconstruction region at this time

        alpha_x_min = fmin(alpha_min, alpha_max);   //(5)
        alpha_x_max = fmax(alpha_min, alpha_max );  //(6) 
                
        alpha_min = (boundary_voxel_y + volumn_y*0 - vertex_x1_y )* inv_y_diff;
        alpha_max = (boundary_voxel_y + volumn_y*N - vertex_x1_y )* inv_y_diff;

        alpha_y_min = fmin(alpha_min, alpha_max);   //(7)
        alpha_y_max = fmax(alpha_min, alpha_max );  //(8)
        
        if (fabs(vertex_x2_z - vertex_x1_z) < volumn_z*1e-6)    // in case x1 and x2 are at the same z position
        {
            alpha_min = -MAX_infi;
            alpha_max = MAX_infi;
//             printf("Same horizontal plane occurs! Detector_z_idx:%d/n", Detector_z_idx);            
//             assert(0);
        }
        else
        {
            alpha_min = (boundary_voxel_z + volumn_z*0 - vertex_x1_z )* inv_z_diff;
            alpha_max = (boundary_voxel_z + volumn_z*ZETA - vertex_x1_z )* inv_z_diff;
        }
        
        alpha_z_min = fmin(alpha_min, alpha_max);   
        alpha_z_max = fmax(alpha_min, alpha_max );  

        // alpha_min / alpha_max reused 
        alpha_min = fmax(fmax(alpha_x_min, alpha_y_min), fmax(alpha_y_min, alpha_z_min)); //(3)
        // i.e. alpha_min = fmax(alpha_x_min,alpha_y_min,alpha_z_min)
        // it indicates the point where the path interacts with the near boundary of reconstruction region        

        alpha_max = fmin(fmin(alpha_x_max, alpha_y_max), fmin(alpha_y_max, alpha_z_max)); //(4)
        // i.e. alpha_max = fmin(alpha_x_max,alpha_y_max,alpha_z_max)
        // it indicates the point where the path last interacts with the far boundary of reconstruction region        
        
        /********Step 2,3: Find i_max, i_min***************/
        
     if (alpha_max <= alpha_min)   // It means no interaction of the ray and the volume
                d_proj_correction[proj_pixel_index] = 0.0f ;
  
	 else 
     {
			// X direction 
			if (vertex_x1_x < vertex_x2_x)
			{	
				if (alpha_min == alpha_x_min)
					i_min = 1;      //(11)
				else //if (alpha_min != alpha_x_min)
					i_min =  floor(( alpha_min*(vertex_x2_x - vertex_x1_x) + vertex_x1_x - boundary_voxel_x)*inv_volumn_x) + 1 ;
                                    //(12)
                     /* Note: i_min is the index of the 1st x plane where the path interacts inside the reconstruction region
                      * It is not the index of alpha_x_min
                      */                
				if (alpha_max == alpha_x_max)
					i_max = M;      //(13)
				else //if (alpha_max != alpha_x_max)
					i_max =  floor(( alpha_max*(vertex_x2_x - vertex_x1_x) + vertex_x1_x - boundary_voxel_x)*inv_volumn_x) ;
                                    //(14)
                     // Note: i_max is the index of the last x plane where the path interacts with the reconstruction region (inside or boundary)                      
			}	
			else //if (vertex_x1_x > vertex_x2_x)
			{	
				if (alpha_min == alpha_x_min)
					i_max = M-1;    //(15)
				else //if (alpha_min != alpha_x_min)
					i_max =  floor(( alpha_min*(vertex_x2_x - vertex_x1_x) + vertex_x1_x - boundary_voxel_x)*inv_volumn_x) ;				
                                    //(16)
				if (alpha_max == alpha_x_max)
					i_min = 0;      //(17)
				else //if (alpha_max != alpha_x_max)
					i_min =  floor(( alpha_max*(vertex_x2_x - vertex_x1_x) + vertex_x1_x - boundary_voxel_x)*inv_volumn_x) + 1 ;
                                    //(18)
			}	
            // Note: overall, i_min is the most left x-plane, i_max the most right x-plane,
            // and the initial point (the first interacted position on the boundary) NOT included.            
               
			//Y direction 
			if (vertex_x1_y < vertex_x2_y)
			{	
				if (alpha_min == alpha_y_min)
					j_min = 1; 
				else //f (alpha_min != alpha_y_min)
					j_min =  floor(( alpha_min*(vertex_x2_y - vertex_x1_y) + vertex_x1_y - boundary_voxel_y)*inv_volumn_y) + 1 ;
				
				if (alpha_max == alpha_y_max)
					j_max = N; 
				else //if (alpha_max != alpha_y_max)
					j_max =  floor(( alpha_max*(vertex_x2_y - vertex_x1_y) + vertex_x1_y - boundary_voxel_y)*inv_volumn_y) ;

			}	
			else //if (vertex_x1_y > vertex_x2_y)
			{	
				if (alpha_min == alpha_y_min)
					j_max = N-1; 
				else //if (alpha_min != alpha_y_min)
					j_max =  floor(( alpha_min*(vertex_x2_y - vertex_x1_y) + vertex_x1_y - boundary_voxel_y )*inv_volumn_y) ;
				
				if (alpha_max == alpha_y_max)
					j_min = 0; 
				else //if (alpha_max != alpha_y_max)
					j_min =  floor(( alpha_max*(vertex_x2_y - vertex_x1_y) + vertex_x1_y - boundary_voxel_y )*inv_volumn_y) + 1 ;

			}	
            // Note: overall, j_min is the most bottom y-plane, j_max the most top y-plane,
            // and the initial point (the first interacted position on the boundary) NOT included.
            
			//Z direction 
            if (fabs(vertex_x1_z-vertex_x2_z)<volumn_z*1e-6 )  
            {
					k_min =  floor(( vertex_x1_z - boundary_voxel_z )*inv_volumn_z) + 1 ;                
					k_max =  floor(( vertex_x1_z - boundary_voxel_z )*inv_volumn_z) ;    
            }
            else if (vertex_x1_z < vertex_x2_z)
			{	
				if (alpha_min == alpha_z_min)
					k_min = 1; 
				else //if (alpha_min != alpha_z_min)
					k_min =  floor(( alpha_min*(vertex_x2_z - vertex_x1_z) + vertex_x1_z  - boundary_voxel_z )*inv_volumn_z) + 1 ;
				
				if (alpha_max == alpha_z_max)
					k_max = ZETA; 
				else //if (alpha_max != alpha_z_max)
					k_max =  floor(( alpha_max*(vertex_x2_z - vertex_x1_z) + vertex_x1_z  - boundary_voxel_z )*inv_volumn_z) ;

			}	
			else //if (vertex_x1_z > vertex_x2_z)
			{	
				if (alpha_min == alpha_z_min)
					k_max = ZETA-1; 
				else //if (alpha_min != alpha_z_min)
					k_max =  floor(( alpha_min*(vertex_x2_z - vertex_x1_z) + vertex_x1_z  - boundary_voxel_z )*inv_volumn_z) ;
				
				if (alpha_max == alpha_z_max)
					k_min = 0; 
				else //if (alpha_max != alpha_z_max)
					k_min =  floor(( alpha_max*(vertex_x2_z - vertex_x1_z) + vertex_x1_z  -boundary_voxel_z )*inv_volumn_z) + 1 ;

			}	
            
        /************ initialization (i,j,k) (alpha_x_1,alpha_y_1,alpha_z_1)**************************/            
        // Note: (i,j,k) is the current x,y,z plane index (@ the initial point at the boundary)
        // Note: (alpha_x,alpha_y,alpha_z) is the next x,y,z plane to go.
            
		N_total_sec = i_max - i_min + 1 + j_max - j_min +1 + k_max-k_min +1; 
            // i.e. N_p (25)

		if (vertex_x1_x < vertex_x2_x)
        {
 			alpha_x = (volumn_x * i_min + boundary_voxel_x - vertex_x1_x )* inv_x_diff;
            i = i_min - 1;   
        }        
		else if (vertex_x1_x > vertex_x2_x) 			
        {
            alpha_x = (volumn_x * i_max + boundary_voxel_x - vertex_x1_x )* inv_x_diff;
            i = i_max + 1;
        }
            // Note: alpha_x_1 is the intersection where the path hit the 1st x plane inside the recon region
		
		if (vertex_x1_y < vertex_x2_y)
        {            
 			alpha_y = (volumn_y * j_min + boundary_voxel_y - vertex_x1_y )* inv_y_diff;
            j = j_min - 1;
        }
		else if (vertex_x1_y > vertex_x2_y)
        {
 			alpha_y = (volumn_y * j_max + boundary_voxel_y - vertex_x1_y )* inv_y_diff;
            j = j_max + 1;
        }            
            // Note: alpha_y_1 is the intersection where the path hit the 1st y plane inside the recon region
	
		if (fabs(vertex_x1_z-vertex_x2_z)<volumn_z*1e-6 )  
        {
            alpha_z = MAX_infi;
            k = k_min-1;
        }
        else if (vertex_x1_z < vertex_x2_z)
        {
 			alpha_z = (volumn_z * k_min + boundary_voxel_z - vertex_x1_z )* inv_z_diff;
            k = k_min - 1;
        }            
		else if (vertex_x1_z > vertex_x2_z)
        {
            alpha_z = (volumn_z * k_max + boundary_voxel_z - vertex_x1_z )* inv_z_diff;
            k = k_max + 1;
        }        
                                     
        /************ initialization (voxel_i,voxel_j,voxel_k) **************************/            
        // Note: (voxel_i,voxel_j,voxel_k) is the current x,y,z voxel index (@ the initial point at the boundary)
        
        if (vertex_x1_x < vertex_x2_x)
            voxel_i = i_min-1;
        else 
            voxel_i = i_max;
                
        if (vertex_x1_y < vertex_x2_y)
            voxel_j = j_min-1;
        else 
            voxel_j = j_max;
               
        if (fabs(vertex_x1_z-vertex_x2_z) < volumn_z*1e-6)
            voxel_k = k_min-1;
        else if (vertex_x1_z < vertex_x2_z)
            voxel_k = k_min-1;
        else 
            voxel_k = k_max;                              
        
        /***************** Updating alpha_x, alpha_y, alpha_z, ************************/
        
        // Note: (alpha_x, alpha_y, alpha_z) the intersection where the path hit the next (i.e. 1st here ) x/y/z plane inside the recon
        
        d_x1_x2 = sqrt((vertex_x2_x-vertex_x1_x)*(vertex_x2_x-vertex_x1_x) + (vertex_x2_y-vertex_x1_y)*(vertex_x2_y - vertex_x1_y) + (vertex_x2_z-vertex_x1_z)*(vertex_x2_z-vertex_x1_z) );
      	                
        alpha_c = alpha_min;    // intersection where the path hit the 1st plane at the boundary of recon region

        // Note : (i,j,k) is the (x,y,z) plane index of the current intersection (with a certain plane)
        // If i or j or k should not be an integer, then its predecessor (along the ray)
        
        while (alpha_max - alpha_c > 1e-16)
       	{
            
          if ((voxel_i > M-1)||(voxel_i <0) || (voxel_j > N-1)||(voxel_j <0) || (voxel_k > ZETA-1)||(voxel_k <0))
          {
                alpha_c = alpha_max +1;  // to terminate the loop
          }         
          else
          {
		
  			if ( (alpha_x < alpha_y) && (alpha_x < alpha_z))
                            // alpha_x is the nearest, so update alpha_x
            {
				one_ray_length += d_x1_x2 * (alpha_x - alpha_c);  //(30)
                one_ray_sum += d_x1_x2 * (alpha_x - alpha_c) * d_f[voxel_k*M*N + voxel_j*M + voxel_i];
                                                                //(31)		              
				alpha_c = alpha_x;          //(33)   Update the current location
				N_total_sec = N_total_sec - 1; 
                
                if (vertex_x1_x < vertex_x2_x)
                {
					i++;
                    voxel_i++;
                    next_alpha_index = i+1;
                }
                if (vertex_x1_x > vertex_x2_x)
                {
                    i--;      //(29)   
                    voxel_i--;
                    next_alpha_index = i-1;
                }
                alpha_x = (volumn_x * next_alpha_index + boundary_voxel_x - vertex_x1_x )* inv_x_diff;
                
           	}
            
			else if ( (alpha_y < alpha_x) && (alpha_y < alpha_z) )
                            // alpha_y is the nearest, so update alpha_y
           	{                        
				one_ray_length += d_x1_x2 * (alpha_y - alpha_c);
                one_ray_sum += d_x1_x2 * (alpha_y - alpha_c) * d_f[voxel_k*M*N + voxel_j*M + voxel_i];
				
                alpha_c = alpha_y; 
                N_total_sec = N_total_sec -1;                
                
                if (vertex_x1_y < vertex_x2_y)
                {
					j++;
                    voxel_j++;
                    next_alpha_index = j+1;
                }
                else if (vertex_x1_y > vertex_x2_y)
                {
					j--;
                    voxel_j--;
                    next_alpha_index = j-1;
                }   
                alpha_y = (volumn_y * next_alpha_index + boundary_voxel_y - vertex_x1_y )* inv_y_diff;                
           	}
            
			else if ( (alpha_z < alpha_x) && (alpha_z < alpha_y) )
                        // alpha_z is the nearest, so update alpha_z                
            {				
				one_ray_length += d_x1_x2 * (alpha_z - alpha_c);
                one_ray_sum += d_x1_x2 * (alpha_z - alpha_c) * d_f[voxel_k*M*N + voxel_j*M + voxel_i];
				
                alpha_c = alpha_z; 
                N_total_sec = N_total_sec -1;                
                
                if (vertex_x1_z < vertex_x2_z)
                {
					k++;
                    voxel_k++;
                    next_alpha_index = k+1;
                }
                else if (vertex_x1_z > vertex_x2_z)
                {
					k--;
                    voxel_k--;
                    next_alpha_index = k-1;
                }
                alpha_z = (volumn_z * next_alpha_index + boundary_voxel_z - vertex_x1_z )* inv_z_diff;
                
            }
		     
			else if ( (alpha_x == alpha_y) && (alpha_x < alpha_z) )
                        //x = y < z
            {        

				one_ray_length += d_x1_x2 * (alpha_x - alpha_c);  //(30)
                one_ray_sum += d_x1_x2 * (alpha_x - alpha_c) * d_f[voxel_k*M*N + voxel_j*M + voxel_i];  //(31)	
                                                                	              
				alpha_c = alpha_x;          //(33)   Update the current location
				N_total_sec = N_total_sec - 2; 
                                  
                if (vertex_x1_x < vertex_x2_x)
                {
					i = i + 1;
                    voxel_i = voxel_i +1;
                    next_alpha_index = i+1;
                }
                if (vertex_x1_x > vertex_x2_x)
                {
                    i = i - 1;      //(29)   
                    voxel_i = voxel_i-1;
                    next_alpha_index = i-1;                                        
                }
                alpha_x = (volumn_x * next_alpha_index + boundary_voxel_x - vertex_x1_x )* inv_x_diff;
                
                if (vertex_x1_y < vertex_x2_y)
                {
					j = j + 1;
                    voxel_j = voxel_j+1;
                    next_alpha_index = j+1; 
                }
                else if (vertex_x1_y > vertex_x2_y)
                {
					j = j - 1;
                    voxel_j = voxel_j-1;
                    next_alpha_index = j-1;
                }                
                alpha_y = (volumn_y * next_alpha_index + boundary_voxel_y - vertex_x1_y )* inv_y_diff;
                
            }
            
          	else if ( (alpha_x == alpha_z) && (alpha_x < alpha_y))// && (sphere_range<=1.0f) )
                        // x = z < y;
            {                      
				one_ray_length += d_x1_x2 * (alpha_x - alpha_c);  //(30)
                one_ray_sum += d_x1_x2 * (alpha_x - alpha_c) * d_f[voxel_k*M*N + voxel_j*M + voxel_i];  //(31)	
                                                                	              
				alpha_c = alpha_x;          //(33)   Update the current location
				N_total_sec = N_total_sec - 2; 

                if (vertex_x1_x < vertex_x2_x)
                {
					i = i + 1;
                    voxel_i = voxel_i +1;
                    next_alpha_index = i+1;                    
                }
                if (vertex_x1_x > vertex_x2_x)
                {
                    i = i - 1;      //(29)   
                    voxel_i = voxel_i-1;
                    next_alpha_index = i-1;                 
                }
                alpha_x = (volumn_x * next_alpha_index + boundary_voxel_x - vertex_x1_x )* inv_x_diff;
                
                if (vertex_x1_z < vertex_x2_z)
                {
					k = k + 1;
                    voxel_k = voxel_k+1;
                    next_alpha_index = k+1;
                }
                else if (vertex_x1_z > vertex_x2_z)
                {
					k = k - 1;
                    voxel_k = voxel_k-1;
                    next_alpha_index = k-1;
                }                
                alpha_z = (volumn_z * next_alpha_index + boundary_voxel_z - vertex_x1_z )* inv_z_diff;

            }
            
			else if ( (alpha_y == alpha_z) && (alpha_y < alpha_x))// && (sphere_range<=1.0f) )
                      	// y = z < x        	
            {            	
				one_ray_length += d_x1_x2 * (alpha_y - alpha_c);
                one_ray_sum += d_x1_x2 * (alpha_y - alpha_c) * d_f[voxel_k*M*N + voxel_j*M + voxel_i];
				
                alpha_c = alpha_y; 
                N_total_sec = N_total_sec -2;                
                
                if (vertex_x1_y < vertex_x2_y)
                {
					j = j + 1;
                    voxel_j = voxel_j+1;
                    next_alpha_index = j+1;
                }
                else if (vertex_x1_y > vertex_x2_y)
                {
					j = j - 1;
                    voxel_j = voxel_j-1;
                    next_alpha_index = j-1;
                }   
                alpha_y = (volumn_y * next_alpha_index + boundary_voxel_y - vertex_x1_y )* inv_y_diff;
                
                if (vertex_x1_z < vertex_x2_z)
                {
					k = k + 1;
                    voxel_k = voxel_k+1;
                    next_alpha_index = k+1;
                }
                else if (vertex_x1_z > vertex_x2_z)
                {
					k = k - 1;
                    voxel_k = voxel_k-1;
                    next_alpha_index = k-1;
                }  
                alpha_z = (volumn_z * next_alpha_index + boundary_voxel_z - vertex_x1_z )* inv_z_diff;
                
            }
			
          	else if ( (alpha_x == alpha_z) && (alpha_x == alpha_y))// && (sphere_range<=1.0f) )
                        // x=y=z            
            {
				one_ray_length += d_x1_x2 * (alpha_x - alpha_c);  //(30)
                one_ray_sum += d_x1_x2 * (alpha_x - alpha_c) * d_f[voxel_k*M*N + voxel_j*M + voxel_i];  //(31)	
                                                                	              
				alpha_c = alpha_x;          //(33)   Update the current location
				N_total_sec = N_total_sec - 3; 

                if (vertex_x1_x < vertex_x2_x)
                {
					i = i + 1;
                    voxel_i = voxel_i +1;
                    next_alpha_index = i+1;
                }
                if (vertex_x1_x > vertex_x2_x)
                {
                    i = i - 1;      //(29)   
                    voxel_i = voxel_i-1;
                    next_alpha_index = i-1;                   
                }
                alpha_x = (volumn_x * next_alpha_index + boundary_voxel_x - vertex_x1_x )* inv_x_diff;
                
                if (vertex_x1_y < vertex_x2_y)
                {
					j = j + 1;
                    voxel_j = voxel_j+1;
                    next_alpha_index = j+1;
                }
                else if (vertex_x1_y > vertex_x2_y)
                {
					j = j - 1;
                    voxel_j = voxel_j-1;
                    next_alpha_index = j-1;
                }   
                alpha_y = (volumn_y * next_alpha_index + boundary_voxel_y - vertex_x1_y )* inv_y_diff;
                
                if (vertex_x1_z < vertex_x2_z)
                {
					k = k + 1;
                    voxel_k = voxel_k+1;
                    next_alpha_index = k+1;
                }
                else if (vertex_x1_z > vertex_x2_z)
                {
					k = k - 1;
                    voxel_k = voxel_k-1;
                    next_alpha_index = k-1;
                }  
                alpha_z = (volumn_z * next_alpha_index + boundary_voxel_z - vertex_x1_z )* inv_z_diff;
           	}
          }
       	}// end while
        
        
        if (one_ray_length < volumn_z*1e-6)            
            d_proj_correction[proj_pixel_index] = 0.0;
        else
        {
            if (command == 0)
                d_proj_correction[proj_pixel_index] = one_ray_sum; // forward operator
            
            else if (command == 1)                
                d_proj_correction[proj_pixel_index] = (d_proj_data[proj_pixel_index] - one_ray_sum)/one_ray_length; 
                                                                    // projection correction (for SART)
        }
                
     }//else if 
   }//else if
   
//    __syncthreads();
    
}

__global__ void forward_ray_driven_3d_kernel_correction_separate(float *d_f , float *d_proj_sumLen, float *d_proj_weightedLen, float sin_theta, float cos_theta)

{
	// d_f: 3D object array;    d_f[i,j,k] = d_f [k*M*N+j*M+i]; 
    // d_proj_data: 2D projection acquired at the angle of t_theta
	// d_proj_sumLen: 2D projection correction,  (output of this function. i.e. c(i) in the paper)    
        
    int Detector_x_idx = threadIdx.x + blockDim.x * blockIdx.x;
    int Detector_z_idx = blockIdx.y;
    
    int proj_pixel_index = Detector_z_idx * R + Detector_x_idx;           
    
	// Source position (X2): coordinate in (x,y,z) system . 
    float vertex_x2_x = Source_x * cos_theta - Source_y * sin_theta;
    float vertex_x2_y = Source_x * sin_theta + Source_y * cos_theta;
	float vertex_x2_z;
    if (FBCT)
        vertex_x2_z = Detector_Zmin + Detector_z_idx * Detector_pixel_x;  // FBCT geometry
    else
        vertex_x2_z = Source_z;  // CBCT geometry
    
    // Detector element center positions (X1): Coordinate in (x,y,z) system --- 
	float vertex_x1_x = DOD * cos_theta - (Detector_Ymin +  Detector_x_idx * Detector_pixel_x) * sin_theta;
    float vertex_x1_y = DOD * sin_theta + (Detector_Ymin +  Detector_x_idx * Detector_pixel_x) * cos_theta;
    float vertex_x1_z = Detector_Zmin + Detector_z_idx * Detector_pixel_x;
        //  Notice: in this system, vertex_x1_x < 0 < vertex_x2_x    
            
    float inv_x_diff = 1.0f / (vertex_x2_x - vertex_x1_x);
    float inv_y_diff = 1.0f / (vertex_x2_y - vertex_x1_y);
    float inv_z_diff = 1.0f / (vertex_x2_z - vertex_x1_z);    
    
    /*****************************************/
                
    float alpha_x_min= 0.0f, alpha_y_min= 0.0f, alpha_z_min= 0.0f;
    float alpha_x_max= 0.0f, alpha_y_max= 0.0f, alpha_z_max= 0.0f;
    float alpha_min= 0.0f, alpha_max= 0.0f;
        
	int i_min=0, j_min=0, k_min=0;
	int i_max=0, j_max=0, k_max=0;
    int i=0, j=0, k=0;
    int voxel_i=0, voxel_j=0, voxel_k=0;
    
	float alpha_x=0.0f, alpha_y=0.0f, alpha_z=0.0f;  
    float one_ray_sum = 0.0f;
    float one_ray_length = 0.0f; 

    float alpha_c= 0.0f;
    float d_x1_x2= 0.0f;

	int N_total_sec=0; 
    
    int next_alpha_index;

            
	/**** Step 1 :find out alpha_min, alpha_max ********/

    if ( (vertex_x1_x == vertex_x2_x) || (vertex_x1_y == vertex_x2_y)  )  //Note: You may rotate the angle to avoid this happening
    {    
        d_proj_weightedLen[proj_pixel_index] = 0.0f ;
        d_proj_sumLen[proj_pixel_index]=0.0f;
//         printf("Vertical or Horizontal line occurs! Detector_x_idx:%d, Detector_z_idx:%d/n", Detector_x_idx,Detector_z_idx);
//         assert(0);
    }    

    else //  if ( (vertex_x1_x != vertex_x2_x) && (vertex_x1_y != vertex_x2_y) )
    {

        alpha_min = (boundary_voxel_x + volumn_x*0 - vertex_x1_x )* inv_x_diff; //(9)
        alpha_max = (boundary_voxel_x + volumn_x*M - vertex_x1_x )* inv_x_diff;
        // Notice: it is still unsure here which one is the parametric value of the first intersection point of the ray with the x-plane
        // It depends on whether source or detector lies on the left side of the reconstruction region at this time

        alpha_x_min = fmin(alpha_min, alpha_max);   //(5)
        alpha_x_max = fmax(alpha_min, alpha_max );  //(6) 
                
        alpha_min = (boundary_voxel_y + volumn_y*0 - vertex_x1_y )* inv_y_diff;
        alpha_max = (boundary_voxel_y + volumn_y*N - vertex_x1_y )* inv_y_diff;

        alpha_y_min = fmin(alpha_min, alpha_max);   //(7)
        alpha_y_max = fmax(alpha_min, alpha_max );  //(8)
        
        if (fabs(vertex_x2_z - vertex_x1_z) < volumn_z*1e-6)    // in case x1 and x2 are at the same z position
        {
            alpha_min = -MAX_infi;
            alpha_max = MAX_infi;
//             printf("Same horizontal plane occurs! Detector_z_idx:%d/n", Detector_z_idx);            
//             assert(0);
        }
        else
        {
            alpha_min = (boundary_voxel_z + volumn_z*0 - vertex_x1_z )* inv_z_diff;
            alpha_max = (boundary_voxel_z + volumn_z*ZETA - vertex_x1_z )* inv_z_diff;
        }
        
        alpha_z_min = fmin(alpha_min, alpha_max);   
        alpha_z_max = fmax(alpha_min, alpha_max );  

        // alpha_min / alpha_max reused 
        alpha_min = fmax(fmax(alpha_x_min, alpha_y_min), fmax(alpha_y_min, alpha_z_min)); //(3)
        // i.e. alpha_min = fmax(alpha_x_min,alpha_y_min,alpha_z_min)
        // it indicates the point where the path interacts with the near boundary of reconstruction region        

        alpha_max = fmin(fmin(alpha_x_max, alpha_y_max), fmin(alpha_y_max, alpha_z_max)); //(4)
        // i.e. alpha_max = fmin(alpha_x_max,alpha_y_max,alpha_z_max)
        // it indicates the point where the path last interacts with the far boundary of reconstruction region        
        
        /********Step 2,3: Find i_max, i_min***************/
        
     if (alpha_max <= alpha_min)   // It means no interaction of the ray and the volume
     {
        d_proj_weightedLen[proj_pixel_index] = 0.0f ;
        d_proj_sumLen[proj_pixel_index]=0.0f;
     }
  
	 else 
     {
			// X direction 
			if (vertex_x1_x < vertex_x2_x)
			{	
				if (alpha_min == alpha_x_min)
					i_min = 1;      //(11)
				else //if (alpha_min != alpha_x_min)
					i_min =  floor(( alpha_min*(vertex_x2_x - vertex_x1_x) + vertex_x1_x - boundary_voxel_x)*inv_volumn_x) + 1 ;
                                    //(12)
                     /* Note: i_min is the index of the 1st x plane where the path interacts inside the reconstruction region
                      * It is not the index of alpha_x_min
                      */                
				if (alpha_max == alpha_x_max)
					i_max = M;      //(13)
				else //if (alpha_max != alpha_x_max)
					i_max =  floor(( alpha_max*(vertex_x2_x - vertex_x1_x) + vertex_x1_x - boundary_voxel_x)*inv_volumn_x) ;
                                    //(14)
                     // Note: i_max is the index of the last x plane where the path interacts with the reconstruction region (inside or boundary)                      
			}	
			else //if (vertex_x1_x > vertex_x2_x)
			{	
				if (alpha_min == alpha_x_min)
					i_max = M-1;    //(15)
				else //if (alpha_min != alpha_x_min)
					i_max =  floor(( alpha_min*(vertex_x2_x - vertex_x1_x) + vertex_x1_x - boundary_voxel_x)*inv_volumn_x) ;				
                                    //(16)
				if (alpha_max == alpha_x_max)
					i_min = 0;      //(17)
				else //if (alpha_max != alpha_x_max)
					i_min =  floor(( alpha_max*(vertex_x2_x - vertex_x1_x) + vertex_x1_x - boundary_voxel_x)*inv_volumn_x) + 1 ;
                                    //(18)
			}	
            // Note: overall, i_min is the most left x-plane, i_max the most right x-plane,
            // and the initial point (the first interacted position on the boundary) NOT included.            
               
			//Y direction 
			if (vertex_x1_y < vertex_x2_y)
			{	
				if (alpha_min == alpha_y_min)
					j_min = 1; 
				else //f (alpha_min != alpha_y_min)
					j_min =  floor(( alpha_min*(vertex_x2_y - vertex_x1_y) + vertex_x1_y - boundary_voxel_y)*inv_volumn_y) + 1 ;
				
				if (alpha_max == alpha_y_max)
					j_max = N; 
				else //if (alpha_max != alpha_y_max)
					j_max =  floor(( alpha_max*(vertex_x2_y - vertex_x1_y) + vertex_x1_y - boundary_voxel_y)*inv_volumn_y) ;

			}	
			else //if (vertex_x1_y > vertex_x2_y)
			{	
				if (alpha_min == alpha_y_min)
					j_max = N-1; 
				else //if (alpha_min != alpha_y_min)
					j_max =  floor(( alpha_min*(vertex_x2_y - vertex_x1_y) + vertex_x1_y - boundary_voxel_y )*inv_volumn_y) ;
				
				if (alpha_max == alpha_y_max)
					j_min = 0; 
				else //if (alpha_max != alpha_y_max)
					j_min =  floor(( alpha_max*(vertex_x2_y - vertex_x1_y) + vertex_x1_y - boundary_voxel_y )*inv_volumn_y) + 1 ;

			}	
            // Note: overall, j_min is the most bottom y-plane, j_max the most top y-plane,
            // and the initial point (the first interacted position on the boundary) NOT included.
            
			//Z direction 
            if (fabs(vertex_x1_z-vertex_x2_z)<volumn_z*1e-6 )  
            {
					k_min =  floor(( vertex_x1_z - boundary_voxel_z )*inv_volumn_z) + 1 ;                
					k_max =  floor(( vertex_x1_z - boundary_voxel_z )*inv_volumn_z) ;    
            }
            else if (vertex_x1_z < vertex_x2_z)
			{	
				if (alpha_min == alpha_z_min)
					k_min = 1; 
				else //if (alpha_min != alpha_z_min)
					k_min =  floor(( alpha_min*(vertex_x2_z - vertex_x1_z) + vertex_x1_z  - boundary_voxel_z )*inv_volumn_z) + 1 ;
				
				if (alpha_max == alpha_z_max)
					k_max = ZETA; 
				else //if (alpha_max != alpha_z_max)
					k_max =  floor(( alpha_max*(vertex_x2_z - vertex_x1_z) + vertex_x1_z  - boundary_voxel_z )*inv_volumn_z) ;

			}	
			else //if (vertex_x1_z > vertex_x2_z)
			{	
				if (alpha_min == alpha_z_min)
					k_max = ZETA-1; 
				else //if (alpha_min != alpha_z_min)
					k_max =  floor(( alpha_min*(vertex_x2_z - vertex_x1_z) + vertex_x1_z  - boundary_voxel_z )*inv_volumn_z) ;
				
				if (alpha_max == alpha_z_max)
					k_min = 0; 
				else //if (alpha_max != alpha_z_max)
					k_min =  floor(( alpha_max*(vertex_x2_z - vertex_x1_z) + vertex_x1_z  -boundary_voxel_z )*inv_volumn_z) + 1 ;

			}	
            
        /************ initialization (i,j,k) (alpha_x_1,alpha_y_1,alpha_z_1)**************************/            
        // Note: (i,j,k) is the current x,y,z plane index (@ the initial point at the boundary)
        // Note: (alpha_x,alpha_y,alpha_z) is the next x,y,z plane to go.
            
		N_total_sec = i_max - i_min + 1 + j_max - j_min +1 + k_max-k_min +1; 
            // i.e. N_p (25)

		if (vertex_x1_x < vertex_x2_x)
        {
 			alpha_x = (volumn_x * i_min + boundary_voxel_x - vertex_x1_x )* inv_x_diff;
            i = i_min - 1;   
        }        
		else if (vertex_x1_x > vertex_x2_x) 			
        {
            alpha_x = (volumn_x * i_max + boundary_voxel_x - vertex_x1_x )* inv_x_diff;
            i = i_max + 1;
        }
            // Note: alpha_x_1 is the intersection where the path hit the 1st x plane inside the recon region
		
		if (vertex_x1_y < vertex_x2_y)
        {            
 			alpha_y = (volumn_y * j_min + boundary_voxel_y - vertex_x1_y )* inv_y_diff;
            j = j_min - 1;
        }
		else if (vertex_x1_y > vertex_x2_y)
        {
 			alpha_y = (volumn_y * j_max + boundary_voxel_y - vertex_x1_y )* inv_y_diff;
            j = j_max + 1;
        }            
            // Note: alpha_y_1 is the intersection where the path hit the 1st y plane inside the recon region
	
		if (fabs(vertex_x1_z-vertex_x2_z)<volumn_z*1e-6 )  
        {
            alpha_z = MAX_infi;
            k = k_min-1;
        }
        else if (vertex_x1_z < vertex_x2_z)
        {
 			alpha_z = (volumn_z * k_min + boundary_voxel_z - vertex_x1_z )* inv_z_diff;
            k = k_min - 1;
        }            
		else if (vertex_x1_z > vertex_x2_z)
        {
            alpha_z = (volumn_z * k_max + boundary_voxel_z - vertex_x1_z )* inv_z_diff;
            k = k_max + 1;
        }        
                                     
        /************ initialization (voxel_i,voxel_j,voxel_k) **************************/            
        // Note: (voxel_i,voxel_j,voxel_k) is the current x,y,z voxel index (@ the initial point at the boundary)
        
        if (vertex_x1_x < vertex_x2_x)
            voxel_i = i_min-1;
        else 
            voxel_i = i_max;
                
        if (vertex_x1_y < vertex_x2_y)
            voxel_j = j_min-1;
        else 
            voxel_j = j_max;
               
        if (fabs(vertex_x1_z-vertex_x2_z) < volumn_z*1e-6)
            voxel_k = k_min-1;
        else if (vertex_x1_z < vertex_x2_z)
            voxel_k = k_min-1;
        else 
            voxel_k = k_max;                              
        
        /***************** Updating alpha_x, alpha_y, alpha_z, ************************/
        
        // Note: (alpha_x, alpha_y, alpha_z) the intersection where the path hit the next (i.e. 1st here ) x/y/z plane inside the recon
        
        d_x1_x2 = sqrt((vertex_x2_x-vertex_x1_x)*(vertex_x2_x-vertex_x1_x) + (vertex_x2_y-vertex_x1_y)*(vertex_x2_y - vertex_x1_y) + (vertex_x2_z-vertex_x1_z)*(vertex_x2_z-vertex_x1_z) );
      	                
        alpha_c = alpha_min;    // intersection where the path hit the 1st plane at the boundary of recon region

        // Note : (i,j,k) is the (x,y,z) plane index of the current intersection (with a certain plane)
        // If i or j or k should not be an integer, then its predecessor (along the ray)
        
        while (alpha_max - alpha_c > 1e-16)
       	{
            
          if ((voxel_i > M-1)||(voxel_i <0) || (voxel_j > N-1)||(voxel_j <0) || (voxel_k > ZETA-1)||(voxel_k <0))
          {
                alpha_c = alpha_max +1;  // to terminate the loop
          }         
          else
          {
		
  			if ( (alpha_x < alpha_y) && (alpha_x < alpha_z))
                            // alpha_x is the nearest, so update alpha_x
            {
				one_ray_length += d_x1_x2 * (alpha_x - alpha_c);  //(30)
                one_ray_sum += d_x1_x2 * (alpha_x - alpha_c) * d_f[voxel_k*M*N + voxel_j*M + voxel_i];
                                                                //(31)		              
				alpha_c = alpha_x;          //(33)   Update the current location
				N_total_sec = N_total_sec - 1; 
                
                if (vertex_x1_x < vertex_x2_x)
                {
					i++;
                    voxel_i++;
                    next_alpha_index = i+1;
                }
                if (vertex_x1_x > vertex_x2_x)
                {
                    i--;      //(29)   
                    voxel_i--;
                    next_alpha_index = i-1;
                }
                alpha_x = (volumn_x * next_alpha_index + boundary_voxel_x - vertex_x1_x )* inv_x_diff;
                
           	}
            
			else if ( (alpha_y < alpha_x) && (alpha_y < alpha_z) )
                            // alpha_y is the nearest, so update alpha_y
           	{                        
				one_ray_length += d_x1_x2 * (alpha_y - alpha_c);
                one_ray_sum += d_x1_x2 * (alpha_y - alpha_c) * d_f[voxel_k*M*N + voxel_j*M + voxel_i];
				
                alpha_c = alpha_y; 
                N_total_sec = N_total_sec -1;                
                
                if (vertex_x1_y < vertex_x2_y)
                {
					j++;
                    voxel_j++;
                    next_alpha_index = j+1;
                }
                else if (vertex_x1_y > vertex_x2_y)
                {
					j--;
                    voxel_j--;
                    next_alpha_index = j-1;
                }   
                alpha_y = (volumn_y * next_alpha_index + boundary_voxel_y - vertex_x1_y )* inv_y_diff;                
           	}
            
			else if ( (alpha_z < alpha_x) && (alpha_z < alpha_y) )
                        // alpha_z is the nearest, so update alpha_z                
            {				
				one_ray_length += d_x1_x2 * (alpha_z - alpha_c);
                one_ray_sum += d_x1_x2 * (alpha_z - alpha_c) * d_f[voxel_k*M*N + voxel_j*M + voxel_i];
				
                alpha_c = alpha_z; 
                N_total_sec = N_total_sec -1;                
                
                if (vertex_x1_z < vertex_x2_z)
                {
					k++;
                    voxel_k++;
                    next_alpha_index = k+1;
                }
                else if (vertex_x1_z > vertex_x2_z)
                {
					k--;
                    voxel_k--;
                    next_alpha_index = k-1;
                }
                alpha_z = (volumn_z * next_alpha_index + boundary_voxel_z - vertex_x1_z )* inv_z_diff;
                
            }
		     
			else if ( (alpha_x == alpha_y) && (alpha_x < alpha_z) )
                        //x = y < z
            {        

				one_ray_length += d_x1_x2 * (alpha_x - alpha_c);  //(30)
                one_ray_sum += d_x1_x2 * (alpha_x - alpha_c) * d_f[voxel_k*M*N + voxel_j*M + voxel_i];  //(31)	
                                                                	              
				alpha_c = alpha_x;          //(33)   Update the current location
				N_total_sec = N_total_sec - 2; 
                                  
                if (vertex_x1_x < vertex_x2_x)
                {
					i = i + 1;
                    voxel_i = voxel_i +1;
                    next_alpha_index = i+1;
                }
                if (vertex_x1_x > vertex_x2_x)
                {
                    i = i - 1;      //(29)   
                    voxel_i = voxel_i-1;
                    next_alpha_index = i-1;                                        
                }
                alpha_x = (volumn_x * next_alpha_index + boundary_voxel_x - vertex_x1_x )* inv_x_diff;
                
                if (vertex_x1_y < vertex_x2_y)
                {
					j = j + 1;
                    voxel_j = voxel_j+1;
                    next_alpha_index = j+1; 
                }
                else if (vertex_x1_y > vertex_x2_y)
                {
					j = j - 1;
                    voxel_j = voxel_j-1;
                    next_alpha_index = j-1;
                }                
                alpha_y = (volumn_y * next_alpha_index + boundary_voxel_y - vertex_x1_y )* inv_y_diff;
                
            }
            
          	else if ( (alpha_x == alpha_z) && (alpha_x < alpha_y))// && (sphere_range<=1.0f) )
                        // x = z < y;
            {                      
				one_ray_length += d_x1_x2 * (alpha_x - alpha_c);  //(30)
                one_ray_sum += d_x1_x2 * (alpha_x - alpha_c) * d_f[voxel_k*M*N + voxel_j*M + voxel_i];  //(31)	
                                                                	              
				alpha_c = alpha_x;          //(33)   Update the current location
				N_total_sec = N_total_sec - 2; 

                if (vertex_x1_x < vertex_x2_x)
                {
					i = i + 1;
                    voxel_i = voxel_i +1;
                    next_alpha_index = i+1;                    
                }
                if (vertex_x1_x > vertex_x2_x)
                {
                    i = i - 1;      //(29)   
                    voxel_i = voxel_i-1;
                    next_alpha_index = i-1;                 
                }
                alpha_x = (volumn_x * next_alpha_index + boundary_voxel_x - vertex_x1_x )* inv_x_diff;
                
                if (vertex_x1_z < vertex_x2_z)
                {
					k = k + 1;
                    voxel_k = voxel_k+1;
                    next_alpha_index = k+1;
                }
                else if (vertex_x1_z > vertex_x2_z)
                {
					k = k - 1;
                    voxel_k = voxel_k-1;
                    next_alpha_index = k-1;
                }                
                alpha_z = (volumn_z * next_alpha_index + boundary_voxel_z - vertex_x1_z )* inv_z_diff;

            }
            
			else if ( (alpha_y == alpha_z) && (alpha_y < alpha_x))// && (sphere_range<=1.0f) )
                      	// y = z < x        	
            {            	
				one_ray_length += d_x1_x2 * (alpha_y - alpha_c);
                one_ray_sum += d_x1_x2 * (alpha_y - alpha_c) * d_f[voxel_k*M*N + voxel_j*M + voxel_i];
				
                alpha_c = alpha_y; 
                N_total_sec = N_total_sec -2;                
                
                if (vertex_x1_y < vertex_x2_y)
                {
					j = j + 1;
                    voxel_j = voxel_j+1;
                    next_alpha_index = j+1;
                }
                else if (vertex_x1_y > vertex_x2_y)
                {
					j = j - 1;
                    voxel_j = voxel_j-1;
                    next_alpha_index = j-1;
                }   
                alpha_y = (volumn_y * next_alpha_index + boundary_voxel_y - vertex_x1_y )* inv_y_diff;
                
                if (vertex_x1_z < vertex_x2_z)
                {
					k = k + 1;
                    voxel_k = voxel_k+1;
                    next_alpha_index = k+1;
                }
                else if (vertex_x1_z > vertex_x2_z)
                {
					k = k - 1;
                    voxel_k = voxel_k-1;
                    next_alpha_index = k-1;
                }  
                alpha_z = (volumn_z * next_alpha_index + boundary_voxel_z - vertex_x1_z )* inv_z_diff;
                
            }
			
          	else if ( (alpha_x == alpha_z) && (alpha_x == alpha_y))// && (sphere_range<=1.0f) )
                        // x=y=z            
            {
				one_ray_length += d_x1_x2 * (alpha_x - alpha_c);  //(30)
                one_ray_sum += d_x1_x2 * (alpha_x - alpha_c) * d_f[voxel_k*M*N + voxel_j*M + voxel_i];  //(31)	
                                                                	              
				alpha_c = alpha_x;          //(33)   Update the current location
				N_total_sec = N_total_sec - 3; 

                if (vertex_x1_x < vertex_x2_x)
                {
					i = i + 1;
                    voxel_i = voxel_i +1;
                    next_alpha_index = i+1;
                }
                if (vertex_x1_x > vertex_x2_x)
                {
                    i = i - 1;      //(29)   
                    voxel_i = voxel_i-1;
                    next_alpha_index = i-1;                   
                }
                alpha_x = (volumn_x * next_alpha_index + boundary_voxel_x - vertex_x1_x )* inv_x_diff;
                
                if (vertex_x1_y < vertex_x2_y)
                {
					j = j + 1;
                    voxel_j = voxel_j+1;
                    next_alpha_index = j+1;
                }
                else if (vertex_x1_y > vertex_x2_y)
                {
					j = j - 1;
                    voxel_j = voxel_j-1;
                    next_alpha_index = j-1;
                }   
                alpha_y = (volumn_y * next_alpha_index + boundary_voxel_y - vertex_x1_y )* inv_y_diff;
                
                if (vertex_x1_z < vertex_x2_z)
                {
					k = k + 1;
                    voxel_k = voxel_k+1;
                    next_alpha_index = k+1;
                }
                else if (vertex_x1_z > vertex_x2_z)
                {
					k = k - 1;
                    voxel_k = voxel_k-1;
                    next_alpha_index = k-1;
                }  
                alpha_z = (volumn_z * next_alpha_index + boundary_voxel_z - vertex_x1_z )* inv_z_diff;
           	}
          }
       	}// end while
        
        d_proj_sumLen[proj_pixel_index] = one_ray_length;                                                                             

        d_proj_weightedLen[proj_pixel_index] = one_ray_sum;            
            
                
     }//else if 
   }//else if
   
//    __syncthreads();
    
}




__global__ void backprj_ray_driven_3d_kernel_multiGPU(float *d_volumn_kernel, float *d_proj_correction, float beta_temp, float sin_theta, float cos_theta, int subVolIdx, int command)
{    
    /* 
     * Reference: "Accelerating simultaneous algebraic reconstruction technique with motion compensation using CUDA-enabled GPU" 
     * Wai-Man Pang, CUHK
     * Section: Back-projection and image update
     
     * d_proj_correction : 2D projection correction, i.e. c(i) in the Wai-Man Pang, CUHK paper
     * t_theta : projection angle
     * beta_temp : lamda in the paper
     * d_volumn: 3D object array
     * d_volumn(j) = d_volumn(j) + beta_temp * sum_i (c(i)*w(ij)) / sum_i (w(ij));  where i is ray index, j is voxel index
     */    
       
    int Idx_voxel_x = threadIdx.x + blockIdx.x * blockDim.x;
    int Idx_voxel_y = blockIdx.y;
    int Idx_voxel_z = blockIdx.z;
    
    int image_voxel_index = M * N * Idx_voxel_z + M * Idx_voxel_y + Idx_voxel_x;    

    //coordinate of center of each voxel in x-y-z system     
	float coord_voxel_x = boundary_voxel_x + volumn_x*0.5f + Idx_voxel_x * volumn_x;
    float coord_voxel_y = boundary_voxel_y + volumn_y*0.5f + Idx_voxel_y * volumn_y;
    float coord_voxel_z = boundary_voxel_z + volumn_z*(ZETA/Number_of_Devices*subVolIdx+0.5f) + Idx_voxel_z * volumn_z;   

    /**************************************/
        
	float coord_vertex_x=0.0f, coord_vertex_y=0.0f, coord_vertex_z=0.0f;  
	float coord_vertex_s=0.0f, coord_vertex_t=0.0f;
	float coord_vertexOnDetector_x=0.0f, coord_vertexOnDetector_z=0.0f; 

	float minY = MAX_infi, minZ=MAX_infi, maxY=-MAX_infi, maxZ=-MAX_infi;

	float coord_pixelOnDetector_x=0.0f, coord_pixelOnDetector_y=0.0f, coord_pixelOnDetector_z=0.0f;
	float coord_source_x=0.0f, coord_source_y=0.0f, coord_source_z=0.0f;
	float alpha_x_i_1=0.0f, alpha_x_i=0.0f;
	float alpha_y_i_1=0.0f, alpha_y_i=0.0f;
	float alpha_z_i_1=0.0f, alpha_z_i=0.0f;
   	float alpha_x_temp=0.0f, alpha_y_temp=0.0f, alpha_z_temp=0.0f; 
	float alpha_min=0.0f, alpha_max=0.0f; 
	
	int minY_index=0, maxY_index=0, minZ_index=0, maxZ_index=0; 
	float sumWeight=0.0f, sumLength=0.0f;
	float d_x1_x2=0.0f; 
    float inv_Detector_pixel = 1.0f/Detector_pixel_x;
    	
//     float weight = 1.0f;
//     float tao;
//     float tao_m1 = atan( (float(R)*Detector_pixel_x/2.0f-abs(Offset)) / DSO); 
    

    /***********************************************************/
    
	if ( (Idx_voxel_x-(float(M)*0.5f-0.5)-M_Offset)*volumn_x*(Idx_voxel_x-(float(M)*0.5f-0.5)-M_Offset)*volumn_x 
            +  (Idx_voxel_y-(float(N)*0.5f-0.5))*volumn_y*(Idx_voxel_y-(float(N)*0.5f-0.5))*volumn_y 
            >= (float(M)*0.5f-0.5)*volumn_x*(float(N)*0.5f-0.5)*volumn_y )
        
        d_volumn_kernel[image_voxel_index]  = 0.0f ;

	else
            // Note: The following codes apply to all the voxels simutaneously
	{
        
        coord_source_x = Source_x * cos_theta - Source_y * sin_theta;
        coord_source_y = Source_x * sin_theta + Source_y * cos_theta;
        if (FBCT)
        	coord_source_z = coord_voxel_z;                 // FBCT geometry, multiple sources
        else
        	coord_source_z = Source_z;                      // CBCT geometry, single source
                // coordinate of the source  in (x,y,z) system after normal gantry rotation

        /******** investigate the eight vertices of each voxel ********/
        
        for (int k=0;k<2;k++)
            for (int j=0;j<2;j++)
                for (int i=0;i<2;i++)
		{

			//coordinate for each of eight vertices of the voxel 
			coord_vertex_x = coord_voxel_x + (i)*volumn_x - 0.5f*volumn_x; 
			coord_vertex_y = coord_voxel_y + (j)*volumn_y - 0.5f*volumn_y; 
			coord_vertex_z = coord_voxel_z + (k)*volumn_z - 0.5f*volumn_z; 
            
			// <t-s> <----> <x,y>
			coord_vertex_t = coord_vertex_x * cos_theta + coord_vertex_y * sin_theta; 
			coord_vertex_s = - coord_vertex_x * sin_theta + coord_vertex_y * cos_theta;			
            // Note: Now rotate the image volume (with - t_theata degree) instead of the normal gantry rotation
            // In the new coordiantor, detector plane remains and is prependicular to the t axis 
			
            // in <t,s> system            
			coord_vertexOnDetector_x = (coord_vertex_t - DOD) / (DSO- coord_vertex_t) * (coord_vertex_s - Source_y) + coord_vertex_s ;           
            if (FBCT)
                coord_vertexOnDetector_z = coord_voxel_z ;  //FBCT geometry, no magnification along z axis
            else
                coord_vertexOnDetector_z = (coord_vertex_t - DOD) / (DSO- coord_vertex_t) * (coord_vertex_z - Source_z) + coord_vertex_z ; // CBCT geometry   			

            // the projcetion of the vertex of the voxel

			minY= fmin(minY, coord_vertexOnDetector_x);
			maxY= fmax(maxY, coord_vertexOnDetector_x); 
			minZ= fmin(minZ, coord_vertexOnDetector_z);
			maxZ= fmax(maxZ, coord_vertexOnDetector_z); 
            // form a minimim bounding rectangle (MBR) for these vertexes
            
		}

        minY_index = floor( (minY -  Detector_Ymin ) * inv_Detector_pixel +0.5f);
        maxY_index = floor( (maxY -  Detector_Ymin ) * inv_Detector_pixel +0.5f);
        minZ_index = floor( (minZ -  Detector_Zmin ) * inv_Detector_pixel +0.5f);
        maxZ_index = floor( (maxZ -  Detector_Zmin ) * inv_Detector_pixel +0.5f);
        // index of pixels of MBR boudaries on the detector 
               
        /***********************************/

        // If this voxel does not project on this detector plane, it means there is no ray passing throught this voxel at this angle.
        if ( (minY_index<0) && (maxY_index <0) )
        {	
            d_volumn_kernel[image_voxel_index]  += 0.0f ;  
        }
        else if ( (minY_index>(R-1)) && (maxY_index >(R-1)) )
        {	
            d_volumn_kernel[image_voxel_index]  += 0.0f ; 
        }
        else if ( (minZ_index<0) && (maxZ_index <0 ) )
        {	
            d_volumn_kernel[image_voxel_index]  += 0.0f ;  		
        }
        else if ( (minZ_index>(Z_prj-1)) && (maxZ_index >(Z_prj -1)) )
        {	
            d_volumn_kernel[image_voxel_index]  += 0.0f ;  		
        }
                
        else            
            // If this voxel projects on the detector plane 
        {
            
	    	if (minY_index <=0)
        	        minY_index = 0;
        	if (maxY_index >=(R-1) )
                	maxY_index = R-1;
        	if (minZ_index <=0)
                	minZ_index = 0;
        	if (maxZ_index >=(Z_prj-1) )
                	maxZ_index = Z_prj-1;
            
            // for those projection pixels whose coordinate loacates inside MBR
            // Each pixel coorresponds to a ray, and that ray must pass through the specific voxel
            for (int j=minZ_index; j<=maxZ_index; j++) 
                for (int i=minY_index; i<=maxY_index; i++)
            {
                coord_pixelOnDetector_x = DOD * cos_theta - (Detector_Ymin + i*Detector_pixel_x) * sin_theta ;
                coord_pixelOnDetector_y = DOD * sin_theta + (Detector_Ymin + i*Detector_pixel_x) * cos_theta ;
                coord_pixelOnDetector_z = Detector_Zmin + j*Detector_pixel_x;                
                // coordinate of the detector pixel inside MBR in (x,y,z) system after normal gantry rotation               
    

                
                /** Weighted Update for Half Detector **/
//                 if ( (float(i)*Detector_pixel_x) < 2.0f*abs(Offset) )
//                     weight = 1.0f;                
//                 else
//                 {
//                     tao = atan( ( float(R/2-i)*Detector_pixel_x + abs(Offset) ) / DSO);                    
//                     weight = cos(PI/4*(tao/tao_m1 - 1));
//                     weight = weight * weight;                     
//                 }
                /******/  
                
                
                // Next: investigate the line starting at x1 and ending at x2 
                // find out all the rays whose projection lies in the rectangle.
                if ( (coord_source_x == coord_pixelOnDetector_x) || (coord_source_y == coord_pixelOnDetector_y) )
                    // Otherwise you should slightly roate the angle to avoid these situations
                {
//                     assert(0);
                    sumWeight = 0.0f;
                }
                else  // if ( (coord_source_x != coord_pixelOnDetector_x) && (coord_source_y != coord_pixelOnDetector_y) )                    
                {	
                    alpha_x_i_1 =  ( (coord_voxel_x - 0.5f*volumn_x) - coord_pixelOnDetector_x )/( coord_source_x - coord_pixelOnDetector_x ); 
                    alpha_x_i   =  ( (coord_voxel_x + 0.5f*volumn_x) - coord_pixelOnDetector_x )/( coord_source_x - coord_pixelOnDetector_x ); 
                    alpha_y_i_1 =  ( (coord_voxel_y - 0.5f*volumn_y) - coord_pixelOnDetector_y )/( coord_source_y - coord_pixelOnDetector_y ); 
                    alpha_y_i   =  ( (coord_voxel_y + 0.5f*volumn_y) - coord_pixelOnDetector_y )/( coord_source_y - coord_pixelOnDetector_y );
                    alpha_z_i_1 =  ( (coord_voxel_z - 0.5f*volumn_z) - coord_pixelOnDetector_z )/( coord_source_z - coord_pixelOnDetector_z ); 
                    alpha_z_i   =  ( (coord_voxel_z + 0.5f*volumn_z) - coord_pixelOnDetector_z )/( coord_source_z - coord_pixelOnDetector_z ); 
                    // find out indices of the two most closet x planes near this specific voxel

                    alpha_x_temp = fmin((alpha_x_i_1), (alpha_x_i));
                    alpha_y_temp = fmin((alpha_y_i_1), (alpha_y_i)); 
                    if (fabs(coord_source_z - coord_pixelOnDetector_z) < volumn_z*1e-6) 
                        alpha_z_temp = -MAX_infi;                    
                    else
                        alpha_z_temp = fmin((alpha_z_i_1), (alpha_z_i)); 
					alpha_min = fmax(fmax(alpha_x_temp, alpha_y_temp), fmax(alpha_y_temp, alpha_z_temp)); 
                    // alpha_min is the enter point for one specific voxel

                    alpha_x_temp = fmax((alpha_x_i_1), (alpha_x_i));
                    alpha_y_temp = fmax((alpha_y_i_1), (alpha_y_i));
                    if (fabs(coord_source_z - coord_pixelOnDetector_z) < volumn_z*1e-6) 
                        alpha_z_temp = MAX_infi;                    
                    else                    
                        alpha_z_temp = fmax((alpha_z_i_1), (alpha_z_i));
					alpha_max = fmin(fmin(alpha_x_temp, alpha_y_temp), fmin(alpha_y_temp, alpha_z_temp));
                    // alpha_max is the exit point of the line passing through this voxel
			                   
                    if (alpha_max-alpha_min>0)        // if the value is negative, it means the ray does not pass through this voxel
                    {
                        d_x1_x2 = sqrt((coord_source_x-coord_pixelOnDetector_x)*(coord_source_x-coord_pixelOnDetector_x) + (coord_source_y-coord_pixelOnDetector_y)*(coord_source_y - coord_pixelOnDetector_y) + (coord_source_z-coord_pixelOnDetector_z)*(coord_source_z-coord_pixelOnDetector_z) );
                        float temp = d_x1_x2*(alpha_max-alpha_min);
                                
                        if  ( temp > volumn_x*1e-6)
                            // the line passes through the voxel with a sufficient length; 
                        {                        
                            sumWeight  = sumWeight +  temp*d_proj_correction[j*R  + i];
                            // Note: d_proj_correction[j*R + i] is c(i) which has been previously calculated
                            // Note: d_x1_x2 * (alpha_max - alpha_min) is w(i) for ray i of this projection 
                            sumLength = sumLength +  temp;	                        
                        }
                    }
			
                }
            }// end for loop: all the rays whose projection fits in the rectangle

            if (sumLength < volumn_x*1e-6)
            	d_volumn_kernel[image_voxel_index] += 0.0f ;  
            else
            {                
                if (command==0)
                    d_volumn_kernel[image_voxel_index] += beta_temp * sumWeight ;   // matched ajoint operator, for test use             
                else if (command==1)
                    d_volumn_kernel[image_voxel_index] += beta_temp * sumWeight/sumLength ;                        
            }
                    
        }//end else if this voxel projects on this detector plane 
    }//end else if the reconstruction region is in the circle           
    
//     __syncthreads();
    
}
 