/********
 * A fast algorithm to calculate the exact radiological path through a pxiel or voxel space
 * Filip Jacobs, etc.
 */

__global__ void forward_ray_driven_3d_kernel_correction(float *d_f , float *d_proj_correction, float *d_proj_data, float sin_theta, float cos_theta, int command)

{
	// d_f: 3D object array;    d_f[i,j,k] = d_f [k*M*N+j*M+i]; 
    // d_proj_data: 2D projection acquired at the angle of t_theta
	// d_proj_correction: 2D projection correction,  (output of this function. i.e. c(i) in the paper)    
        
    int Detector_x_idx = threadIdx.x + blockDim.x * blockIdx.x;
    int Detector_z_idx = blockIdx.y;
    
    int proj_pixel_index = Detector_z_idx * R + Detector_x_idx;
    
    
	// Source position (X2): coordinate in (x,y,z) system . 
    float vertex_x2_x,vertex_x2_y,vertex_x2_z;    
    if (CT_style==0)   //CBCT
    {
        vertex_x2_x = Source_x * cos_theta - Source_y * sin_theta;
        vertex_x2_y = Source_x * sin_theta + Source_y * cos_theta;
        vertex_x2_z = Source_z;
    }
    else if (CT_style==1) //FBCT
    {
        vertex_x2_x = Source_x * cos_theta - Source_y * sin_theta;
        vertex_x2_y = Source_x * sin_theta + Source_y * cos_theta;
        vertex_x2_z = Detector_Zmin + Detector_z_idx * Detector_pixel_x; 
    }    
    else if (CT_style==2) //parallel beam
    {
        vertex_x2_x = Source_x * cos_theta - (Detector_Ymin +  Detector_x_idx * Detector_pixel_x) * sin_theta;
        vertex_x2_y = Source_x * sin_theta + (Detector_Ymin +  Detector_x_idx * Detector_pixel_x) * cos_theta;
        vertex_x2_z = Detector_Zmin + Detector_z_idx * Detector_pixel_x;        
    }
            
    
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
        
    alpha_min = (boundary_voxel_z + volumn_z*0 - vertex_x1_z )* inv_z_diff;
    alpha_max = (boundary_voxel_z + volumn_z*ZETA - vertex_x1_z )* inv_z_diff;
        
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
          one_ray_length = 0.0f ;
  
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
			else //if (vertex_x1_x >= vertex_x2_x)
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
			else //if (vertex_x1_y >= vertex_x2_y)
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
            if (vertex_x1_z < vertex_x2_z)
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
			else //if (vertex_x1_z >= vertex_x2_z)
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

		if (fabs(vertex_x1_x-vertex_x2_x)<volumn_x*1e-6 )  
        {
            alpha_x = MAX_infi;
            i = i_min-1;
        }
        else if (vertex_x1_x < vertex_x2_x)
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
		
        if (fabs(vertex_x1_y-vertex_x2_y)<volumn_y*1e-6 )  
        {
            alpha_y = MAX_infi;
            j = j_min-1;
        }
        else if (vertex_x1_y < vertex_x2_y)
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
               
        if (vertex_x1_z < vertex_x2_z)
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
       	}// end tracing the ray                      
     }//end if the ray interacts with the volume
   
    if (one_ray_length < volumn_z*1e-6)            
        d_proj_correction[proj_pixel_index] = 0.0;
	else
    {
    	if (command == 0)
        	d_proj_correction[proj_pixel_index] = one_ray_sum; // forward operator
            
         else if (command == 1)                
         	d_proj_correction[proj_pixel_index] = (d_proj_data[proj_pixel_index] - one_ray_sum)/one_ray_length;                                                                     // projection correction (for SART)
	}  
    
//    __syncthreads();
    
}


__global__ void backprj_ray_driven_3d_kernel(float *d_f_weightedLenSum , float *d_f_LenSum , float *d_proj_correction, float sin_theta, float cos_theta)

{          
    int Detector_x_idx = threadIdx.x + blockDim.x * blockIdx.x;
    int Detector_z_idx = blockIdx.y;
    
    int proj_pixel_index = Detector_z_idx * R + Detector_x_idx;
    
	// Source position (X2): coordinate in (x,y,z) system . 
    float vertex_x2_x,vertex_x2_y,vertex_x2_z;    
    if (CT_style==0)   //CBCT
    {
        vertex_x2_x = Source_x * cos_theta - Source_y * sin_theta;
        vertex_x2_y = Source_x * sin_theta + Source_y * cos_theta;
        vertex_x2_z = Source_z;
    }
    else if (CT_style==1) //FBCT
    {
        vertex_x2_x = Source_x * cos_theta - Source_y * sin_theta;
        vertex_x2_y = Source_x * sin_theta + Source_y * cos_theta;
        vertex_x2_z = Detector_Zmin + Detector_z_idx * Detector_pixel_x; 
    }    
    else if (CT_style==2) //parallel beam
    {
        vertex_x2_x = Source_x * cos_theta - (Detector_Ymin +  Detector_x_idx * Detector_pixel_x) * sin_theta;
        vertex_x2_y = Source_x * sin_theta + (Detector_Ymin +  Detector_x_idx * Detector_pixel_x) * cos_theta;
        vertex_x2_z = Detector_Zmin + Detector_z_idx * Detector_pixel_x;        
    }
    
    // Detector element center positions (X1): Coordinate in (x,y,z) system --- 
	float vertex_x1_x = DOD * cos_theta - (Detector_Ymin +  Detector_x_idx * Detector_pixel_x) * sin_theta;
    float vertex_x1_y = DOD * sin_theta + (Detector_Ymin +  Detector_x_idx * Detector_pixel_x) * cos_theta;
    float vertex_x1_z = Detector_Zmin + Detector_z_idx * Detector_pixel_x;
        //  Notice: in this system, vertex_x1_x < 0 < vertex_x2_x    
            
    float inv_x_diff = 1.0f / (vertex_x2_x - vertex_x1_x);
    float inv_y_diff = 1.0f / (vertex_x2_y - vertex_x1_y);
    float inv_z_diff = 1.0f / (vertex_x2_z - vertex_x1_z);        
    
    float proj_val = d_proj_correction[ proj_pixel_index];        
            
    /*****************************************/
                
    float alpha_x_min= 0.0f, alpha_y_min= 0.0f, alpha_z_min= 0.0f;
    float alpha_x_max= 0.0f, alpha_y_max= 0.0f, alpha_z_max= 0.0f;
    float alpha_min= 0.0f, alpha_max= 0.0f;
        
	int i_min=0, j_min=0, k_min=0;
	int i_max=0, j_max=0, k_max=0;
    int i=0, j=0, k=0;
    int voxel_i=0, voxel_j=0, voxel_k=0;
    
	float alpha_x=0.0f, alpha_y=0.0f, alpha_z=0.0f;  

    float alpha_c= 0.0f;
    float d_x1_x2= 0.0f;

	int N_total_sec=0; 
    
    int next_alpha_index;

        
	/**** Step 1 :find out alpha_min, alpha_max ********/

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
        
    alpha_min = (boundary_voxel_z + volumn_z*0 - vertex_x1_z )* inv_z_diff;
    alpha_max = (boundary_voxel_z + volumn_z*ZETA - vertex_x1_z )* inv_z_diff;
        
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
            if (vertex_x1_z < vertex_x2_z)
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
            
        /************ initialization (i,j,k) (alpha_x_1,y,z)**************************/            
        // Note: (i,j,k) is the current x,y,z plane index (@ the initial point at the boundary)
        // Note: (alpha_x_1,alpha_y_1,alpha_z_1) is the next x,y,z plane to go.
            
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
               
        if (vertex_x1_z < vertex_x2_z)
            voxel_k = k_min-1;
        else 
            voxel_k = k_max;                                                 
        
        /***************** Updating alpha_x, alpha_y, alpha_z, ************************/
        
        // Note: (alpha_x, alpha_y, alpha_z) the intersection where the path hit the next (i.e. 1st here ) x/y/z plane inside the recon
        
        d_x1_x2 = sqrt((vertex_x2_x-vertex_x1_x)*(vertex_x2_x-vertex_x1_x) + (vertex_x2_y-vertex_x1_y)*(vertex_x2_y - vertex_x1_y) + (vertex_x2_z-vertex_x1_z)*(vertex_x2_z-vertex_x1_z) );

        alpha_c = alpha_min;    // intersection where the path hit the 1st plane at the boundary of recon region

        // Note : (i,j,k) is the (x,y,z) plane index of the current intersection (with a certain plane)
        // If i or j or k should not be an integer, then its predecessor (along the ray)
        
        while (alpha_max - alpha_c > 1e-8)
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
//                 d_f_weightedLenSum[voxel_k*M*N + voxel_j*M + voxel_i]+=d_x1_x2 * (alpha_x - alpha_c) * proj_val;
//                 d_f_LenSum[voxel_k*M*N + voxel_j*M + voxel_i]+=d_x1_x2 * (alpha_x - alpha_c);
                atomicAdd(d_f_weightedLenSum + voxel_k*M*N + voxel_j*M + voxel_i, d_x1_x2 * (alpha_x - alpha_c) * proj_val);
                atomicAdd(d_f_LenSum + voxel_k*M*N + voxel_j*M + voxel_i, d_x1_x2 * (alpha_x - alpha_c));       

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
//                 d_f_weightedLenSum[voxel_k*M*N + voxel_j*M + voxel_i]+=d_x1_x2 * (alpha_y - alpha_c)* proj_val;
//                 d_f_LenSum[voxel_k*M*N + voxel_j*M + voxel_i]+=d_x1_x2 * (alpha_y - alpha_c);
                atomicAdd(d_f_weightedLenSum + voxel_k*M*N + voxel_j*M + voxel_i, d_x1_x2 * (alpha_y - alpha_c) * proj_val);
                atomicAdd(d_f_LenSum + voxel_k*M*N + voxel_j*M + voxel_i, d_x1_x2 * (alpha_y - alpha_c));   				
				
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
//                 d_f_weightedLenSum[voxel_k*M*N + voxel_j*M + voxel_i]+=d_x1_x2 * (alpha_z - alpha_c)* proj_val;
//                 d_f_LenSum[voxel_k*M*N + voxel_j*M + voxel_i]+=d_x1_x2 * (alpha_z - alpha_c);
                atomicAdd(d_f_weightedLenSum + voxel_k*M*N + voxel_j*M + voxel_i, d_x1_x2 * (alpha_z - alpha_c) * proj_val);
                atomicAdd(d_f_LenSum + voxel_k*M*N + voxel_j*M + voxel_i, d_x1_x2 * (alpha_z - alpha_c));   				
				
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
//                 d_f_weightedLenSum[voxel_k*M*N + voxel_j*M + voxel_i]+=d_x1_x2 * (alpha_x - alpha_c)* proj_val;
//                 d_f_LenSum[voxel_k*M*N + voxel_j*M + voxel_i]+=d_x1_x2 * (alpha_x - alpha_c);
                atomicAdd(d_f_weightedLenSum + voxel_k*M*N + voxel_j*M + voxel_i, d_x1_x2 * (alpha_x - alpha_c) * proj_val);
                atomicAdd(d_f_LenSum + voxel_k*M*N + voxel_j*M + voxel_i, d_x1_x2 * (alpha_x - alpha_c)); 
                                                                	              
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
//                 d_f_weightedLenSum[voxel_k*M*N + voxel_j*M + voxel_i]+=d_x1_x2 * (alpha_x - alpha_c)* proj_val;
//                 d_f_LenSum[voxel_k*M*N + voxel_j*M + voxel_i]+=d_x1_x2 * (alpha_x - alpha_c);
                atomicAdd(d_f_weightedLenSum + voxel_k*M*N + voxel_j*M + voxel_i, d_x1_x2 * (alpha_x - alpha_c) * proj_val);
                atomicAdd(d_f_LenSum + voxel_k*M*N + voxel_j*M + voxel_i, d_x1_x2 * (alpha_x - alpha_c)); 
                                                                	              
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
//                 d_f_weightedLenSum[voxel_k*M*N + voxel_j*M + voxel_i]+=d_x1_x2 * (alpha_y - alpha_c)* proj_val;
//                 d_f_LenSum[voxel_k*M*N + voxel_j*M + voxel_i]+=d_x1_x2 * (alpha_y - alpha_c);
                atomicAdd(d_f_weightedLenSum + voxel_k*M*N + voxel_j*M + voxel_i, d_x1_x2 * (alpha_y - alpha_c) * proj_val);
                atomicAdd(d_f_LenSum + voxel_k*M*N + voxel_j*M + voxel_i, d_x1_x2 * (alpha_y - alpha_c)); 
				
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
//                 d_f_weightedLenSum[voxel_k*M*N + voxel_j*M + voxel_i]+=d_x1_x2 * (alpha_x - alpha_c) * proj_val;
//                 d_f_LenSum[voxel_k*M*N + voxel_j*M + voxel_i]+=d_x1_x2 * (alpha_x - alpha_c);
                atomicAdd(d_f_weightedLenSum + voxel_k*M*N + voxel_j*M + voxel_i, d_x1_x2 * (alpha_x - alpha_c) * proj_val);
                atomicAdd(d_f_LenSum + voxel_k*M*N + voxel_j*M + voxel_i, d_x1_x2 * (alpha_x - alpha_c)); 
                                                                	              
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
       	}// end tracing the ray                      
        
     }//end if the ray interacts with the volume
}



__global__ void update(float *d_f, float *d_f_weightedLenSum , float *d_f_LenSum, float beta)
{
    int Idx_image_x = threadIdx.x;
    int Idx_image_y = blockIdx.x;
    int Idx_image_z = blockIdx.y;
    int image_voxel_index = Idx_image_z*M*N + Idx_image_y*M + Idx_image_x;      
    
    if (d_f_LenSum[image_voxel_index] > volumn_x*1e-6)
        d_f[image_voxel_index] += beta * d_f_weightedLenSum[image_voxel_index] / d_f_LenSum[image_voxel_index];
}



__global__ void reduce_norm_2_kernel_l1(float *g_idata, float *g_odata, unsigned int n)
{
 	
	//load shared_mem
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.y* blockDim.x*gridDim.x + blockIdx.x*blockDim.x + threadIdx.x;

	sdata[tid] = (i < n) ? (g_idata[i]*g_idata[i]) : 0;

	__syncthreads();
	// do reduction in shared mem
	for(unsigned int s=blockDim.x/2; s>0; s>>=1)
        {
            if (tid < s)
                {
                        sdata[tid] += sdata[tid + s];
                }
            __syncthreads();
        }
    	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.y*gridDim.x + blockIdx.x]  = sdata[0];
}

__global__ void reduce_norm_tv_kernel_l1(float *g_idata, float *g_odata, unsigned int n)
{
 	
	//load shared_mem
	extern __shared__ float sdata[];
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.y* blockDim.x*gridDim.x + blockIdx.x*blockDim.x + threadIdx.x;

	sdata[tid] = (i < n) ? (g_idata[i]) : 0;

        __syncthreads();
        // do reduction in shared mem
	for(unsigned int s=blockDim.x/2; s>0; s>>=1)
        {
                if (tid < s)
                {
                        sdata[tid] += sdata[tid + s];
                }
        __syncthreads();
        }
    	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.y*gridDim.x + blockIdx.x]  = sdata[0];
}

__global__ void reduce_norm_2_kernel_l2(float *g_idata, float *g_odata, unsigned int n)
{

	//load shared mem 
	extern __shared__ float sdata[];
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

         sdata[tid] = (i < n) ? fabs(g_idata[i]) : 0;

        __syncthreads();
	// do reduction in shared mem
        for(unsigned int s=blockDim.x/2; s>0; s>>=1)
        {
                if (tid < s)
                {
                        sdata[tid] += sdata[tid + s];
                }
        __syncthreads();
        }
	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];

}

__global__ void reduce_norm_2_kernel_end(float *g_idata, float *g_odata, unsigned int n)
{
 
        extern __shared__ float sdata[];
        unsigned int tid = threadIdx.x;

         sdata[tid] = (tid < n) ? fabs(g_idata[tid]) : 0;

        __syncthreads();
        // do reduction in shared mem
        for(unsigned int s=blockDim.x/2; s>0; s>>=1)
        {
                if (tid < s)
                {
                        sdata[tid] += sdata[tid + s];
                }
        __syncthreads();
        }
	
	// write result for this block to global mem
        if (tid == 0) g_odata[0] = sqrt(sdata[0]);

}


__global__ void tv_gradient_matrix_3d_kernel(float *df, float *d_volumn, float epi)
{
        
	int t_id, bx_id, by_id;
        t_id  = threadIdx.x+1;
        bx_id = blockIdx.x+1;
        by_id = blockIdx.y+1;
        float stl, s_sub_1_tl, s_t_sub_1_l, st_l_sub_1;
        float s_add_1_tl, s_add_1_t_sub_1_l, s_add_1_t_l_sub_1;
        float s_t_add_1_l, s_sub_1_t_add_1_l, s_t_add_1_l_sub_1;
        float st_l_add_1, s_sub_1_t_l_add_1, s_t_sub_1_l_add_1;

        stl             = d_volumn[by_id*N*M + bx_id*M + t_id];
        s_sub_1_tl      = d_volumn[(by_id-1)*N*M + bx_id*M + t_id];
        s_t_sub_1_l     = d_volumn[by_id*N*M + (bx_id-1)*M + t_id];
        st_l_sub_1      = d_volumn[by_id*N*M + bx_id*M + t_id-1];

        s_add_1_tl      = d_volumn[(by_id+1)*N*M + bx_id*M + t_id];
        s_add_1_t_sub_1_l =  d_volumn[(by_id+1)*N*M + (bx_id-1)*M + t_id];
        s_add_1_t_l_sub_1 =  d_volumn[(by_id+1)*N*M + bx_id*M + t_id-1];

        s_t_add_1_l     = d_volumn[by_id*N*M + (bx_id+1)*M + t_id];
        s_sub_1_t_add_1_l = d_volumn[(by_id-1)*N*M + (bx_id+1)*M + t_id];
        s_t_add_1_l_sub_1 = d_volumn[by_id*N*M + (bx_id+1)*M + t_id-1];

        st_l_add_1      =d_volumn[by_id*N*M + bx_id*M + t_id + 1];
        s_sub_1_t_l_add_1 = d_volumn[(by_id-1)*N*M + bx_id*M + t_id + 1];
        s_t_sub_1_l_add_1 = d_volumn[by_id*N*M + (bx_id-1)*M + t_id + 1];

        df[by_id*N*M + bx_id*M + t_id] = ((stl - s_sub_1_tl) + (stl - s_t_sub_1_l) + (stl - st_l_sub_1) ) /sqrt(epi +  (stl - s_sub_1_tl)* (stl - s_sub_1_tl) + (stl - s_t_sub_1_l)* (stl - s_t_sub_1_l) +   (stl - st_l_sub_1)* (stl - st_l_sub_1) )
        - (s_add_1_tl - stl)/sqrt(epi +  (s_add_1_tl - stl)*(s_add_1_tl - stl)  +  (s_add_1_tl - s_add_1_t_sub_1_l)*(s_add_1_tl - s_add_1_t_sub_1_l) + (s_add_1_tl - s_add_1_t_l_sub_1)*(s_add_1_tl - s_add_1_t_l_sub_1))

        - (s_t_add_1_l - stl)/sqrt(epi +  (s_t_add_1_l - s_sub_1_t_add_1_l)*(s_t_add_1_l - s_sub_1_t_add_1_l) + (s_t_add_1_l - stl)*(s_t_add_1_l - stl) + (s_t_add_1_l - s_t_add_1_l_sub_1)* (s_t_add_1_l - s_t_add_1_l_sub_1))

        - (st_l_add_1 - stl)/sqrt(epi +  (st_l_add_1 - s_sub_1_t_l_add_1)*(st_l_add_1 - s_sub_1_t_l_add_1) + (st_l_add_1 - s_t_sub_1_l_add_1)*(st_l_add_1 - s_t_sub_1_l_add_1) + (st_l_add_1 - stl)* (st_l_add_1 - stl));


}


__global__ void tv_matrix_3d_kernel(float *df, float *d_volumn)
{
        
	int t_id, bx_id, by_id;
        t_id  = threadIdx.x+1;
        bx_id = blockIdx.x+1;
        by_id = blockIdx.y+1;
       
	float stl, s_sub_1_tl, s_t_sub_1_l, st_l_sub_1;

        stl             = d_volumn[by_id*N*M + bx_id*M + t_id];
        s_sub_1_tl      = d_volumn[(by_id-1)*N*M + bx_id*M + t_id];
        s_t_sub_1_l     = d_volumn[by_id*N*M + (bx_id-1)*M + t_id];
        st_l_sub_1      = d_volumn[by_id*N*M + bx_id*M + t_id-1];

	df[by_id*N*M + bx_id*M + t_id] = sqrt( (stl - s_sub_1_tl)*(stl - s_sub_1_tl) + (stl - s_t_sub_1_l)*(stl - s_t_sub_1_l) + (stl - st_l_sub_1)*(stl - st_l_sub_1)) ;

}

__global__ void backtracking_update_kernel(float *d_volumn_f_update,float *d_volumn_f, float *d_tv_gradient_matrix ,float alpha_temp)
{
     
        unsigned int i = blockIdx.y* blockDim.x*gridDim.x + blockIdx.x*blockDim.x + threadIdx.x;	
	d_volumn_f_update[i] = d_volumn_f[i] -  alpha_temp*d_tv_gradient_matrix[i];
}


///************ GHF new Code **************///
