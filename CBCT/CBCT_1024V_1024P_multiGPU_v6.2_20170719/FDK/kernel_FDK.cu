__global__ void kernel_FDK_projection_adjust(float *d_a, float *d_b)
{    
    int idx_x =  blockDim.x * blockIdx.x + threadIdx.x;
    int idx_z = blockIdx.y;
    int idx_view = blockIdx.z;
    
    int pixel_idx = R*Z_prj*idx_view + R*idx_z + idx_x;
    
    float x1 = DSO;
    float x2 = (Detector_Ymin + idx_x*Detector_pixel_x)*DSO/(DSO-DOD);
    float x3 = (Detector_Zmin + idx_z*Detector_pixel_x)*DSO/(DSO-DOD);
    
    d_b[pixel_idx] = d_a[pixel_idx]*DSO/sqrt(x1*x1+x2*x2+x3*x3);    
}

__global__ void kernel_FDK_reconstruction(float *d_volume, float *d_projection, float sin_theta, float cos_theta)
{
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_y = blockIdx.y;
    int idx_z = blockIdx.z;
    
    int idx_voxel = M*N*idx_z + M*idx_y + idx_x;
    
    float x,y,z;
    float t,s;
    float ksi,p;
    float temp1,temp2;
    
    x = boundary_voxel_x + volumn_x*0.5f + idx_x * volumn_x;
    y = boundary_voxel_y + volumn_y*0.5f + idx_y * volumn_y;
    z = boundary_voxel_z + volumn_z*0.5f + idx_z * volumn_z;
        
    t = x*cos_theta + y*sin_theta;
    s = -x*sin_theta + y*cos_theta; 
        
    temp1 = DSO*DSO/((DSO-s)*(DSO-s));
        
    ksi=DSO*z/(DSO-s);
    ksi=(ksi*(DSO-DOD)/DSO - Detector_Zmin) / Detector_pixel_x;
    
    p=DSO*t/(DSO-s);
	p=(p*(DSO-DOD)/DSO - Detector_Ymin) / Detector_pixel_x;
    
    int an,bn;
    float an0,bn0;
            
	if ((0<=ksi)&&(ksi+1<Z_prj)&&(0<=p)&&(p+1<R)) //If the boundaries of Projection are all zero, it works
	{
        an = floor(ksi);
		an0 = ksi-an;
		bn = floor(p);
		bn0 = p-bn;
		temp2 = (1-an0)*(1-bn0)*d_projection[R*an+bn]+(1-an0)*bn0*d_projection[R*an+bn+1]+an0*(1-bn0)*d_projection[(an+1)*R+bn]+an0*bn0*d_projection[(an+1)*R+bn+1];
    }
    else
    {
        temp2=0;
    }

    d_volume[idx_voxel]+=temp1*temp2 * (us_rate*PI/180.0f) * 360.0/(us_rate*Nviews);            
       
}

