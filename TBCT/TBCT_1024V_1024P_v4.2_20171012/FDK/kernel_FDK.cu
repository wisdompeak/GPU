__global__ void kernel_FDK_projection_adjust(float *d_a, float *d_b)
{    
    int idx_x =  blockIdx.x / Z_prj * blockDim.x + threadIdx.x;
    int idx_z =  blockIdx.x % Z_prj;
    int idx_source = blockIdx.y;
    int idx_view = blockIdx.z;     
    
    int pixel_idx = R*Z_prj*N_source*idx_view + R*Z_prj*idx_source + R*idx_z + idx_x;
    
    float Source_z = Source_z_min + idx_source * Source_interval;
    
    float x1 = DSO;
    float x2 = (Detector_Ymin + idx_x*Detector_pixel_x)*DSO/(DSO-DOD);
    float x3 = (Detector_Zmin + idx_z*Detector_pixel_x - Source_z)*DSO/(DSO-DOD);
    
    d_b[pixel_idx] = d_a[pixel_idx]*DSO/sqrt(x1*x1+x2*x2+x3*x3);    
}



__global__ void kernel_volume_divide(float *d_a, float *d_b, float *d_c)
{    
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y;
    int idx_z = blockIdx.z;
    
    int idx_voxel = M*N*idx_z + M*idx_y + idx_x;
           
//     if (d_c[idx_voxel]!=0.0)
//         d_a[idx_voxel] += d_b[idx_voxel]/d_c[idx_voxel];    

    d_a[idx_voxel] += d_b[idx_voxel];
}




__global__ void kernel_FDK_reconstruction(float *d_volume, float* d_volume_times, float *d_projection, float sin_theta, float cos_theta)
{
    int idx_x = blockIdx.x / N * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.x % N;
    int idx_z = blockIdx.y;
    
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
        
    float totalWeights=0.0f;
    
    for (int i=0; i<N_source; i++)
    {
    
        int idx_source = i;
    
        float Source_z = Source_z_min + idx_source * Source_interval;    

        ksi=DSO*(z-Source_z)/(DSO-s);

        ksi=(ksi*(DSO-DOD)/DSO+Source_z - Detector_Zmin) /Detector_pixel_x;
    
        p=DSO*t/(DSO-s);
    
        p=(p*(DSO-DOD)/DSO - Detector_Ymin) /Detector_pixel_x;
    
        int an,bn;
        float an0,bn0;
        float weight;
        
        if ((0<=ksi)&&(ksi+1<Z_prj)&&(0<=p)&&(p+1<R)) //If the boundaries of Projection are all zero, it works
        {
            an = floor(ksi);
            an0 = ksi-an;
            bn = floor(p);
            bn0 = p-bn;
            temp2 = (1-an0)*(1-bn0)*d_projection[R*Z_prj*idx_source+R*an+bn]+(1-an0)*bn0*d_projection[R*Z_prj*idx_source+R*an+bn+1]+an0*(1-bn0)*d_projection[R*Z_prj*idx_source+(an+1)*R+bn]+an0*bn0*d_projection[R*Z_prj*idx_source+(an+1)*R+bn+1];
            d_volume_times[idx_voxel]+=1.0f;
            
            weight=1.0f/fabs(z-Source_z)/fabs(z-Source_z);            
            totalWeights+=weight;
            d_volume[idx_voxel]+=temp1*temp2 * (us_rate*PI/180.0f) * 360.0f/(us_rate*Nviews) *weight;
        }
    }
    
    if (totalWeights>0.0f) d_volume[idx_voxel] /= totalWeights;
       
}

