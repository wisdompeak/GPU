float F_CalTV(float *A)
{
  int nz,ny,nx;
  float tv=0.0f;
  for(nz=0;nz<NO_Z;nz++)
  for(ny=0;ny<NO_Y;ny++)
  for(nx=0;nx<NO_X;nx++)
    if(nz==NO_Z-1)
      if(ny==NO_Y-1)
        if(nx==NO_X-1)
          ;//nothing to do
        else
          tv+=fabs(A[nz*NO_Y*NO_X+ny*NO_X+nx]-A[nz*NO_Y*NO_X+ny*NO_X+(nx+1)]);
      else
        if(nx==NO_X-1)
          tv+=fabs(A[nz*NO_Y*NO_X+ny*NO_X+nx]-A[nz*NO_Y*NO_X+(ny+1)*NO_X+nx]);
        else
          tv+=sqrtf(
                 (A[nz*NO_Y*NO_X+ny*NO_X+nx]-A[nz*NO_Y*NO_X+(ny+1)*NO_X+nx])
                *(A[nz*NO_Y*NO_X+ny*NO_X+nx]-A[nz*NO_Y*NO_X+(ny+1)*NO_X+nx])
               + (A[nz*NO_Y*NO_X+ny*NO_X+nx]-A[nz*NO_Y*NO_X+ny*NO_X+(nx+1)])
                *(A[nz*NO_Y*NO_X+ny*NO_X+nx]-A[nz*NO_Y*NO_X+ny*NO_X+(nx+1)])
              );
    else
      if(ny==NO_Y-1)
        if(nx==NO_X-1)
          tv+=fabs(A[nz*NO_Y*NO_X+ny*NO_X+nx]-A[(nz+1)*NO_Y*NO_X+ny*NO_X+nx]);
        else
          tv+=sqrtf(
                 (A[nz*NO_Y*NO_X+ny*NO_X+nx]-A[(nz+1)*NO_Y*NO_X+ny*NO_X+nx])
                *(A[nz*NO_Y*NO_X+ny*NO_X+nx]-A[(nz+1)*NO_Y*NO_X+ny*NO_X+nx])
               + (A[nz*NO_Y*NO_X+ny*NO_X+nx]-A[nz*NO_Y*NO_X+ny*NO_X+(nx+1)])
                *(A[nz*NO_Y*NO_X+ny*NO_X+nx]-A[nz*NO_Y*NO_X+ny*NO_X+(nx+1)])
              );
      else
        if(nx==NO_X-1)
          tv+=sqrtf(
                 (A[nz*NO_Y*NO_X+ny*NO_X+nx]-A[(nz+1)*NO_Y*NO_X+ny*NO_X+nx])
                *(A[nz*NO_Y*NO_X+ny*NO_X+nx]-A[(nz+1)*NO_Y*NO_X+ny*NO_X+nx])
               + (A[nz*NO_Y*NO_X+ny*NO_X+nx]-A[nz*NO_Y*NO_X+(ny+1)*NO_X+nx])
                *(A[nz*NO_Y*NO_X+ny*NO_X+nx]-A[nz*NO_Y*NO_X+(ny+1)*NO_X+nx])
              );
        else
          tv+=sqrtf(
                 (A[nz*NO_Y*NO_X+ny*NO_X+nx]-A[(nz+1)*NO_Y*NO_X+ny*NO_X+nx])
                *(A[nz*NO_Y*NO_X+ny*NO_X+nx]-A[(nz+1)*NO_Y*NO_X+ny*NO_X+nx])
               + (A[nz*NO_Y*NO_X+ny*NO_X+nx]-A[nz*NO_Y*NO_X+(ny+1)*NO_X+nx])
                *(A[nz*NO_Y*NO_X+ny*NO_X+nx]-A[nz*NO_Y*NO_X+(ny+1)*NO_X+nx])
               + (A[nz*NO_Y*NO_X+ny*NO_X+nx]-A[nz*NO_Y*NO_X+ny*NO_X+(nx+1)])
                *(A[nz*NO_Y*NO_X+ny*NO_X+nx]-A[nz*NO_Y*NO_X+ny*NO_X+(nx+1)])
              );
  return(tv);
}

/*************************************************/

void F_OperatorL(float *X, float *Y, float *Z, float *A)
{
// A = L(X,Y,Z)
// Note: X: NO_Z*NO_Y*(NO_X-1)  Y: NO_Z*(NO_Y-1)*NO_X   X: (NO_Z-1)*NO_Y*NO_X  A:NO_Z*NO_Y*NO_X
    
int nz,ny,nx;

for(nz=0;nz<NO_Z;nz++)
    for(ny=0;ny<NO_Y;ny++)
        for(nx=0;nx<NO_X;nx++)
    {
        // A[(nz*NO_Y+ny)*NO_X+nx] = A [nx,ny,nz] 
        // nx: row index;  ny: column index;  nz: layer index; 
        // in general A[nz,ny,nx] = X[nz,ny,nx] - X[nz,ny,nx-1] + Y[nz,ny,nx] - Y[nz,ny-1,nx] + Z[nz,ny,nx] - Z[nz-1,ny,nx] 
        
        if(nz && ny && nx)      // nz,ny,nx <> 0
            A[(nz*NO_Y+ny)*NO_X+nx]
                = X[(nz*NO_Y+ny)*(NO_X-1)+nx]-X[(nz*NO_Y+ny)*(NO_X-1)+(nx-1)]
                +Y[(nz*(NO_Y-1)+ny)*NO_X+nx]-Y[(nz*(NO_Y-1)+(ny-1))*NO_X+nx]
                +Z[(nz*NO_Y+ny)*NO_X+nx]-Z[((nz-1)*NO_Y+ny)*NO_X+nx];
                
        if(!nz &&  ny &&  nx)   // nz = 0
            A[(nz*NO_Y+ny)*NO_X+nx]
                = X[(nz*NO_Y+ny)*(NO_X-1)+nx]-X[(nz*NO_Y+ny)*(NO_X-1)+(nx-1)]
                +Y[(nz*(NO_Y-1)+ny)*NO_X+nx]-Y[(nz*(NO_Y-1)+(ny-1))*NO_X+nx]
                +Z[(nz*NO_Y+ny)*NO_X+nx];
    
        if( nz && !ny &&  nx)
            A[(nz*NO_Y+ny)*NO_X+nx]
                = X[(nz*NO_Y+ny)*(NO_X-1)+nx]-X[(nz*NO_Y+ny)*(NO_X-1)+(nx-1)]
                +Y[(nz*(NO_Y-1)+ny)*NO_X+nx]
                +Z[(nz*NO_Y+ny)*NO_X+nx]-Z[((nz-1)*NO_Y+ny)*NO_X+nx];
    
        if( nz &&  ny && !nx)
            A[(nz*NO_Y+ny)*NO_X+nx]
                = X[(nz*NO_Y+ny)*(NO_X-1)+nx]
                +Y[(nz*(NO_Y-1)+ny)*NO_X+nx]-Y[(nz*(NO_Y-1)+(ny-1))*NO_X+nx]
                +Z[(nz*NO_Y+ny)*NO_X+nx]-Z[((nz-1)*NO_Y+ny)*NO_X+nx];
    
        if(!nz && !ny &&  nx)
            A[(nz*NO_Y+ny)*NO_X+nx]
                = X[(nz*NO_Y+ny)*(NO_X-1)+nx]-X[(nz*NO_Y+ny)*(NO_X-1)+(nx-1)]
                +Y[(nz*(NO_Y-1)+ny)*NO_X+nx]
                +Z[(nz*NO_Y+ny)*NO_X+nx];
        
        if(!nz &&  ny && !nx)
            A[(nz*NO_Y+ny)*NO_X+nx]
                = X[(nz*NO_Y+ny)*(NO_X-1)+nx]
                +Y[(nz*(NO_Y-1)+ny)*NO_X+nx]-Y[(nz*(NO_Y-1)+(ny-1))*NO_X+nx]
                +Z[(nz*NO_Y+ny)*NO_X+nx];
    
        if( nz && !ny && !nx)
            A[(nz*NO_Y+ny)*NO_X+nx]
                = X[(nz*NO_Y+ny)*(NO_X-1)+nx]
                +Y[(nz*(NO_Y-1)+ny)*NO_X+nx]
                +Z[(nz*NO_Y+ny)*NO_X+nx]-Z[((nz-1)*NO_Y+ny)*NO_X+nx];
    
        if(!nz && !ny && !nx)   // i=j=k=0
            A[(nz*NO_Y+ny)*NO_X+nx]
                = X[(nz*NO_Y+ny)*(NO_X-1)+nx]
                +Y[(nz*(NO_Y-1)+ny)*NO_X+nx]
                +Z[(nz*NO_Y+ny)*NO_X+nx];
    }
}


/*************************************************/

void F_POSI (float *A)  // Pc Operator, positive constraint
{
  int nvoxel;
  for(nvoxel=0;nvoxel<NO_VOXEL;nvoxel++)
        A[nvoxel]=fmax(0.0f,A[nvoxel]);
}

/*************************************************/

void F_OperatorLT(float *A, float *X, float *Y, float *Z)
// (X,Y,Z) = LT(A)
// in general Z[nz,ny,nx] = A[nz,ny,nx]-A[nz+1,ny,nx]; 
//            Y[nz,ny,nx] = A[nz,ny,nx]-A[nz,ny+1,nx]; 
//            X[nz,ny,nx] = A[nz,ny,nx]-A[nz,ny,nx+1];
// Note: X: NO_Z*NO_Y*(NO_X-1)  Y: NO_Z*(NO_Y-1)*NO_X   X: (NO_Z-1)*NO_Y*NO_X  A:NO_Z*NO_Y*NO_X
{
    int nx,ny,nz;
    for(nz=0;nz<NO_Z;nz++)
        for(ny=0;ny<NO_Y;ny++)
            for(nx=0;nx<NO_X;nx++)
            {
                if(nx<NO_X-1)
                    X[(nz*NO_Y+ny)*(NO_X-1)+nx]=A[(nz*NO_Y+ny)*NO_X+nx]-A[(nz*NO_Y+ny)*NO_X+(nx+1)];
                if(ny<NO_Y-1)
                    Y[(nz*(NO_Y-1)+ny)*NO_X+nx]=A[(nz*NO_Y+ny)*NO_X+nx]-A[(nz*NO_Y+(ny+1))*NO_X+nx];
                if(nz<NO_Z-1)
                    Z[(nz*NO_Y+ny)*NO_X+nx]=A[(nz*NO_Y+ny)*NO_X+nx]-A[((nz+1)*NO_Y+ny)*NO_X+nx];
            }
}

/*************************************************/

void F_ProjectionP(float*X1, float *Y1, float* Z1, float *X, float *Y, float *Z)
// (X,Y,Z) = P(X1,Y1,Z1) 
// Note: X: NO_Z*NO_Y*(NO_X-1)  Y: NO_Z*(NO_Y-1)*NO_X   X: (NO_Z-1)*NO_Y*NO_X
{
  int nz, ny, nx;
  for(nz=0;nz<NO_Z;nz++)
    for(ny=0;ny<NO_Y;ny++)
        for(nx=0;nx<NO_X;nx++)
  {
    if(nx<NO_X-1)  // Update X
            // X[i,j,k] = X1[i,j,k]/max{1,sqrt[X1(i,j,k)^2+Y1(i,j,k)^2+Z1(i,j,k)^2]}
    {
      if (nz<NO_Z-1 && ny<NO_Y-1)
        X[(nz*NO_Y+ny)*(NO_X-1)+nx]=X1[(nz*NO_Y+ny)*(NO_X-1)+nx]
            /fmax(1.0f,sqrt( X1[(nz*NO_Y+ny)*(NO_X-1)+nx]*X1[(nz*NO_Y+ny)*(NO_X-1)+nx]
                            +Y1[(nz*(NO_Y-1)+ny)*NO_X+nx]*Y1[(nz*(NO_Y-1)+ny)*NO_X+nx]
                            +Z1[(nz*NO_Y+ny)*NO_X+nx]*Z1[(nz*NO_Y+ny)*NO_X+nx]));
      else
        if (ny<NO_Y-1)  //i.e. nz==NO_Z-1 
            // X[i,j,k] = X1[i,j,k]/max{1,sqrt[X1(i,j,k)^2+Y1(i,j,k)^2]}
          X[(nz*NO_Y+ny)*(NO_X-1)+nx]=X1[(nz*NO_Y+ny)*(NO_X-1)+nx]
            /fmax(1.0f,sqrt( X1[(nz*NO_Y+ny)*(NO_X-1)+nx]*X1[(nz*NO_Y+ny)*(NO_X-1)+nx]
                            +Y1[(nz*(NO_Y-1)+ny)*NO_X+nx]*Y1[(nz*(NO_Y-1)+ny)*NO_X+nx]));
        else
          if (nz<NO_Z-1)  //i.e.  ny==NO_Y-1
              // X[i,j,k] = X1[i,j,k]/max{1,sqrt[X1(i,j,k)^2+Z1(i,j,k)^2]}
            X[(nz*NO_Y+ny)*(NO_X-1)+nx]=X1[(nz*NO_Y+ny)*(NO_X-1)+nx]
              /fmax(1.0f,sqrt( X1[(nz*NO_Y+ny)*(NO_X-1)+nx]*X1[(nz*NO_Y+ny)*(NO_X-1)+nx]
                              +Z1[(nz*NO_Y+ny)*NO_X+nx]*Z1[(nz*NO_Y+ny)*NO_X+nx]));
          else  //i.e. (nz==NO_Z-1 && ny==NO_Y-1)
              // X[i,j,k] = X1[i,j,k]/max{1,abs[X1(i,j,k)]}
            X[(nz*NO_Y+ny)*(NO_X-1)+nx]=X1[(nz*NO_Y+ny)*(NO_X-1)+nx]
              /fmax(1.0f,fabs( X1[(nz*NO_Y+ny)*(NO_X-1)+nx]) );
    }
    
    if(ny<NO_Y-1)  // Update Y
        // Y[i,j,k] = Y1[i,j,k]/max{1,sqrt[X1(i,j,k)^2+Y1(i,j,k)^2+Z1(i,j,k)^2]}
    {
      if(nz<NO_Z-1 && nx<NO_X-1)
        Y[(nz*(NO_Y-1)+ny)*NO_X+nx]=Y1[(nz*(NO_Y-1)+ny)*NO_X+nx]
            /fmax(1.0f,sqrt( X1[(nz*NO_Y+ny)*(NO_X-1)+nx]*X1[(nz*NO_Y+ny)*(NO_X-1)+nx]
                            +Y1[(nz*(NO_Y-1)+ny)*NO_X+nx]*Y1[(nz*(NO_Y-1)+ny)*NO_X+nx]
                            +Z1[(nz*NO_Y+ny)*NO_X+nx]*Z1[(nz*NO_Y+ny)*NO_X+nx]));
      else
        if(nx<NO_X-1)
          Y[(nz*(NO_Y-1)+ny)*NO_X+nx]=Y1[(nz*(NO_Y-1)+ny)*NO_X+nx]
            /fmax(1.0f,sqrt( X1[(nz*NO_Y+ny)*(NO_X-1)+nx]*X1[(nz*NO_Y+ny)*(NO_X-1)+nx]
                            +Y1[(nz*(NO_Y-1)+ny)*NO_X+nx]*Y1[(nz*(NO_Y-1)+ny)*NO_X+nx]));
        else
          if(nz<NO_Z-1)
            Y[(nz*(NO_Y-1)+ny)*NO_X+nx]=Y1[(nz*(NO_Y-1)+ny)*NO_X+nx]
              /fmax(1.0f,sqrt( Y1[(nz*(NO_Y-1)+ny)*NO_X+nx]*Y1[(nz*(NO_Y-1)+ny)*NO_X+nx]
                              +Z1[(nz*NO_Y+ny)*NO_X+nx]*Z1[(nz*NO_Y+ny)*NO_X+nx]));
          else
            Y[(nz*(NO_Y-1)+ny)*NO_X+nx]=Y1[(nz*(NO_Y-1)+ny)*NO_X+nx]
              /fmax(1.0f,fabs( Y1[(nz*(NO_Y-1)+ny)*NO_X+nx]) );
    }
    
    if(nz<NO_Z-1)   // Update Z
        // Z[i,j,k] = Z1[i,j,k]/max{1,sqrt[X1(i,j,k)^2+Y1(i,j,k)^2+Z1(i,j,k)^2]}
    {
      if(ny<NO_Y-1 && nx<NO_X-1)
        Z[(nz*NO_Y+ny)*NO_X+nx]=Z1[(nz*NO_Y+ny)*NO_X+nx]
           /fmax(1.0f,sqrt( X1[(nz*NO_Y+ny)*(NO_X-1)+nx]*X1[(nz*NO_Y+ny)*(NO_X-1)+nx]
                            +Y1[(nz*(NO_Y-1)+ny)*NO_X+nx]*Y1[(nz*(NO_Y-1)+ny)*NO_X+nx]
                            +Z1[(nz*NO_Y+ny)*NO_X+nx]*Z1[(nz*NO_Y+ny)*NO_X+nx]));
      else
        if(nx<NO_X-1)
          Z[(nz*NO_Y+ny)*NO_X+nx]=Z1[(nz*NO_Y+ny)*NO_X+nx]
           /fmax(1.0f,sqrt( X1[(nz*NO_Y+ny)*(NO_X-1)+nx]*X1[(nz*NO_Y+ny)*(NO_X-1)+nx]
                            +Z1[(nz*NO_Y+ny)*NO_X+nx]*Z1[(nz*NO_Y+ny)*NO_X+nx]));
        else
          if(ny<NO_Y-1)
            Z[(nz*NO_Y+ny)*NO_X+nx]=Z1[(nz*NO_Y+ny)*NO_X+nx]
             /fmax(1.0f,sqrt(  Y1[(nz*(NO_Y-1)+ny)*NO_X+nx]*Y1[(nz*(NO_Y-1)+ny)*NO_X+nx]
                              +Z1[(nz*NO_Y+ny)*NO_X+nx]*Z1[(nz*NO_Y+ny)*NO_X+nx]));
          else
             Z[(nz*NO_Y+ny)*NO_X+nx]=Z1[(nz*NO_Y+ny)*NO_X+nx]
             /fmax(1.0f,fabs( Z1[(nz*NO_Y+ny)*NO_X+nx]));
    }
  }
}

/*************************************************/

void F_Dnoise(float *Anoise, float *Aclean, float lambda, int N)
/*  Anoise: b
 *  lambda: regularization paramter
 *  Aclean: x* = argmin(x) ||x-b||^2+TV(x) 
 *  A[(nz*NO_Y+ny)*NO_X+nx] = A [nx,ny,nz];    nx: row index;  ny: column index;  nz: layer index; 
 */
{
    int niter,nvoxel;
    float t,tp1;
    t=1.0f;
    
    // Note: X: NO_Z*NO_Y*(NO_X-1)  Y: NO_Z*(NO_Y-1)*NO_X   X: (NO_Z-1)*NO_Y*NO_X    
    
    float *X=(float*)malloc((NO_X-1)*NO_Y*NO_Z*sizeof(float));
    float *Y=(float*)malloc(NO_X*(NO_Y-1)*NO_Z*sizeof(float));
    float *Z=(float*)malloc(NO_X*NO_Y*(NO_Z-1)*sizeof(float));

    float *X0=(float*)malloc((NO_X-1)*NO_Y*NO_Z*sizeof(float));
    float *Y0=(float*)malloc(NO_X*(NO_Y-1)*NO_Z*sizeof(float));
    float *Z0=(float*)malloc(NO_X*NO_Y*(NO_Z-1)*sizeof(float));

    float *X1=(float*)malloc((NO_X-1)*NO_Y*NO_Z*sizeof(float));
    float *Y1=(float*)malloc(NO_X*(NO_Y-1)*NO_Z*sizeof(float));
    float *Z1=(float*)malloc(NO_X*NO_Y*(NO_Z-1)*sizeof(float));

    float *Xm1=(float*)malloc((NO_X-1)*NO_Y*NO_Z*sizeof(float));
    float *Ym1=(float*)malloc(NO_X*(NO_Y-1)*NO_Z*sizeof(float));
    float *Zm1=(float*)malloc(NO_X*NO_Y*(NO_Z-1)*sizeof(float));

    float *tmp=(float*)malloc(NO_VOXEL*sizeof(float));

    bzero(X,(NO_X-1)*NO_Y*NO_Z*sizeof(float));
    bzero(Y,NO_X*(NO_Y-1)*NO_Z*sizeof(float));
    bzero(Z,NO_X*NO_Y*(NO_Z-1)*sizeof(float));

    bzero(X0,(NO_X-1)*NO_Y*NO_Z*sizeof(float));
    bzero(Y0,NO_X*(NO_Y-1)*NO_Z*sizeof(float));
    bzero(Z0,NO_X*NO_Y*(NO_Z-1)*sizeof(float));

    bzero(X1,(NO_X-1)*NO_Y*NO_Z*sizeof(float));
    bzero(Y1,NO_X*(NO_Y-1)*NO_Z*sizeof(float));
    bzero(Z1,NO_X*NO_Y*(NO_Z-1)*sizeof(float));

    bzero(Xm1,(NO_X-1)*NO_Y*NO_Z*sizeof(float));
    bzero(Ym1,NO_X*(NO_Y-1)*NO_Z*sizeof(float));
    bzero(Zm1,NO_X*NO_Y*(NO_Z-1)*sizeof(float));
  
    // (X0,Y0,Z0) -> (r_k,s_k)
    // (X,Y,Z) -> (p_k, q_k)
    // (Xm1,Ym1,Zm1) -> (p_k-1, q_k-1)
    // (X1,Y1,Z1) temp for Eq(4.9)
    
	for(niter=0;niter<N;niter++)
	{
        /*********************************/  
        // This section calculates Eq(4.9)
        
        if (niter%5==0)
            printf("Denoise iteration No. %d\n",niter+1);
      
        F_OperatorL(X0,Y0,Z0,tmp);  // tmp = L[r(k),s(k)]
    
        for(nvoxel=0;nvoxel<NO_VOXEL;nvoxel++)
            tmp[nvoxel]=Anoise[nvoxel]-lambda*tmp[nvoxel];
                                    // tmp = b - lamda * tmp
    
//         F_POSI(tmp);                // tmp = Pc(tmp)
    
        F_OperatorLT(tmp,X1,Y1,Z1);     // (X1,Y1,Z1) = LT(tmp)

        for(nvoxel=0;nvoxel<(NO_X-1)*NO_Y*NO_Z;nvoxel++)
            X1[nvoxel]=X0[nvoxel]+X1[nvoxel]/(12.0f*lambda);
        for(nvoxel=0;nvoxel<NO_X*(NO_Y-1)*NO_Z;nvoxel++)
            Y1[nvoxel]=Y0[nvoxel]+Y1[nvoxel]/(12.0f*lambda);
        for(nvoxel=0;nvoxel<NO_X*NO_Y*(NO_Z-1);nvoxel++)
            Z1[nvoxel]=Z0[nvoxel]+Z1[nvoxel]/(12.0f*lambda);
                                    // (X1,Y1,Z1) = (X0,Y0,Z0) + 1/(8*lamda)*(X1,Y1,Z1)
    
        F_ProjectionP(X1,Y1,Z1,X,Y,Z);  // (X,Y,Z) = P(X1,Y1,Z1) 
    
        /*********************************/
        
        tp1=(1.0f+sqrtf(1.0f+4.0f*t*t))/2.0f;   // Eq(4.10)      
        
        /*********************************/
        
        for(nvoxel=0;nvoxel<(NO_X-1)*NO_Y*NO_Z;nvoxel++)
        {
            X0[nvoxel]=X[nvoxel]+(X[nvoxel]-Xm1[nvoxel])*(t-1)/tp1;
            Xm1[nvoxel]=X[nvoxel];
        }
        for(nvoxel=0;nvoxel<NO_X*(NO_Y-1)*NO_Z;nvoxel++)
        {
            Y0[nvoxel]=Y[nvoxel]+(Y[nvoxel]-Ym1[nvoxel])*(t-1)/tp1;
            Ym1[nvoxel]=Y[nvoxel];
        }   
        for(nvoxel=0;nvoxel<NO_X*NO_Y*(NO_Z-1);nvoxel++)
        {
            Z0[nvoxel]=Z[nvoxel]+(Z[nvoxel]-Zm1[nvoxel])*(t-1)/tp1;
            Zm1[nvoxel]=Z[nvoxel];
        }
        // Eq (4.11)
        
        t=tp1;                
        
	}//for niter
    
    /*********************************/   
  
    F_OperatorL(X,Y,Z,tmp);     // tmp = L(X,Y,Z)
  
    for(nvoxel=0;nvoxel<NO_VOXEL;nvoxel++)
        Aclean[nvoxel]=Anoise[nvoxel]-lambda*tmp[nvoxel];
                                // x* = b = lamda * tmp
  
//     F_POSI(Aclean); // x* = Pc(x*)
  
    free(X);free(Y);free(Z);
    free(X0);free(Y0);free(Z0);
    free(X1);free(Y1);free(Z1);
    free(Xm1);free(Ym1);free(Zm1);
    free(tmp);
}

