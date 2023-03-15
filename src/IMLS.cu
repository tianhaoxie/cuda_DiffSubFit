#include "IMLS.h"
__device__ float sq_distance3(float* q,float* x){
    float r=0;
    for (int i=0;i<3;i++){
        r += powf((q[i]-x[i]),2.);
    }
    
    return r;
}

__device__ float phi(float r, float h){
    return powf((1.-(r/(h*h))),4.);
}

__device__ void sub3(float* a , float* b,float * c){
    
    for (int i=0;i<3;i++){
        c[i] = a[i]-b[i];
    }

}
__device__ float dot3(float* a,float* b){
    float r =0;
    for (int i =0;i<3;i++){
        r += a[i]*b[i];
    }
    return r;
}

__device__ float dphi(float r, float h,  float x,float p){
    return (-8./(h*h))*powf((1.-r/(h*h)),3.)*(x-p);
}

__device__ float dw(float r,float h, float *q, float* pts,float* n,int dim){
    float tmp[3];
    sub3(q,pts,tmp); 
    return n[dim]*phi(r,h)+dot3(n,tmp)*dphi(r,h,q[dim],pts[dim]);
}

__device__ float IMLS_deriv(float r,float h,float *q, float* pts,float* n,int dim){
    float tmp[3];
    float w;
    float dw_;
    float dphi_;
    sub3(q,pts,tmp);
    w = dot3(n,tmp)*phi(r,h);
    dw_ = dw(r,h,q,pts,n,dim);
    dphi_ = dphi(r,h,q[dim],pts[dim]);
    return 2*(w*dw_/powf(phi(r,h),2.)-w*w*dphi_/powf(phi(r,h),3.));
}

__global__ void IMLS_energy(float* query, float* pcd, float* N,float* energy,float* jacobian, int num_query,int num_pcd,float h){
    int present_thread = blockIdx.x * blockDim.x + threadIdx.x;
    float epsilon = 0.00001;
    //int present_thread = 0;
    if (present_thread>num_query-1){
        return;
    }
    float pts[3];
    float n[3];
    float q[3];
    float tmp[3];
    float r;
    float u=0;
    float v=0;
    float du[3];
    float dv[3];
    du[0] = 0;
    du[1] = 0;
    du[2] = 0;
    dv[0] = 0;
    dv[1] = 0;
    dv[2] = 0;
    q[0] = query[present_thread*3];
    q[1] = query[present_thread*3+1];
    q[2] = query[present_thread*3+2];
    for (int i=0;i<num_pcd;i++){
        pts[0] = pcd[i*3];
        pts[1] = pcd[i*3+1];
        pts[2] = pcd[i*3+2];
        n[0] = N[i*3];
        n[1] = N[i*3+1];
        n[2] = N[i*3+2];
        r = sq_distance3(q,pts);
        if (r>=h*h-epsilon){
            continue;
        }
        sub3(q,pts,tmp);
        u += dot3(n,tmp)*phi(r,h);
        v += phi(r,h);
        
        
        //printf("%f %f\n",pts[0],pts[1]);
        
        //derivative
        for (int j=0;j<3;j++){
            du[j] += dw(r,h,q,pts,n,j);
            dv[j] += dphi(r,h,q[j],pts[j]); 
        }
    }
    if (v==0){
        energy[present_thread] = 0;
        jacobian[present_thread*3] = 0;
        jacobian[present_thread*3+1] =0;
        jacobian[present_thread*3+2] =0;
    }
    else{
        energy[present_thread] = powf(u/v,2.);
        jacobian[present_thread*3] = 2*(u/v)*(du[0]*v-u*dv[0])/(v*v);
        jacobian[present_thread*3+1] =2*(u/v)*(du[1]*v-u*dv[1])/(v*v);
        jacobian[present_thread*3+2] =2*(u/v)*(du[2]*v-u*dv[2])/(v*v);
    }
    
    
}

__global__ void sum_e(float* energy,int num_query){
    energy[num_query]=0;
    for (int i=0;i<num_query;i++){
        energy[num_query]+=energy[i];
    }
    
}