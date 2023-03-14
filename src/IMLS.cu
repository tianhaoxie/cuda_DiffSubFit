#include "IMLS.h"
__device__ float sq_distance3(float* q,float* x){
    float r=0;
    for (int i=0;i<3;i++){
        r += powf((q[i]-x[i]),2.);
    }
    
    return r;
}

__device__ float phi(float r, float h){
    return powf((1-(r/h*h)),4.);
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

__global__ void IMLS_energy(float* query, float* pcd, float* N,float* energy, int num_query,int num_pcd,float h){
    int present_thread = blockIdx.x * blockDim.x + threadIdx.x;
    if (present_thread>num_query-1){
        return;
    }
    float pts[3];
    float n[3];
    float q[3];
    float tmp[3];
    float d;
    float u=0;
    float v=0;
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
        d = sq_distance3(q,pts);
        if (sqrtf(d)>=h){
            continue;
        }
        sub3(q,pts,tmp);
        u += dot3(n,tmp)*phi(d,h);
        v += phi(d,h);
    }
    energy[present_thread] = powf(u/v,2.);
}

__global__ void sum(float* energy,int num_query){
    energy[num_query]=0;
    for (int i=0;i<num_query;i++){
        energy[num_query]+=energy[i];
    }
    printf("%f\n",energy[num_query]);
}
