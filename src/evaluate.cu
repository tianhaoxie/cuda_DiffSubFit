#include "evaluate.h"
#include "collect_points.h"
#include <math.h>
#include <stdio.h>
__device__ void getb(float v, float w, float* b){
    float u = 1-v-w;
    b[0] = (u*u*u*u + 2*u*u*u*v)/12;
    b[1] = (u*u*u*u + 2*u*u*u*w)/12;
    b[2] = (u*u*u*u + 2*u*u*u*w + 6*u*u*u*v + 6*u*u*v*w + 12*u*u*v*v + 6*u*v*v*w + 6*u*v*v*v + 2*v*v*v*w + v*v*v*v)/12;
    b[3] = (6*u*u*u*u + 24*u*u*u*w + 24*u*u*w*w + 8*u*w*w*w + w*w*w*w + 24*u*u*u*v + 60*u*u*v*w + 36*u*v*w*w +
    6*v*w*w*w + 24*u*u*v*v + 36*u*v*v*w + 12*v*v*w*w + 8*u*v*v*v + 6*v*v*v*w + v*v*v*v)/12;
    b[4] = (u*u*u*u + 6*u*u*u*w + 12*u*u*w*w + 6*u*w*w*w + w*w*w*w + 2*u*u*u*v + 6*u*u*v*w + 6*u*v*w*w + 2*v*w*w*w)/12;
    b[5] = (2*u*v*v*v + v*v*v*v)/12;
    b[6] = (u*u*u*u + 6*u*u*u*w + 12*u*u*w*w + 6*u*w*w*w + w*w*w*w + 8*u*u*u*v + 36*u*u*v*w +
    36*u*v*w*w + 8*v*w*w*w + 24*u*u*v*v + 60*u*v*v*w + 24*v*v*w*w + 24*u*v*v*v + 24*v*v*v*w + 6*v*v*v*v)/12;
    b[7] = (u*u*u*u + 8*u*u*u*w + 24*u*u*w*w + 24*u*w*w*w + 6*w*w*w*w + 6*u*u*u*v + 36*u*u*v*w + 60*u*v*w*w +
    24*v*w*w*w + 12*u*u*v*v + 36*u*v*v*w + 24*v*v*w*w + 6*u*v*v*v + 8*v*v*v*w + v*v*v*v)/12;
    b[8] = (2*u*w*w*w + w*w*w*w)/12;
    b[9] = (2*v*v*v*w + v*v*v*v)/12;
    b[10] = (2*u*w*w*w + w*w*w*w + 6*u*v*w*w + 6*v*w*w*w + 6*u*v*v*w + 12*v*v*w*w + 2*u*v*v*v + 6*v*v*v*w + v*v*v*v)/12;
    b[11] = (w*w*w*w + 2*v*w*w*w)/12;
}   


 
__device__ void evaluateRegular(int vertex_idx,int face_idx,float* V,int* face, float* L,int* collected_patches,int num_per_patch){
    float bary[3];
    float C[36];
    float basis[12];
    float r[3];
    float tmp =0;

    for (int i=0;i<3;i++){
        if (face[i]==vertex_idx){
            bary[i]=1;
            bary[(i+1)%3]=0;
            bary[(i+2)%3]=0;
            break;
        }
    }
    for (int i=0;i<3;i++){
        for (int j=0;j<12;j++){
            C[i*12+j] = V[collected_patches[face_idx*num_per_patch+j]*3+i];
            }    
    }
    getb(bary[1],bary[2],basis);
    for (int i =0;i<3;i++){
        for (int j=0;j<12;j++){
            tmp += C[i*12+j]*basis[j];
        }
        r[i] = tmp;
        tmp = 0;
    }
    L[vertex_idx*3] = r[0];
    L[vertex_idx*3+1] = r[1];
    L[vertex_idx*3+2] = r[2];
}



__device__ float evalBasis(float* m, float v, float w){
    float basis[12];
    getb(v,w,basis);
    float r=0;
    for (int i=0;i<12;i++){
        //printf("%f %f\n", basis[i],m[i]);
        r += basis[i]*m[i];
    }
    
    return r;
    
}

__device__ void projectPoints(float* C, float* Cp,int num_per_patch, int N,float** eigenVectors){
    float tmp;

    for (int i=0;i<N+6;i++){
        for (int j=0;j<3;j++){
            Cp[i*3+j]=0;
        }
        for (int j=0;j<N+6;j++){
            tmp = eigenVectors[N-3][j*(N+6)+i];
            Cp[i*3] += tmp * C[j*3];
            Cp[i*3+1] += tmp * C[j*3+1];
            Cp[i*3+2] += tmp * C[j*3+2];
        }
    }
}

__device__ void evalSurf(float* Pp, float* Cp,int num_per_patch,int N, float v, float w,float** eigenValues, float** Phi){
    
    int m = (int)(floorf(1.0 - log2(v+w)));
    
    int p2 = (int)(powf(2.0, (m-1)));
    //printf("%d ",p2);
    v*=p2;
    w*=p2;
    
    int k = 0;
    if(v>0.5){
        k=0;
        v = 2 * v-1;
        w = 2 * w;
    } else if(w>0.5){
        k=2;
        v=2*v;
        w=2*w-1;
    } else {
        k=1;
        v = 1 - 2*v;
        w = 1 - 2*w;
    }
    
    for (int i=0;i<3;i++){
        Pp[i] = 0;
    }
    float ma[12];
    
    for(int i=0;i<N+6;i++){
        
        for(int c=0;c<12;c++)
            ma[c] = Phi[N-3][k*(N+6)*12+i*12+c];
        
        float e = pow(eigenValues[N-3][i], m-1) * evalBasis(ma, v, w);
        Pp[0] += e * Cp[i*3];
        Pp[1] += e * Cp[i*3+1];
        Pp[2] += e * Cp[i*3+2];
    }
}

__device__ void evaluateIrregular(int vertex_idx,int face_idx, float* V,float* L,int* adj,int* collected_patches,int num_per_patch,int num_neighbor,float** eigenValues,float** eigenVectors,float** Phi){
    
    float C[18*3];
    float bary[3];
    float Cp[18*3];
    float Pp[3];
    int face_ordered[3];
    int N;
    face_ordered[0] = collected_patches[face_idx*num_per_patch];
    face_ordered[1] = collected_patches[face_idx*num_per_patch+1];
    N = valence(adj,face_ordered[0],num_neighbor);
    face_ordered[2] = collected_patches[face_idx*num_per_patch+N]; 
    for (int i=0;i<3;i++){
        if (face_ordered[i]==vertex_idx){
            bary[i]=1;
            bary[(i+1)%3]=0;
            bary[(i+2)%3]=0;
            break;
        }
    }
    for (int i=0;i<N+6;i++){
        for (int j=0;j<3;j++){
            C[i*3+j] = V[collected_patches[face_idx*num_per_patch+i]*3+j];
            }    
    }
    projectPoints(C,Cp,num_per_patch, N,eigenVectors);
    evalSurf(Pp,Cp,num_per_patch,N,bary[1],bary[2],eigenValues,Phi);
    L[vertex_idx*3] = Pp[0];
    L[vertex_idx*3+1] = Pp[1];
    L[vertex_idx*3+2] = Pp[2];
}



__global__ void evaluate(float* V,int* F,int verts_num, float* L,int* adj,int* vf,int* collected_patches,int num_neighbor,int num_per_patch,float** eigenValues,float** eigenVectors,float** Phi){
    int vert_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vert_idx>verts_num-1){
        return;
    }
    int face_idx = vf[vert_idx];
    //int face_idx = 10140;
    int face[3];
    face[0] = F[face_idx*3];
    face[1] = F[face_idx*3+1];
    face[2] = F[face_idx*3+2];
    if (regular(adj,face,num_neighbor)){
        evaluateRegular(vert_idx,face_idx,V,face,L,collected_patches,num_per_patch);
    }
    else{
        evaluateIrregular(vert_idx,face_idx,V,L,adj,collected_patches,num_per_patch,num_neighbor,eigenValues,eigenVectors,Phi);
    }
}

__device__ void evaluateJacobianRegular(int vert_idx,int face_idx,int* face, float* J,int * collected_patches, int num_per_patch,int verts_num){
    float basis[12];
    float bary[3];
    for (int i=0;i<3;i++){
        if (face[i]==vert_idx){
            bary[i]=1;
            bary[(i+1)%3]=0;
            bary[(i+2)%3]=0;
            break;
        }
    }
    getb(bary[1],bary[2],basis);
    int K;
    for (int i=0;i<12;i++){
        K = collected_patches[face_idx*num_per_patch+i];
        J[vert_idx*verts_num+K] = basis[i];
    }
}


__device__ void evaluateJacobianIrregular(int vert_idx,int face_idx,float* J,int * collected_patches, int* adj,int num_neighbor,int num_per_patch,int verts_num, float** eigenValues,float** eigenVectors,float** Phi){
    
    float Cp[18*3];
    float C[18*3];
    float Pp[3];
    int bary[3],face_ordered[3];
    int K,N;
    
    face_ordered[0] = collected_patches[face_idx*num_per_patch];
    face_ordered[1] = collected_patches[face_idx*num_per_patch+1];
    N = valence(adj,face_ordered[0],num_neighbor);
    face_ordered[2] = collected_patches[face_idx*num_per_patch+N];

    for (int i=0;i<3;i++){
        if (face_ordered[i]==vert_idx){
            bary[i]=1;
            bary[(i+1)%3]=0;
            bary[(i+2)%3]=0;
            break;
        }
    }

    for (int i=0;i<N+6;i++){
        for (int j=0;j<3;j++){
            C[i*3+j] = 0;
        }    
    }

    for (int i=0;i<N;i++){
        K = collected_patches[face_idx*num_per_patch+i];
        C[i*num_per_patch] = 1;
        C[i*num_per_patch+1] = 1;
        C[i*num_per_patch+2] = 1;
        projectPoints(C,Cp,num_per_patch, N,eigenVectors);
        evalSurf(Pp,Cp,num_per_patch,N,bary[1],bary[2],eigenValues,Phi);
        J[vert_idx*verts_num+K] = Pp[0];
        C[i*num_per_patch] = 0;
        C[i*num_per_patch+1] = 0;
        C[i*num_per_patch+2] = 0;
    }
    
    
}



__global__ void evaluateJacobian(int* F,int verts_num, float* J,int* adj,int* vf,int* collected_patches,int num_neighbor,int num_per_patch,float** eigenValues,float** eigenVectors,float** Phi){
    int verts_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (verts_idx>verts_num-1){
        return;
    }
    int face_idx = vf[verts_idx];
    int face[3];
    face[0] = F[face_idx*3];
    face[1] = F[face_idx*3+1];
    face[2] = F[face_idx*3+2];
    if (regular(adj,face,num_neighbor)){
        evaluateJacobianRegular(verts_idx,face_idx,face,J,collected_patches,num_per_patch,verts_num);
    }
    else{
        evaluateJacobianIrregular(verts_idx,face_idx,J,collected_patches,adj,num_neighbor,num_per_patch,verts_num,eigenValues,eigenVectors,Phi);
    }
}
