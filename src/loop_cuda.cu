#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "eigenStructure.h"
#include "adj_list.h"
#include "collect_points.h"
#include "evaluate.h"
#include "subdivision.h"


void loop_cuda_forward(float* p_vertices,int* p_faces,float* p_limit,int* p_adj,int* p_vf,int* p_collected_patch,int num_verts,int num_faces,int num_neighbor=12)
{
    eigenStructure* es = new eigenStructure();
    float** p_eigenValues;
    float** p_eigenVectors;
    float** p_Phi;
    cudaHostAlloc((void**)&p_eigenValues, (num_neighbor-3)*sizeof(float*),cudaHostAllocMapped);
    cudaHostAlloc((void**)&p_eigenVectors, (num_neighbor-3)*sizeof(float*),cudaHostAllocMapped);
    cudaHostAlloc((void**)&p_Phi, (num_neighbor-3)*sizeof(float*),cudaHostAllocMapped);
    es->to_device(p_eigenValues,p_eigenVectors,p_Phi);
    int thread_num = 512;
    int total_thread_num = num_verts;
    int block_num = total_thread_num/thread_num +1;
    sorted_adjacency_list<<< block_num,thread_num >>> (p_faces,num_faces*3,p_adj,p_vf,num_verts,num_neighbor);
    cudaDeviceSynchronize();
    total_thread_num = num_faces;
    block_num = total_thread_num/thread_num +1;
    collect_patch<<< block_num,thread_num >>> (p_faces,num_faces*3,p_collected_patch,p_adj,num_neighbor,num_neighbor+6);
    cudaDeviceSynchronize();
    total_thread_num = num_verts;
    block_num = total_thread_num/thread_num +1;
    evaluate<<<block_num,thread_num>>> (p_vertices,p_faces,num_verts,p_limit,p_adj,p_vf,p_collected_patch,num_neighbor,num_neighbor+6, p_eigenValues,p_eigenVectors,p_Phi);
    cudaDeviceSynchronize();
}

void loop_cuda_backward(int* p_faces,float* p_J,int* p_adj,int* p_vf,int* p_collected_patch,int num_verts,int num_faces,int num_neighbor=12)
{
    eigenStructure* es = new eigenStructure();
    float** p_eigenValues;
    float** p_eigenVectors;
    float** p_Phi;
    cudaHostAlloc((void**)&p_eigenValues, (num_neighbor-3)*sizeof(float*),cudaHostAllocMapped);
    cudaHostAlloc((void**)&p_eigenVectors, (num_neighbor-3)*sizeof(float*),cudaHostAllocMapped);
    cudaHostAlloc((void**)&p_Phi, (num_neighbor-3)*sizeof(float*),cudaHostAllocMapped);
    es->to_device(p_eigenValues,p_eigenVectors,p_Phi);
    int thread_num = 512;
    int total_thread_num = num_verts;
    int block_num = total_thread_num/thread_num +1;
    evaluateJacobian<<<block_num,thread_num>>>(p_faces,num_verts,p_J,p_adj,p_vf,p_collected_patch,num_neighbor,num_neighbor+6,p_eigenValues,p_eigenVectors,p_Phi);
    cudaDeviceSynchronize();
}

void sub(int* F, int* NF,float* V,float* NV,float* S,int* adj,int* vf,int* ff,int* ffi,int* ex2,int num_faces,int num_edges,int num_verts,int num_neighbor)
{
    
    int thread_num = 512;
    int total_thread_num = num_verts;
    int block_num = total_thread_num/thread_num +1;
    sorted_adjacency_list<<< block_num,thread_num >>> (F,num_faces*3,adj,vf,num_verts,num_neighbor);
    cudaDeviceSynchronize();
    
    total_thread_num = num_faces;
    block_num = total_thread_num/thread_num +1;
    face_face_adjacency<<<block_num,thread_num >>>(F,num_faces*3,ff,ffi);
    cudaDeviceSynchronize();
    subdivision<<<block_num,thread_num >>>(F,NF,V,NV,S,adj,ff,ffi,ex2,num_faces*3,num_edges*2,num_verts,num_neighbor);
    cudaDeviceSynchronize();
}
