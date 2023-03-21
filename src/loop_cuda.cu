#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cublas_v2.h>
#include "eigenStructure.h"
#include "adj_list.h"
#include "collect_points.h"
#include "evaluate.h"
#include "subdivision.h"
#include "IMLS.h"

void loop_cuda_fb(float* p_vertices,int* p_faces,float* p_limit,float* p_J,float* p_S,int num_verts,int num_verts_before_sub,int num_faces,int num_neighbor=12)
{   
    thrust::device_vector<int> d_adj(num_verts*num_neighbor);
    thrust::device_vector<int> d_vf(num_verts);
    thrust::device_vector<int> d_collected_patch(num_faces*(num_neighbor+6));
    thrust::fill(d_adj.begin(), d_adj.end(), -1);
    int* p_adj = thrust::raw_pointer_cast(d_adj.data());
    int* p_vf = thrust::raw_pointer_cast(d_vf.data());
    int* p_collected_patch = thrust::raw_pointer_cast(d_collected_patch.data());
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
    /*
    dim3 threads_per_block(32,32);
    dim3 blocks_per_grid(1,1);
    blocks_per_grid.x = std::ceil(static_cast<double>(num_verts_before_sub) /
                                  static_cast<double>(threads_per_block.x));
    blocks_per_grid.y = std::ceil(static_cast<double>(num_verts) /
                                  static_cast<double>(threads_per_block.y)); 
        
    evaluateJacobian<<<blocks_per_grid, threads_per_block>>>(p_faces,num_verts,num_verts_before_sub,p_J,p_S,p_adj,p_vf,p_collected_patch,num_neighbor,num_neighbor+6,p_eigenValues,p_eigenVectors,p_Phi);
    */
    evaluateJacobian<<<block_num,thread_num>>>(p_faces,num_verts,p_J,p_adj,p_vf,p_collected_patch,num_neighbor,num_neighbor+6,p_eigenValues,p_eigenVectors,p_Phi);
    cudaDeviceSynchronize();
    
}

void sub(int* F, int* NF,float* V,float* NV,float* S,int* ex2,int num_faces,int num_edges,int num_verts,int num_neighbor)
{
    thrust::device_vector<int> d_ff(num_faces*3);
    thrust::device_vector<int> d_ffi(num_faces*3);
    thrust::device_vector<int> d_adj(num_verts*num_neighbor);
    thrust::device_vector<int> d_vf(num_verts);
    thrust::fill(d_adj.begin(), d_adj.end(), -1);
    thrust::fill(d_ff.begin(), d_ff.end(), -1);
    thrust::fill(d_ffi.begin(), d_ffi.end(), -1);
    int* p_adj = thrust::raw_pointer_cast(d_adj.data());
    int* p_ff = thrust::raw_pointer_cast(d_ff.data());
    int* p_ffi = thrust::raw_pointer_cast(d_ffi.data());
    int* p_vf = thrust::raw_pointer_cast(d_vf.data());
    int thread_num = 512;
    int total_thread_num = num_verts;
    int block_num = total_thread_num/thread_num +1;
    sorted_adjacency_list<<< block_num,thread_num >>> (F,num_faces*3,p_adj,p_vf,num_verts,num_neighbor);
    cudaDeviceSynchronize();
    
    total_thread_num = num_faces;
    block_num = total_thread_num/thread_num +1;
    face_face_adjacency<<<block_num,thread_num >>>(F,num_faces*3,p_ff,p_ffi);
    cudaDeviceSynchronize();
    subdivision<<<block_num,thread_num >>>(F,NF,V,NV,S,p_adj,p_ff,p_ffi,ex2,num_faces*3,num_edges*2,num_verts,num_neighbor);
    cudaDeviceSynchronize();
}

void imls_cuda_fb(float* p_vertices,float* p_pcd,float* p_n,float* p_energy,float* p_jacobian,int num_verts,int num_pcd,float radius){
    
    int thread_num =512;
    int total_thread_num = num_verts;
    int block_num = total_thread_num/thread_num +1;
    
    IMLS_energy<<<block_num,thread_num>>>(p_vertices,p_pcd,p_n,p_energy,p_jacobian,num_verts,num_pcd,radius);
    cudaDeviceSynchronize();
    
    sum_e<<<1,1>>>(p_energy,num_verts);
}

void loop_imls_cuda_fb(float* p_vertices,int* p_faces,float* p_pcd,float* p_n, float* p_limit,float* p_J,float* p_S,float* p_energy,float* p_jacobian,int num_verts,int num_verts_before_sub,int num_faces,int num_pcd,float radius,int num_neighbor=12){
    thrust::device_vector<int> d_adj(num_verts*num_neighbor);
    thrust::device_vector<int> d_vf(num_verts);
    thrust::device_vector<int> d_collected_patch(num_faces*(num_neighbor+6));
    thrust::fill(d_adj.begin(), d_adj.end(), -1);
    int* p_adj = thrust::raw_pointer_cast(d_adj.data());
    int* p_vf = thrust::raw_pointer_cast(d_vf.data());
    int* p_collected_patch = thrust::raw_pointer_cast(d_collected_patch.data());
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
    evaluateJacobian<<<block_num,thread_num>>>(p_faces,num_verts,p_J,p_adj,p_vf,p_collected_patch,num_neighbor,num_neighbor+6,p_eigenValues,p_eigenVectors,p_Phi);
    cudaDeviceSynchronize();
    
    IMLS_energy<<<block_num,thread_num>>>(p_limit,p_pcd,p_n,p_energy,p_jacobian,num_verts,num_pcd,radius);
    cudaDeviceSynchronize();
    
    sum_e<<<1,1>>>(p_energy,num_verts);
}