#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//costruct a sorted list of adj points, padded with -1. vf is a #verts_num array record the first face idx contain that verts. 
__global__ void sorted_adjacency_list(int* f,int N_f,int* adj,int* vf,const int vertex_num,const int num_neighbor);
__global__ void face_face_adjacency(int* f,int N_f,int* ff,int*ffi);