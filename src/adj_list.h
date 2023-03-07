#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void sorted_adjacency_list(int* f,int N_f,int* adj,const int vertex_num,const int num_neighbor);