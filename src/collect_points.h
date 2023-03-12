#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void collect_patch(int* f, int N_f, int* collected_patch,int *adj,int num_neighbor,int p_per_patch);
__device__ int valence(int*adj, int vert_idx,int num_neighbor);
__device__ bool regular(int* adj, int* face,int num_neighbor);