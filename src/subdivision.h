#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void subdivision(int* F, int* NF,float* V,float* NV,float* S,int* adj,int* ff,int* ffi,int* ex2,int N_f,int N_e,int num_verts,int num_neighbor);