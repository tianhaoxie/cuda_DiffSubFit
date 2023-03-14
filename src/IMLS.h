#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void IMLS_energy(float* query, float* pcd, float* N,float* energy, int num_query,int num_pcd,float h);
__global__ void sum(float* energy,int num_query);