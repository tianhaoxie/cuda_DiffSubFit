#include "cuda_runtime.h"
#include "device_launch_parameters.h"
__global__ void evaluate(float* V,int* F,int N_f, float* L,int* adj,int* collected_pacthes,int num_neighbor,int num_per_patch,float** eigenValues,float** eigenVectors,float** Phi);
__global__ void evaluateJacobian(int* F,int N_f, float* J,int* adj,int* collected_patches,int verts_num,int num_neighbor,int num_per_patch,float** eigenValues,float** eigenVectors,float** Phi);