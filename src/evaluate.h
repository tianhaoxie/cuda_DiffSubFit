#include "cuda_runtime.h"
#include "device_launch_parameters.h"
__global__ void evaluate(float* V,int* F,int verts_num, float* L,int* adj,int* vf,int* collected_patches,int num_neighbor,int num_per_patch,float** eigenValues,float** eigenVectors,float** Phi);
//__global__ void evaluateJacobian(int* F,int rows, int cols, float* J,float* S,int* adj,int* vf,int* collected_patches,int num_neighbor,int num_per_patch,float** eigenValues,float** eigenVectors,float** Phi);
__global__ void evaluateJacobian(int* F,int verts_num, float* J,int* adj,int* vf,int* collected_patches,int num_neighbor,int num_per_patch,float** eigenValues,float** eigenVectors,float** Phi);