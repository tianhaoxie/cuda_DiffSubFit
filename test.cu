#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <sstream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define idx(a,b) a*3+b

__device__ void collect_one_ring(int* f,int N_f,int* adj,int num_neighbor){
    int present_thread = blockIdx.x * blockDim.x + threadIdx.x;
    int vert_idx = present_thread;
    int face_idx;
    int p_in;
    for (int i =0;i<N_f;i++){
        if (f[i]==vert_idx){
            p_in = i%3;
            face_idx = i/3;
            int v1 = f[face_idx*3+(p_in+1)%3];
            int v2 = f[face_idx*3+(p_in+2)%3];
            for (int j=0;j<num_neighbor;j++){
                if (adj[vert_idx*num_neighbor+j]==v1){
                    break;
                }
                else if (adj[vert_idx*num_neighbor+j]==-1){
                    adj[vert_idx*num_neighbor+j] = v1;
                    break;
                }
            }
            for (int j=0;j<num_neighbor;j++){
                if (adj[vert_idx*num_neighbor+j]==v2){
                    break;
                }
                else if (adj[vert_idx*num_neighbor+j]==-1){
                    adj[vert_idx*num_neighbor+j] = v2;
                    break;
                }
            }
        }
    }
}

__device__ bool regular(int* adj, int* face,int num_neighbor){
    for (int i=0;i<3;i++){
        int valence = 0;
        for (int j=0;j<num_neighbor;j++){
            if(adj[face[i]*num_neighbor]==-1){
                break;
            }
            else{
                valence++;
            }
        }
        if (valence!=6){
            return false;
        }
    }
    return true;
}
__device__ void collect_regular(int* collected_patch, int* f, int* adj,int* face,int num_neighbor){
    
}
__device__ void collect_patch(int* f, int N_f, int*adj,int num_neighbor){
    int present_thread = blockIdx.x * blockDim.x + threadIdx.x;
    int face_idx = present_thread;
    int face[3];
    for (int i=0;i<3;i++){
        face[i] = f[face_idx*3+i];
    }
    if (regular(adj,face,num_neighbor)){
        
    }

}

__global__ void test(int* f, int N_f,int* adj,const int num_neighbor){
    collect_one_ring(f,N_f,adj,num_neighbor);
}

void write(const std::string path,thrust::host_vector<int>& adj,int num_neighbor){
    std::ofstream fout(path);
    for (int i=0;i<adj.size()/num_neighbor;i++){
        for (int j=0; j<num_neighbor;j++){
            if (j==num_neighbor-1){
                fout<<adj[i*num_neighbor+j]<<"\n";
            }
            else{
                fout<<adj[i*num_neighbor+j]<<" ";
            }
        }
    }
}
void read_obj(const std::string path,thrust::host_vector<float>& vertices,thrust::host_vector<int>& faces){
    std::ifstream file(path);
    std::string line;
    std::string word;
    while (std::getline(file,line)){
        std::stringstream str(line);
        std::getline(str,word,' ');
        if (word[0] == 'v'){
            for (int i=0;i<3;i++){
                std::getline(str,word,' ');
                vertices.push_back(std::stof(word));
            }
        }
        else if (word[0] == 'f'){
            for (int i=0;i<3;i++){
                std::getline(str,word,' ');
                faces.push_back(std::stoi(word)-1);
            }
        }
    } 
}
int main(){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    thrust::host_vector<float> vertices;
    thrust::host_vector<int> faces;
    int num_neighbor = 12;
    read_obj("../extracted_mesh_puppet.obj",vertices,faces);
    thrust::device_vector<float> d_vertices;
    thrust::device_vector<int> d_faces;
    thrust::device_vector<int> d_faces_adj(vertices.size()/3*num_neighbor);
    thrust::host_vector<int> faces_adj(vertices.size()/3*num_neighbor);
    thrust::fill(d_faces_adj.begin(), d_faces_adj.end(), -1);
    d_vertices = vertices;
    d_faces = faces;
    float* p_vertices = thrust::raw_pointer_cast(d_vertices.data());
    int* p_faces = thrust::raw_pointer_cast(d_faces.data());
    int* p_adj = thrust::raw_pointer_cast(d_faces_adj.data());
    int thread_num = 256;
    int total_thread_num = vertices.size();
    int block_num = total_thread_num/thread_num +1;
    
    cudaEventRecord(start);
    test<<< block_num,thread_num >>> (p_faces,d_faces.size(),p_adj,num_neighbor);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout<<milliseconds<<std::endl;
    faces_adj = d_faces_adj;
    write("../adj.txt",faces_adj,num_neighbor);
    //cudaDeviceReset();
    return 0;

}