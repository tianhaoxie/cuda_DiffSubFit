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



__device__ void sorted_adjacency_list(int* f,int N_f,int* adj,const int vertex_num,const int num_neighbor){
    int present_thread = blockIdx.x * blockDim.x + threadIdx.x;
    if (present_thread>vertex_num-1){
        return;
    }
    int vert_idx = present_thread;
    int face_idx;
    int p_in;
    int tmp[30];
    for (int i=0;i<30;i++){
        tmp[i]=-1;
    }
    for (int i =0;i<N_f;i++){
        if (f[i]==vert_idx){
            p_in = i%3;
            face_idx = i/3;
            int v1 = f[face_idx*3+(p_in+1)%3];
            int v2 = f[face_idx*3+(p_in+2)%3];
            for (int j=0;j<num_neighbor;j++){
                if (tmp[j*2]==-1){
                    tmp[j*2]=v1;
                    tmp[j*2+1]=v2;
                    break;
                }
            }
        }
    }
    // sort neighbors
    adj[vert_idx*num_neighbor] = tmp[0];
    adj[vert_idx*num_neighbor+1] = tmp[1];
    for (int idx=2;idx<num_neighbor;idx++){
        for (int i=0;i<num_neighbor;i++){
            if (tmp[i*2]==adj[vert_idx*num_neighbor+idx-1]){
                //full chain of neighbors
                if (tmp[i*2+1]==tmp[0]){
                    return;
                }
                adj[vert_idx*num_neighbor+idx]=tmp[i*2+1];
                break;
            }
            // no chain
            else if (tmp[i*2]==-1){
                return;
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
__device__ void collect_regular(int* collected_patch, int* adj,int* face,int num_neighbor,int p_per_patch){
    int present_thread = blockIdx.x * blockDim.x + threadIdx.x;

    int adj_order[3];
    for (int i=0;i<3;i++){
        for (int j=0;j<6;j++){
            if (adj[face[i]*num_neighbor+j]==face[(i+1)%3]){
                adj_order[i] = j;
                break;
            }
        }
    }
    collected_patch[present_thread*p_per_patch] = adj[face[0]*num_neighbor+(adj_order[0]+4)%6];
    collected_patch[present_thread*p_per_patch+1] = adj[face[0]*num_neighbor+(adj_order[0]+3)%6]; 
    collected_patch[present_thread*p_per_patch+2] = adj[face[0]*num_neighbor+(adj_order[0]+5)%6];
    collected_patch[present_thread*p_per_patch+3] = face[0];
    collected_patch[present_thread*p_per_patch+4] = adj[face[0]*num_neighbor+(adj_order[0]+2)%6];
    collected_patch[present_thread*p_per_patch+5] = adj[face[1]*num_neighbor+(adj_order[0]+3)%6];
    collected_patch[present_thread*p_per_patch+6] = face[1];
    collected_patch[present_thread*p_per_patch+7] = face[2];
    collected_patch[present_thread*p_per_patch+8] = adj[face[2]*num_neighbor+(adj_order[2]+4)%6];
    collected_patch[present_thread*p_per_patch+9] = adj[face[1]*num_neighbor+(adj_order[1]+4)%6];
    collected_patch[present_thread*p_per_patch+10] = adj[face[2]*num_neighbor+(adj_order[2]+2)%6];
    collected_patch[present_thread*p_per_patch+11] = adj[face[2]*num_neighbor+(adj_order[2]+3)%6];
}

__device__ void collect_patch(int* f, int N_f, int* collected_patch,int *adj,int num_neighbor,int p_per_patch){
    int present_thread = blockIdx.x * blockDim.x + threadIdx.x;
    int face_idx = present_thread;
    int face[3];
    for (int i=0;i<3;i++){
        face[i] = f[face_idx*3+i];
    }
    if (regular(adj,face,num_neighbor)){
        collect_regular(collected_patch,adj,face,num_neighbor,p_per_patch);
    }

}

__global__ void test(int* f, int N_f,int* adj,const int vertex_num,const int num_neighbor){
    sorted_adjacency_list(f,N_f,adj,vertex_num,num_neighbor);
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
    thrust::device_vector<int> d_adj(vertices.size()/3*num_neighbor);
    thrust::device_vector<int> d_collected_patch(faces.size()/3*(num_neighbor+6));
    thrust::host_vector<int> adj(vertices.size()/3*num_neighbor);
    thrust::fill(d_adj.begin(), d_adj.end(), -1);
    thrust::fill(d_collected_patch.begin(), d_collected_patch.end(), -1);
    d_vertices = vertices;
    d_faces = faces;
    float* p_vertices = thrust::raw_pointer_cast(d_vertices.data());
    int* p_faces = thrust::raw_pointer_cast(d_faces.data());
    int* p_adj = thrust::raw_pointer_cast(d_adj.data());
    int thread_num = 256;
    int total_thread_num = vertices.size()/3;
    std::cout<<total_thread_num<<std::endl;
    int block_num = total_thread_num/thread_num +1;
    
    cudaEventRecord(start);
    test<<< block_num,thread_num >>> (p_faces,d_faces.size(),p_adj,vertices.size()/3,num_neighbor);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout<<milliseconds<<std::endl;
    adj = d_adj;
    write("../adj.txt",adj,num_neighbor);
    //cudaDeviceReset();
    return 0;

}