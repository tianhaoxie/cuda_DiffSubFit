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
#include <chrono>
#include "src/eigenStructure.h"
#include "src/adj_list.h"
#include "src/collect_points.h"
#include "src/evaluate.h"

__global__ void test(float** eigenValues,float** eigenVectors,float** Phi){
    for (int N=3;N<13;N++){
    for(int i=0;i<N+6;i++){
        
        for(int c=0;c<12;c++)
             printf("%f ",Phi[N-3][1*(N+6)*12+i*12+c]);
        printf("\n");
    }
    printf("\n");
    }
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
void write_obj(const std::string path,thrust::host_vector<float>& V,thrust::host_vector<int>& F){
    std::ofstream fout(path);
    std::setprecision(6);
    for (int i=0;i<V.size()/3;i++){
        
        fout<<"v "<<V[i*3]<<" "<<V[i*3+1]<<" "<<V[i*3+2]<<"\n";
        
    }
    for (int i=0;i<F.size()/3;i++){
        
        fout<<"f "<<F[i*3]+1<<" "<<F[i*3+1]+1<<" "<<F[i*3+2]+1<<"\n";
        
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
    eigenStructure* es = new eigenStructure();
    float** p_eigenValues;
    float** p_eigenVectors;
    float** p_Phi;
    thrust::host_vector<float> vertices;
    thrust::host_vector<int> faces;
    int num_neighbor = 12;
    read_obj("../puppet.obj",vertices,faces);
    thrust::device_vector<float> d_vertices;
    thrust::device_vector<float> d_limit(vertices.size());
    thrust::device_vector<int> d_faces;
    thrust::device_vector<int> d_adj(vertices.size()/3*num_neighbor);
    thrust::device_vector<int> d_collected_patch(faces.size()/3*(num_neighbor+6));
    thrust::device_vector<float> d_J(vertices.size()*vertices.size()/9);
    
    thrust::host_vector<float> limit(vertices.size());
    thrust::host_vector<int> adj(vertices.size()/3*num_neighbor);
    thrust::host_vector<int> patch(faces.size()/3*(num_neighbor+6));
    thrust::fill(d_adj.begin(), d_adj.end(), -1);
    thrust::fill(d_collected_patch.begin(), d_collected_patch.end(), -1);
    thrust::fill(d_J.begin(), d_J.end(), 0);
    d_vertices = vertices;
    d_faces = faces;
    float* p_vertices = thrust::raw_pointer_cast(d_vertices.data());
    float* p_limit = thrust::raw_pointer_cast(d_limit.data());\
    float* p_J = thrust::raw_pointer_cast(d_J.data());
    //float* p_Cp = thrust::raw_pointer_cast(d_Cp.data());
    int* p_faces = thrust::raw_pointer_cast(d_faces.data());
    int* p_adj = thrust::raw_pointer_cast(d_adj.data());
    int* p_collected_patch = thrust::raw_pointer_cast(d_collected_patch.data());

    cudaHostAlloc((void**)&p_eigenValues, (num_neighbor-3)*sizeof(float*),cudaHostAllocMapped);
    cudaHostAlloc((void**)&p_eigenVectors, (num_neighbor-3)*sizeof(float*),cudaHostAllocMapped);
    cudaHostAlloc((void**)&p_Phi, (num_neighbor-3)*sizeof(float*),cudaHostAllocMapped);
    es->to_device(p_eigenValues,p_eigenVectors,p_Phi);
    /*
    size_t free_byte ;
    size_t total_byte ;
    cudaError_t  cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;
    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
    */
    int thread_num = 512;
    int total_thread_num = vertices.size()/3;
    int block_num = total_thread_num/thread_num +1;
    
    
    sorted_adjacency_list<<< block_num,thread_num >>> (p_faces,d_faces.size(),p_adj,vertices.size()/3,num_neighbor);
    cudaDeviceSynchronize();
    total_thread_num = faces.size()/3;
    block_num = total_thread_num/thread_num +1;
    
    collect_patch<<< block_num,thread_num >>> (p_faces,d_faces.size(),p_collected_patch,p_adj,num_neighbor,num_neighbor+6);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    //test<<<1,1>>> (p_eigenValues,p_eigenVectors,p_Phi);
    evaluate<<<block_num,thread_num>>> (p_vertices,p_faces,d_faces.size(),p_limit,p_adj,p_collected_patch,num_neighbor,num_neighbor+6, p_eigenValues,p_eigenVectors,p_Phi);
    cudaDeviceSynchronize();
    
    evaluateJacobian<<<block_num,thread_num>>>(p_faces,d_faces.size(),p_J,p_adj,p_collected_patch,vertices.size()/3,num_neighbor,num_neighbor+6,p_eigenValues,p_eigenVectors,p_Phi);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout<<milliseconds<<std::endl;
    adj = d_adj;
    patch = d_collected_patch;
    limit = d_limit;
    write_obj("../limit.obj",limit,faces);
    //write("../adj.txt",adj,num_neighbor);
    write("../patch.txt",patch,num_neighbor+6);
    //cudaDeviceReset();
    
    return 0;

}