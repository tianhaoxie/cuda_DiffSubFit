#include "adj_list.h"

__global__ void sorted_adjacency_list(int* f,int N_f,int* adj,const int vertex_num,const int num_neighbor){
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