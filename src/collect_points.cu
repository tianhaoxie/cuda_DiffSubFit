#include "collect_points.h"
__device__ int valence(int*adj, int vert_idx,int num_neighbor){
    int valence = 0;
    for (int j=0;j<num_neighbor;j++){
        if(adj[vert_idx*num_neighbor+j]==-1){
            break;
        }
        valence++;   
    }
    return valence;
}
__device__ bool regular(int* adj, int* face,int num_neighbor){
    for (int i=0;i<3;i++){
        if (valence(adj,face[i],num_neighbor)!=6){
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
    collected_patch[present_thread*p_per_patch+5] = adj[face[1]*num_neighbor+(adj_order[1]+3)%6];
    collected_patch[present_thread*p_per_patch+6] = face[1];
    collected_patch[present_thread*p_per_patch+7] = face[2];
    collected_patch[present_thread*p_per_patch+8] = adj[face[2]*num_neighbor+(adj_order[2]+4)%6];
    collected_patch[present_thread*p_per_patch+9] = adj[face[1]*num_neighbor+(adj_order[1]+4)%6];
    collected_patch[present_thread*p_per_patch+10] = adj[face[2]*num_neighbor+(adj_order[2]+2)%6];
    collected_patch[present_thread*p_per_patch+11] = adj[face[2]*num_neighbor+(adj_order[2]+3)%6];
}
__device__ void collect_irregular(int* collected_patch, int* adj,int* face,int num_neighbor,int p_per_patch){
    int present_thread = blockIdx.x * blockDim.x + threadIdx.x;
    int adj_order[3];
    int face_ordered[3];
    int N;
    for (int i=0;i<3;i++){
        N = valence(adj,face[i],num_neighbor);
        if (N!=6){
            face_ordered[0] = face[i];
            face_ordered[1] = face[(i+1)%3];
            face_ordered[2] = face[(i+2)%3];
            break;
        }
    }
    for (int j=0;j<N;j++){
            if (adj[face_ordered[0]*num_neighbor+j]==face_ordered[1]){
                adj_order[0] = j;
                break;
            }
    }
    for (int i=1;i<3;i++){
        for (int j=0;j<6;j++){
            if (adj[face_ordered[i]*num_neighbor+j]==face_ordered[(i+1)%3]){
                adj_order[i] = j;
                break;
            }
        }
    }
    collected_patch[present_thread*p_per_patch] = face_ordered[0];
    collected_patch[present_thread*p_per_patch+1] = face_ordered[1];
    for (int i=N;i>1;i--){
        collected_patch[present_thread*p_per_patch+i]=adj[face_ordered[0]*num_neighbor+(adj_order[0]+N-i+1)%N];
    }
    collected_patch[present_thread*p_per_patch+N+1] = adj[face_ordered[2]*num_neighbor+(adj_order[2]+2)%6];
    collected_patch[present_thread*p_per_patch+N+2] = adj[face_ordered[1]*num_neighbor+(adj_order[1]+4)%6];
    collected_patch[present_thread*p_per_patch+N+3] = adj[face_ordered[1]*num_neighbor+(adj_order[1]+3)%6];
    collected_patch[present_thread*p_per_patch+N+4] = adj[face_ordered[2]*num_neighbor+(adj_order[2]+3)%6];
    collected_patch[present_thread*p_per_patch+N+5] = adj[face_ordered[2]*num_neighbor+(adj_order[2]+4)%6];
}

__global__ void collect_patch(int* f, int N_f, int* collected_patch,int *adj,int num_neighbor,int p_per_patch){
    int present_thread = blockIdx.x * blockDim.x + threadIdx.x;
    if (present_thread>N_f/3-1)
        return;
    int face_idx = present_thread;
    int face[3];
    for (int i=0;i<3;i++){
        face[i] = f[face_idx*3+i];
    }
    
    if (regular(adj,face,num_neighbor)){
        collect_regular(collected_patch,adj,face,num_neighbor,p_per_patch);
    }
    else{
        collect_irregular(collected_patch,adj,face,num_neighbor,p_per_patch);
    }

}
