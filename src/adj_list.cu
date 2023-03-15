#include "adj_list.h"
__device__ int search_boundary_start(int* f,int N_f,int* adj,int *tmp, const int vertex_num,const int num_neighbor){
    int vert_idx  = blockIdx.x * blockDim.x + threadIdx.x;
    int start = adj[vert_idx*num_neighbor];
    int idx = 0;
    for (int i=0;i<num_neighbor;i++){
        for (int j=0;j<num_neighbor;j++){
            if (tmp[2*j+1]==start){
                start = tmp[2*j];
                idx = j;
                break;
            }
            else if(tmp[2*j+1]==-1 || j==num_neighbor-1){
                return idx;
            }
        }
    }
}

__device__ void search_boundary(int* f,int N_f,int* adj,int *tmp, const int vertex_num,const int num_neighbor){
    int vert_idx  = blockIdx.x * blockDim.x + threadIdx.x;
    int start_idx = search_boundary_start(f,N_f,adj,tmp,vertex_num,num_neighbor);
    adj[vert_idx*num_neighbor] = tmp[start_idx*2];
    adj[vert_idx*num_neighbor+1] = tmp[start_idx*2+1];
    for (int idx=2;idx<num_neighbor;idx++){
        for (int i=0;i<num_neighbor;i++){
            if (tmp[i*2]==adj[vert_idx*num_neighbor+idx-1]){
                adj[vert_idx*num_neighbor+idx]=tmp[i*2+1];
                break;
            }
            
            else if (tmp[i*2]==-1){
                return;
            }
        }
    }
}


__global__ void sorted_adjacency_list(int* f,int N_f,int* adj,int* vf, const int vertex_num,const int num_neighbor){
    int present_thread = blockIdx.x * blockDim.x + threadIdx.x;
    if (present_thread>vertex_num-1){
        return;
    }
    int vert_idx = present_thread;
    int face_idx;
    int p_in;
    int tmp[30];
    int n=0;
    for (int i=0;i<30;i++){
        tmp[i]=-1;
    }
    for (int i =0;i<N_f;i++){
        
        if (f[i]==vert_idx){
            p_in = i%3;
            face_idx = i/3;
            if (n==0){
                vf[vert_idx]=face_idx;
                n++;
            }
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
            // no chain,search boundary
            else if (tmp[i*2]==-1){
                search_boundary(f,N_f,adj,tmp,vertex_num,num_neighbor);
                return;
            }
        }
    }
    
}

__global__ void face_face_adjacency(int* f,int N_f,int* ff,int*ffi){
    int present_thread = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (present_thread>N_f/3-1){
        return;
    }
    int ei,ej,face_idx,p_in;
    for (int i=0;i<3;i++){
        ei = f[present_thread*3+i];
        ej = f[present_thread*3+(i+1)%3];
        
        for (int j=0;j<N_f;j++){
            if (f[j]==ei){
                face_idx = j/3;
                p_in = j%3;
                
                if (f[face_idx*3+(p_in+2)%3]==ej){
                    ff[present_thread*3+i] = face_idx;
                    for (int k =0;k<3;k++){
                        if (f[face_idx*3+k]==ej){
                            ffi[present_thread*3+i] = k;
                        }
                    }
                }
            }
        }
    }
}