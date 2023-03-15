#include "subdivision.h"
#include "collect_points.h"
__global__ void subdivision(int* F, int* NF,float* V,float* NV,float* S,int* adj,int* ff,int* ffi,int* ex2,int N_f,int N_e,int num_verts,int num_neighbor){
    int face_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (face_idx>N_f/3-1){
        return;
    }
    int ei,ej,e_idx,p_in;
    int face[3];
    int pts[9];
    face[0] = F[face_idx*3];
    face[1] = F[face_idx*3+1];
    face[2] = F[face_idx*3+2];
    for (int i=0;i<3;i++){
        ei = face[i];
        ej = face[(i+1)%3];
        pts[i*3]=ei;
        pts[i*3+2]=ej;
        //printf("%d %d %d ",pts[i*3],pts[i*3+1],pts[i*3+2]);
        for (int j=0;j<N_e;j++){
            
            if (ex2[j]==ei){
                e_idx = j/2;
                p_in = j%2;
                
                if (ex2[e_idx*2+(p_in+1)%2]==ej){
                    pts[i*3+1] = num_verts+e_idx;
                }
            }
        }
    }
    for (int i=0;i<3;i++){
        NF[face_idx*4*3+i*3]=pts[i*3];
        NF[face_idx*4*3+i*3+1]=pts[i*3+1];
        NF[face_idx*4*3+i*3+2]=pts[((i+2)%3)*3+1];
    }
    NF[face_idx*4*3+3*3]=pts[1];
    NF[face_idx*4*3+3*3+1]=pts[4];
    NF[face_idx*4*3+3*3+2]=pts[7];
    int N,f_adj;
    float alpha;
    float new_pt[3];
    new_pt[0]=0;
    new_pt[1]=0;
    new_pt[2]=0;

    for (int i=0;i<3;i++){
        N = valence(adj,pts[i*3],num_neighbor);
        
        f_adj = ff[face_idx*3+i];

        if (f_adj == -1){
            //boundary
            //even vertice
            S[pts[i*3]*num_verts+pts[i*3]] = 3./4.;
            S[pts[i*3]*num_verts+adj[pts[i*3]*num_neighbor]] = 1./4.;
            S[pts[i*3]*num_verts+adj[pts[i*3]*num_neighbor+N-1]] = 1./4.;
            NV[pts[i*3]*3] = 3./4.*V[pts[i*3]*3]+1./4.*(V[adj[pts[i*3]*num_neighbor]*3]+V[adj[pts[i*3]*num_neighbor+N-1]*3]);
            NV[pts[i*3]*3+1] = 3./4.*V[pts[i*3]*3+1]+1./4.*(V[adj[pts[i*3]*num_neighbor]*3+1]+V[adj[pts[i*3]*num_neighbor+N-1]*3+1]);
            NV[pts[i*3]*3+2] = 3./4.*V[pts[i*3]*3+2]+1./4.*(V[adj[pts[i*3]*num_neighbor]*3+2]+V[adj[pts[i*3]*num_neighbor+N-1]*3+2]);
            //odd vertice
            S[pts[i*3+1]*num_verts+pts[i*3]] = 1./2.;
            S[pts[i*3+1]*num_verts+pts[i*3+2]] = 1./2.;
            NV[pts[i*3+1]*3] = (1./2.)*(V[pts[i*3]*3]+V[pts[i*3+2]*3]);
            NV[pts[i*3+1]*3+1] = (1./2.)*(V[pts[i*3]*3+1]+V[pts[i*3+2]*3+1]);
            NV[pts[i*3+1]*3+2] = (1./2.)*(V[pts[i*3]*3+2]+V[pts[i*3+2]*3+2]);

        }
        else{
            //alpha = 5.0/8.0-(3/8+powf(3.0+2*cosf(2.0*3.14/(float)N),2.0))/64.0;
            if (N==3){
                alpha = 3./16.;
            }
            else{
                alpha = 3./8./N;
            }
        
            //even vertice
            S[pts[i*3]*num_verts+pts[i*3]] = 1.-N*alpha;
            new_pt[0] += (1-alpha)*V[pts[i*3]*3];
            new_pt[1] += (1-alpha)*V[pts[i*3]*3+1];
            new_pt[2] += (1-alpha)*V[pts[i*3]*3+2];
            for (int j=0;j<N;j++){
                S[pts[i*3]*num_verts+adj[pts[i*3]*num_neighbor+j]]= alpha;
                new_pt[0] += alpha/N*V[adj[pts[i*3]*num_neighbor+j]*3];
                new_pt[1] += alpha/N*V[adj[pts[i*3]*num_neighbor+j]*3+1];
                new_pt[2] += alpha/N*V[adj[pts[i*3]*num_neighbor+j]*3+2];
            }
            NV[pts[i*3]*3] = new_pt[0];
            NV[pts[i*3]*3+1] = new_pt[1];
            NV[pts[i*3]*3+2] = new_pt[2];
            new_pt[0]=0;
            new_pt[1]=0;
            new_pt[2]=0;
            // odd vertice
            S[pts[i*3+1]*num_verts+pts[i*3]] = 3./8.;
            S[pts[i*3+1]*num_verts+pts[i*3+2]] = 3./8.;
            S[pts[i*3+1]*num_verts+pts[((i+1)%3)*3+2]] = 1./8.;
            S[pts[i*3+1]*num_verts+F[f_adj*3+(ffi[face_idx*3+i]+2)%3]] = 1./8.; 
            NV[pts[i*3+1]*3] = (3./8.)*(V[pts[i*3]*3]+V[pts[i*3+2]*3])+(1./8.)*(V[pts[((i+1)%3)*3+2]*3]+V[F[f_adj*3+(ffi[face_idx*3+i]+2)%3]*3]);
            NV[pts[i*3+1]*3+1] = (3./8.)*(V[pts[i*3]*3+1]+V[pts[i*3+2]*3+1])+(1./8.)*(V[pts[((i+1)%3)*3+2]*3+1]+V[F[f_adj*3+(ffi[face_idx*3+i]+2)%3]*3+1]);
            NV[pts[i*3+1]*3+2] = (3./8.)*(V[pts[i*3]*3+2]+V[pts[i*3+2]*3+2])+(1./8.)*(V[pts[((i+1)%3)*3+2]*3+2]+V[F[f_adj*3+(ffi[face_idx*3+i]+2)%3]*3+2]);
        }
    }
}