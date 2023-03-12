#include <torch/extension.h>
#include <vector>
#include <stdio.h>
// CUDA functions declarations
void loop_cuda_forward(
    float* p_vertices,
    int* p_faces,
    float* p_limit,
    int* p_adj,
    int* p_vf,
    int* p_collected_patch,
    int num_verts,
    int num_faces,
    int num_neighbor=12);

void loop_cuda_backward(
    int* p_faces,
    float* p_J,
    int* p_adj,
    int* p_vf,
    int* p_collected_patch,
    int num_verts,
    int num_faces,
    int num_neighbor=12);

void sub(int* F, int* NF,float* V,float* NV,float* S,int* adj,int* vf,int* ff,int* ffi,int* ex2,int num_faces,int num_edges,int num_verts,int num_neighbor);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
using namespace torch::indexing;
struct loop_limitation_cuda : torch::CustomClassHolder{


    loop_limitation_cuda(){}

    
    std::vector<torch::Tensor> loop_forward_backward(torch::Tensor& V,torch::Tensor& F){
        CHECK_INPUT(V);
        CHECK_INPUT(F);
        auto options_float =torch::TensorOptions()
        .dtype(torch::kFloat32)
        .layout(torch::kStrided)
        .device(torch::kCUDA, 0)
        .requires_grad(false);
        auto options_int =torch::TensorOptions()
        .dtype(torch::kInt32)
        .layout(torch::kStrided)
        .device(torch::kCUDA, 0)
        .requires_grad(false);
        std::vector<torch::Tensor> r(2);
        
        int num_neighbor=12;
        int num_verts = V.size(0);
        int num_faces = F.size(0);
        torch::Tensor limit = torch::zeros({num_verts,3},options_float);
        torch::Tensor adj = torch::full({num_verts,num_neighbor},-1,options_int);
        torch::Tensor vf = torch::zeros({num_verts},options_int);
        torch::Tensor ff = torch::zeros({num_faces,3},options_int);
        torch::Tensor ffi = torch::zeros({num_faces,3},options_int);
        torch::Tensor NF = torch::zeros({num_faces*4,3},options_int);
        torch::Tensor collected_patches = torch::full({num_verts,num_neighbor+6},-1,options_int);
        torch::Tensor J = torch::zeros({num_verts,num_verts},options_float);
        float* p_vertices = V.data_ptr<float>();
        float* p_limit = limit.data_ptr<float>();
        float* p_J = J.data_ptr<float>();
        int* p_adj = adj.data_ptr<int>();
        int* p_faces = F.data_ptr<int>();
        int* p_vf = vf.data_ptr<int>();
        int* p_ff = ff.data_ptr<int>();
        int* p_ffi = ffi.data_ptr<int>();
        int* p_NF = NF.data_ptr<int>();
        int* p_collected_patches = collected_patches.data_ptr<int>();

        torch::Tensor ex2 = torch::_cast_Int(find_edges(F));
        torch::Tensor S = torch::zeros({num_verts+ex2.size(0),num_verts},options_float);
        torch::Tensor NV = torch::zeros({num_verts+ex2.size(0),3},options_float);
        float* p_S = S.data_ptr<float>();
        float* p_NV = NV.data_ptr<float>();
        int* p_ex2 = ex2.data_ptr<int>();
        
        sub(p_faces,p_NF,p_vertices,p_NV,p_S,p_adj,p_vf,p_ff,p_ffi,p_ex2,num_faces,ex2.size(0),num_verts,num_neighbor);
        //r[0] = torch::matmul(S,V);
        r[0] = NV;
        r[1] = NF;
        //return r;
        /*
        loop_cuda_forward(
            p_vertices,
            p_faces,
            p_limit,
            p_adj,
            p_vf,
            p_collected_patches,
            num_verts,
            num_faces,
            num_neighbor);
        r[0] = limit;
        loop_cuda_backward(
            p_faces,
            p_J,
            p_adj,
            p_vf,
            p_collected_patches,
            num_verts,
            num_faces,
            num_neighbor);
        r[1] = J;
        */
        return r;
        
    }
    
    torch::Tensor find_edges(torch::Tensor& F){
        auto options_int =torch::TensorOptions()
            .dtype(torch::kInt64)
            .layout(torch::kStrided)
            .device(torch::kCUDA, 0)
            .requires_grad(false);
        {
        torch::NoGradGuard no_grad;
        //torch::Tensor edges_fx3x2 = torch::zeros({edges_fx3x2.size(0),edges_fx3x2.size(1),2},options_int);
        
        torch::Tensor ind = torch::tensor({{0,1},{1,2},{2,0}},options_int);
        auto edges_fx3x2 = F.index({Slice(),ind.index({0,Slice()})});
        for (int i=1;i<3;i++){
            edges_fx3x2 = torch::cat({edges_fx3x2,F.index({Slice(),ind.index({i,Slice()})})},1);
        }
        edges_fx3x2 = edges_fx3x2.reshape({F.size(0),3,2});
        edges_fx3x2 = edges_fx3x2.reshape({edges_fx3x2.size(0) * edges_fx3x2.size(1),2});
        auto edges_fx3x2_sorted = torch::sort(edges_fx3x2,-1);
        torch::Tensor a = std::get<0>(edges_fx3x2_sorted);
        auto edges_ex2 = torch::unique_dim(a,0);
        return std::get<0>(edges_ex2);
        }
        
}
    

};

TORCH_LIBRARY(cuda_loop, m) {
    m.class_<loop_limitation_cuda>("loop_limitation_cuda")
        .def (torch::init())
        .def ("loop_forward_backward", &loop_limitation_cuda::loop_forward_backward)
        
        
    ;
}
