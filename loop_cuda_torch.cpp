#include <torch/extension.h>
#include <vector>
#include <stdio.h>
// CUDA functions declarations
void loop_cuda_fb(
    float* p_vertices,
    int* p_faces,
    float* p_limit,
    float* p_J,
    float* p_S,
    int num_verts,
    int num_verts_before_sub,
    int num_faces,
    int num_neighbor=12);



void sub(int* F, int* NF,float* V,float* NV,float* S,int* ex2,int num_faces,int num_edges,int num_verts,int num_neighbor);

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
        std::vector<torch::Tensor> r(3);
        
        int num_neighbor=12;
        int num_verts = V.size(0);
        int num_faces = F.size(0);
        torch::Tensor ex2 = torch::_cast_Int(find_edges(F));
        torch::Tensor S = torch::zeros({num_verts+ex2.size(0),num_verts},options_float);
        torch::Tensor NV = torch::zeros({num_verts+ex2.size(0),3},options_float);
        torch::Tensor NF = torch::zeros({num_faces*4,3},options_int);
        float* p_vertices = V.data_ptr<float>();
        float* p_S = S.data_ptr<float>();
        float* p_NV = NV.data_ptr<float>();        
        int* p_faces = F.data_ptr<int>();
        int* p_ex2 = ex2.data_ptr<int>();
        int* p_NF = NF.data_ptr<int>();
        sub(p_faces,p_NF,p_vertices,p_NV,p_S,p_ex2,num_faces,ex2.size(0),num_verts,num_neighbor);
        
        //S = torch::_sparse_coo_tensor_with_dims(1,0,{S.size(0),S.size(1)},options_float);
        
        num_verts = NV.size(0);
        num_faces = NF.size(0);
        torch::Tensor limit = torch::zeros({num_verts,3},options_float);
        // J = jacobian of loop evaluation* S 
        torch::Tensor J = torch::zeros({num_verts,num_verts},options_float);
        //torch::Tensor J = torch::zeros({num_verts,num_verts},options_float);
        float* p_limit = limit.data_ptr<float>();
        float* p_J = J.data_ptr<float>();

        loop_cuda_fb(
            p_NV,
            p_NF,
            p_limit,
            p_J,
            p_S,
            num_verts,
            V.size(0),
            num_faces,
            num_neighbor);
        
        //J = torch::matmul(J,S);
        J = J.to_sparse();
        S = S.to_sparse();
        //J = J.mm(S);
        //r[0] = limit.index({Slice(V.size(0)),Slice()});
        //r[1] = J.index({Slice(V.size(0)),Slice()});
        limit = limit.index({Slice(None,V.size(0)),Slice()});
        //J = J.index({Slice(None,V.size(0)),Slice()});
        r[0] = limit;
        r[1] = J;
        r[2] = S;
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
