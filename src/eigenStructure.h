//
//  LoopData.h
//  
//
//  Created by xietianhao
//

#ifndef LOOP_EIGEN_STRUCTURE_h
#define LOOP_EIGEN_STRUCTURE_h
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>


class eigenStructure
{
public:
    
    struct EVALSTRUCT
        {
            thrust::host_vector<float> eigenValues;
            thrust::host_vector<float> inverseEigenVectorsTransposed;
            thrust::host_vector<float> Phi;
            thrust::device_vector<float> d_eigenValues;
            thrust::device_vector<float> d_eigenVectors;
            thrust::device_vector<float> d_Phi;
        };
    eigenStructure();
    ~eigenStructure();
    std::vector<EVALSTRUCT> ev;
    void load();
    void to_device(float** p_eigenValues, float** p_eigenVectors,float** Phi);

};



#endif
