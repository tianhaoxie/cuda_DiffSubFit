//
//  eigenStructure.cpp
//  Eigen structure
//
//  Created by xie tianhao
//



#include "LoopData.h"
#include "eigenStructure.h"
#include <iostream>

eigenStructure::eigenStructure(){
    load();
}
eigenStructure::~eigenStructure(){};
void eigenStructure::load(){
    int Nmax = LoopSubdivisionData::Nmax;

    int index = 0;

    ev.resize(Nmax - 2);

    for (int i = 0; i < Nmax - 2; i++)
    {
        int N = i + 3;
        int K = N + 6;

        ev[i].eigenValues.resize(K);
        for (int j = 0; j < K; j++)
        {
            ev[i].eigenValues[j] = *(double*)&LoopSubdivisionData::data[index++];
        }
        ev[i].inverseEigenVectorsTransposed.resize(K*K);
        for (int l = 0; l < K; l++)
        {
            for (int j = 0; j < K; j++)
            {
                ev[i].inverseEigenVectorsTransposed[l*K+j] = *(double*)&LoopSubdivisionData::data[index++];
            }
        }
        ev[i].Phi.resize(K*12*3);
        for (int k = 0; k < 3; k++)
        {
            for (int l = 0; l < 12; l++)
            {
                for (int j = 0; j < K; j++)
                {
                    // data contains Phi in row major
                    ev[i].Phi[k*K*12+j*12+l] = *(double*)&LoopSubdivisionData::data[index++];
                }
            }
        }

    }
}

void eigenStructure::to_device(float** p_eigenValues, float** p_eigenVectors,float** Phi){
    int Nmax = LoopSubdivisionData::Nmax;
    for (int i = 0; i < Nmax - 2; i++){
        ev[i].d_eigenValues = ev[i].eigenValues;
        ev[i].d_eigenVectors = ev[i].inverseEigenVectorsTransposed;
        ev[i].d_Phi = ev[i].Phi;
        p_eigenValues[i] = thrust::raw_pointer_cast(ev[i].d_eigenValues.data());
        p_eigenVectors[i] = thrust::raw_pointer_cast(ev[i].d_eigenVectors.data());
        Phi[i] = thrust::raw_pointer_cast(ev[i].d_Phi.data());
    }
}

