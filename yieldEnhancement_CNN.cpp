//Name: Reynaldo Gomez
//Last Edited: 2/3/2026
//Reference: Advanced Cuda C++ Algorithms for Semiconductor Engineering - Xi Shan

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

//----------------------------------------------------------
// Constants and Simple Structures

constexpr int Batch_size = 8;
constexpr int in_width = 8;
constexpr int in_height = 8;
constexpr int filter_size = 3;
constexpr int num_filters = 1;
constexpr int Fc_out = 1;

//----------------------------------------------------------
// Derived dimensions for the convolution output
constexpr int conv_out_height = in_height - filter_size + 1; 
constexpr int conv_out_width = in_width - filter_size + 1;

//----------------------------------------------------------
// Number of training samples
constexpr int num_samples = 32;
constexpr int EPOCH = 2;




//----------------------------------------------------------
// Forward Declarations
__global__ forwardConvKernel(const float* __restrict__ d_input,
                             const float* __restrict__ d_filter,
                             const float* __restrict__ d_biasConv,
                             const float* __restrict__ d_batchOffset
                            );

__global__ forwardFCKernel(const float* __restrict__ d_convOut



                          );