//Name: Reynaldo Gomez
//Last Edited: 2/4/2026
//Reference: Advanced Cuda C++ Algorithms for Semiconductor Engineering - Xi Shan

// #include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CUDA_CHECK(call) do {                                      \
  cudaError_t err = (call);                                        \
  if (err != cudaSuccess) {                                        \
    fprintf(stderr, "CUDA error %s at %s:%d\n",                     \
            cudaGetErrorString(err), __FILE__, __LINE__);          \
    std::exit(EXIT_FAILURE);                                       \
  }                                                                \
} while (0)

constexpr int BATCH_SIZE   = 8;
constexpr int INPUT_W      = 8;
constexpr int INPUT_H      = 8;
constexpr int FILTER_SIZE  = 3;
constexpr int NUM_FILTERS  = 1;
constexpr int FC_OUT       = 1;

constexpr int CONV_OUT_W   = INPUT_W - FILTER_SIZE + 1; // 6
constexpr int CONV_OUT_H   = INPUT_H - FILTER_SIZE + 1; // 6
constexpr int FLAT_SIZE    = NUM_FILTERS * CONV_OUT_W * CONV_OUT_H; // 36

constexpr int NUM_SAMPLES  = 32;
constexpr int EPOCHS       = 2;

// ----------------------- Forward: Convolution -----------------------
__global__ void forwardConvKernel(const float* __restrict__ d_input,
                                  const float* __restrict__ d_filter,
                                  const float* __restrict__ d_biasConv,
                                  float* __restrict__ d_convOut,
                                  int batchOffset)
{
  // One block processes one mini-batch.
  // Threads cooperate: load filter into shared memory, then each thread handles multiple output elements.
  extern __shared__ float s_filter[]; // FILTER_SIZE*FILTER_SIZE

  int t = threadIdx.x;
  if (t < FILTER_SIZE * FILTER_SIZE) s_filter[t] = d_filter[t];
  __syncthreads();

  // Total conv outputs per sample = CONV_OUT_H * CONV_OUT_W
  const int perSampleOut = CONV_OUT_H * CONV_OUT_W;
  const int totalWork = BATCH_SIZE * perSampleOut;

  int idx = t;
  while (idx < totalWork) {
    int localSample = idx / perSampleOut;        // 0..BATCH_SIZE-1
    int outIndex    = idx % perSampleOut;        // 0..perSampleOut-1
    int sampleIdx   = batchOffset + localSample; // global sample index

    if (sampleIdx < NUM_SAMPLES) {
      int outRow = outIndex / CONV_OUT_W;
      int outCol = outIndex % CONV_OUT_W;

      float sum = 0.0f;
      #pragma unroll
      for (int fr = 0; fr < FILTER_SIZE; fr++) {
        #pragma unroll
        for (int fc = 0; fc < FILTER_SIZE; fc++) {
          int inRow = outRow + fr;
          int inCol = outCol + fc;
          float w = s_filter[fr * FILTER_SIZE + fc];
          float x = d_input[sampleIdx * (INPUT_H * INPUT_W) + inRow * INPUT_W + inCol];
          sum += w * x;
        }
      }
      sum += d_biasConv[0]; // one filter
      d_convOut[sampleIdx * perSampleOut + outIndex] = sum;
    }

    idx += blockDim.x;
  }
}

// ----------------------- Forward: Fully Connected -----------------------
__global__ void forwardFCKernel(const float* __restrict__ d_convOut,
                                const float* __restrict__ d_fcWeight,
                                const float* __restrict__ d_biasFC,
                                float* __restrict__ d_fcOut,
                                int batchOffset)
{
  int t = threadIdx.x;
  // One thread computes one sample output (enough for BATCH_SIZE <= blockDim.x)
  if (t >= BATCH_SIZE) return;

  int sampleIdx = batchOffset + t;
  if (sampleIdx >= NUM_SAMPLES) return;

  float sum = 0.0f;
  const int base = sampleIdx * FLAT_SIZE;

  for (int i = 0; i < FLAT_SIZE; i++) {
    sum += d_convOut[base + i] * d_fcWeight[i];
  }
  sum += d_biasFC[0];

  // Sigmoid
  float y = 1.0f / (1.0f + expf(-sum));
  d_fcOut[sampleIdx] = y;
}

// ----------------------- Loss + dL/dy -----------------------
__global__ void computeLossAndDoutKernel(const float* __restrict__ d_fcOut,
                                        const float* __restrict__ d_labels,
                                        float* __restrict__ d_lossPerSample,
                                        float* __restrict__ d_fcOutGrad, // dL/dy
                                        int batchOffset)
{
  int t = threadIdx.x;
  if (t >= BATCH_SIZE) return;

  int sampleIdx = batchOffset + t;
  if (sampleIdx >= NUM_SAMPLES) return;

  float y = d_fcOut[sampleIdx];
  float target = d_labels[sampleIdx];
  float diff = (y - target);

  // MSE: 0.5*(y-target)^2
  d_lossPerSample[t] = 0.5f * diff * diff;

  // dL/dy = (y-target)
  d_fcOutGrad[sampleIdx] = diff;
}

// ----------------------- Backward: FC (weights + bias grads, and grad into convOut) -----------------------
__global__ void backwardFCKernel(const float* __restrict__ d_convOut,
                                 const float* __restrict__ d_fcOut,     // y
                                 const float* __restrict__ d_fcOutGrad, // dL/dy
                                 float* __restrict__ d_fcWeightGrad,
                                 float* __restrict__ d_biasFCGrad,
                                 float* __restrict__ d_convOutGrad,     // dL/d(convOut)
                                 int batchOffset)
{
  int t = threadIdx.x;
  // Each thread handles one sample and atomically accumulates grads (toy approach)
  if (t >= BATCH_SIZE) return;

  int sampleIdx = batchOffset + t;
  if (sampleIdx >= NUM_SAMPLES) return;

  float y = d_fcOut[sampleIdx];
  float dL_dy = d_fcOutGrad[sampleIdx];

  // sigmoid: dy/dz = y*(1-y) where z is pre-sigmoid logit
  float dL_dz = dL_dy * (y * (1.0f - y));

  // Bias grad
  atomicAdd(&d_biasFCGrad[0], dL_dz);

  // Weight grad and convOut grad
  int base = sampleIdx * FLAT_SIZE;
  for (int i = 0; i < FLAT_SIZE; i++) {
    float x = d_convOut[base + i];
    atomicAdd(&d_fcWeightGrad[i], dL_dz * x);

    // dL/dx_i = dL/dz * w_i
    d_convOutGrad[base + i] = dL_dz * d_fcWeightGrad[0]; // WRONG: placeholder
  }

  // Fix: need actual weights, not weightGrad. We'll write correct computation:
  // (We cannot read d_fcWeight here unless it's passed. Keep it simple: pass weights.)
}

// Correct FC backward kernel (reads weights)
__global__ void backwardFCKernel2(const float* __restrict__ d_convOut,
                                  const float* __restrict__ d_fcWeight,
                                  const float* __restrict__ d_fcOut,
                                  const float* __restrict__ d_fcOutGrad,
                                  float* __restrict__ d_fcWeightGrad,
                                  float* __restrict__ d_biasFCGrad,
                                  float* __restrict__ d_convOutGrad,
                                  int batchOffset)
{
  int t = threadIdx.x;
  if (t >= BATCH_SIZE) return;

  int sampleIdx = batchOffset + t;
  if (sampleIdx >= NUM_SAMPLES) return;

  float y = d_fcOut[sampleIdx];
  float dL_dy = d_fcOutGrad[sampleIdx];
  float dL_dz = dL_dy * (y * (1.0f - y));

  atomicAdd(&d_biasFCGrad[0], dL_dz);

  int base = sampleIdx * FLAT_SIZE;
  for (int i = 0; i < FLAT_SIZE; i++) {
    float x = d_convOut[base + i];
    atomicAdd(&d_fcWeightGrad[i], dL_dz * x);
    d_convOutGrad[base + i] = dL_dz * d_fcWeight[i];
  }
}

// ----------------------- Backward: Conv (filter + bias grads) -----------------------
__global__ void backwardConvKernel(const float* __restrict__ d_input,
                                   const float* __restrict__ d_convOutGrad, // dL/d(convOut) per sample
                                   float* __restrict__ d_filterGrad,
                                   float* __restrict__ d_biasConvGrad,
                                   int batchOffset)
{
  // Compute grads for a single 3x3 filter and bias.
  // We accumulate with atomics across samples/threads (toy approach).

  int t = threadIdx.x;
  const int perSampleOut = CONV_OUT_H * CONV_OUT_W;

  // Work over all conv outputs in the batch; each output contributes to filter grad
  int idx = t;
  while (idx < BATCH_SIZE * perSampleOut) {
    int localSample = idx / perSampleOut;
    int outIndex    = idx % perSampleOut;
    int sampleIdx   = batchOffset + localSample;

    if (sampleIdx < NUM_SAMPLES) {
      int outRow = outIndex / CONV_OUT_W;
      int outCol = outIndex % CONV_OUT_W;

      float go = d_convOutGrad[sampleIdx * perSampleOut + outIndex]; // grad at this conv output
      atomicAdd(&d_biasConvGrad[0], go);

      for (int fr = 0; fr < FILTER_SIZE; fr++) {
        for (int fc = 0; fc < FILTER_SIZE; fc++) {
          int inRow = outRow + fr;
          int inCol = outCol + fc;
          float x = d_input[sampleIdx * (INPUT_H * INPUT_W) + inRow * INPUT_W + inCol];
          atomicAdd(&d_filterGrad[fr * FILTER_SIZE + fc], go * x);
        }
      }
    }

    idx += blockDim.x;
  }
}

// ----------------------- Aggregate grads (optional) -----------------------
__global__ void aggregateGradientsKernel(float* d_fcWeightGrad,
                                        float* d_biasFCGrad,
                                        float* d_filterGrad,
                                        float* d_biasConvGrad,
                                        int numBatches)
{
  // In this clean-room version, grads are already global atomics into single buffers,
  // so aggregation is a no-op. But we can average by number of samples if desired.
  int i = threadIdx.x;

  // average gradients by total samples (or batches). We'll use total samples for stability.
  float scale = 1.0f / float(NUM_SAMPLES);

  if (i < FLAT_SIZE) d_fcWeightGrad[i] *= scale;
  if (i == 0) {
    d_biasFCGrad[0]   *= scale;
    d_biasConvGrad[0] *= scale;
  }
  if (i < FILTER_SIZE * FILTER_SIZE) d_filterGrad[i] *= scale;
}

// ----------------------- Update weights -----------------------
__global__ void updateWeightsKernel(float* d_filter,
                                   float* d_biasConv,
                                   float* d_fcWeight,
                                   float* d_biasFC,
                                   const float* __restrict__ d_filterGrad,
                                   const float* __restrict__ d_biasConvGrad,
                                   const float* __restrict__ d_fcWeightGrad,
                                   const float* __restrict__ d_biasFCGrad,
                                   float lr)
{
  int i = threadIdx.x;

  if (i < FILTER_SIZE * FILTER_SIZE) {
    d_filter[i] -= lr * d_filterGrad[i];
  }
  if (i < FLAT_SIZE) {
    d_fcWeight[i] -= lr * d_fcWeightGrad[i];
  }
  if (i == 0) {
    d_biasConv[0] -= lr * d_biasConvGrad[0];
    d_biasFC[0]   -= lr * d_biasFCGrad[0];
  }
}

// ----------------------- Host: main -----------------------
int main()
{
  const size_t inputCount = NUM_SAMPLES * INPUT_H * INPUT_W;
  const size_t labelCount = NUM_SAMPLES;

  const size_t convCount  = NUM_SAMPLES * (CONV_OUT_H * CONV_OUT_W);
  const size_t fcOutCount = NUM_SAMPLES;

  // Host data
  float* h_input  = new float[inputCount];
  float* h_labels = new float[labelCount];

  // Random init data + binary labels
  for (int n = 0; n < NUM_SAMPLES; n++) {
    h_labels[n] = (std::rand() % 2) ? 1.0f : 0.0f;
    for (int i = 0; i < INPUT_H * INPUT_W; i++) {
      h_input[n * (INPUT_H * INPUT_W) + i] = float(std::rand()) / float(RAND_MAX);
    }
  }

  // Device allocations
  float *d_input=nullptr, *d_labels=nullptr;
  float *d_filter=nullptr, *d_biasConv=nullptr;
  float *d_convOut=nullptr;

  float *d_fcWeight=nullptr, *d_biasFC=nullptr;
  float *d_fcOut=nullptr;

  float *d_lossBatch=nullptr;    // per mini-batch loss array of size BATCH_SIZE
  float *d_fcOutGrad=nullptr;    // per sample
  float *d_convOutGrad=nullptr;  // per sample

  float *d_filterGrad=nullptr, *d_biasConvGrad=nullptr;
  float *d_fcWeightGrad=nullptr, *d_biasFCGrad=nullptr;

  CUDA_CHECK(cudaMalloc(&d_input,      inputCount * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_labels,     labelCount * sizeof(float)));

  CUDA_CHECK(cudaMalloc(&d_filter,     (FILTER_SIZE*FILTER_SIZE) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_biasConv,   NUM_FILTERS * sizeof(float)));

  CUDA_CHECK(cudaMalloc(&d_convOut,    convCount * sizeof(float)));

  CUDA_CHECK(cudaMalloc(&d_fcWeight,   FLAT_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_biasFC,     FC_OUT * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_fcOut,      fcOutCount * sizeof(float)));

  CUDA_CHECK(cudaMalloc(&d_lossBatch,  BATCH_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_fcOutGrad,  fcOutCount * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_convOutGrad,convCount * sizeof(float)));

  CUDA_CHECK(cudaMalloc(&d_filterGrad,   (FILTER_SIZE*FILTER_SIZE) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_biasConvGrad, NUM_FILTERS * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_fcWeightGrad, FLAT_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_biasFCGrad,   FC_OUT * sizeof(float)));

  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_input,  h_input,  inputCount * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_labels, h_labels, labelCount * sizeof(float), cudaMemcpyHostToDevice));

  // Initialize weights
  float h_filter[FILTER_SIZE*FILTER_SIZE];
  float h_biasConv[NUM_FILTERS] = {0.0f};
  float* h_fcWeight = new float[FLAT_SIZE];
  float h_biasFC[FC_OUT] = {0.0f};

  for (int i = 0; i < FILTER_SIZE*FILTER_SIZE; i++) h_filter[i] = 0.01f * (float(std::rand()) / float(RAND_MAX));
  for (int i = 0; i < FLAT_SIZE; i++)              h_fcWeight[i] = 0.01f * (float(std::rand()) / float(RAND_MAX));

  CUDA_CHECK(cudaMemcpy(d_filter,   h_filter,   sizeof(h_filter), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_biasConv, h_biasConv, sizeof(h_biasConv), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_fcWeight, h_fcWeight, FLAT_SIZE*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_biasFC,   h_biasFC,   sizeof(h_biasFC), cudaMemcpyHostToDevice));

  delete[] h_fcWeight;

  const int numBatches = (NUM_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE;
  const float lr = 0.1f;

  dim3 block(256);
  dim3 grid(numBatches);

  std::cout << "Starting training for " << EPOCHS << " epochs\n";

  for (int epoch = 0; epoch < EPOCHS; epoch++) {
    // Clear gradients
    CUDA_CHECK(cudaMemset(d_filterGrad,   0, (FILTER_SIZE*FILTER_SIZE)*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_biasConvGrad, 0, NUM_FILTERS*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_fcWeightGrad, 0, FLAT_SIZE*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_biasFCGrad,   0, FC_OUT*sizeof(float)));

    float epochLossSum = 0.0f;
    int   epochLossCount = 0;

    for (int b = 0; b < numBatches; b++) {
      int batchOffset = b * BATCH_SIZE;

      // 1) Forward conv
      size_t shmem = (FILTER_SIZE*FILTER_SIZE) * sizeof(float);
      forwardConvKernel<<<1, 256, shmem>>>(d_input, d_filter, d_biasConv, d_convOut, batchOffset);
      CUDA_CHECK(cudaGetLastError());

      // 2) Forward FC
      forwardFCKernel<<<1, 256>>>(d_convOut, d_fcWeight, d_biasFC, d_fcOut, batchOffset);
      CUDA_CHECK(cudaGetLastError());

      // 3) Loss + dL/dy
      computeLossAndDoutKernel<<<1, 256>>>(d_fcOut, d_labels, d_lossBatch, d_fcOutGrad, batchOffset);
      CUDA_CHECK(cudaGetLastError());

      // 4) Backward FC (compute grads + convOutGrad)
      backwardFCKernel2<<<1, 256>>>(d_convOut, d_fcWeight, d_fcOut, d_fcOutGrad,
                                    d_fcWeightGrad, d_biasFCGrad, d_convOutGrad, batchOffset);
      CUDA_CHECK(cudaGetLastError());

      // 5) Backward conv (filter + bias grads)
      backwardConvKernel<<<1, 256>>>(d_input, d_convOutGrad, d_filterGrad, d_biasConvGrad, batchOffset);
      CUDA_CHECK(cudaGetLastError());

      // Optional: pull batch loss to host to print an average
      float h_loss[BATCH_SIZE] = {0};
      CUDA_CHECK(cudaMemcpy(h_loss, d_lossBatch, BATCH_SIZE*sizeof(float), cudaMemcpyDeviceToHost));
      for (int i = 0; i < BATCH_SIZE; i++) {
        // last batch may include “extra” entries if NUM_SAMPLES not multiple of BATCH_SIZE
        if (batchOffset + i < NUM_SAMPLES) {
          epochLossSum += h_loss[i];
          epochLossCount++;
        }
      }
    }

    // 6) Aggregate (average) gradients
    int threads = 256;
    aggregateGradientsKernel<<<1, threads>>>(d_fcWeightGrad, d_biasFCGrad, d_filterGrad, d_biasConvGrad, numBatches);
    CUDA_CHECK(cudaGetLastError());

    // 7) Update weights
    updateWeightsKernel<<<1, 256>>>(d_filter, d_biasConv, d_fcWeight, d_biasFC,
                                    d_filterGrad, d_biasConvGrad, d_fcWeightGrad, d_biasFCGrad, lr);
    CUDA_CHECK(cudaGetLastError());

    float avgLoss = (epochLossCount > 0) ? (epochLossSum / float(epochLossCount)) : 0.0f;
    std::cout << "Epoch " << epoch << " average loss = " << avgLoss << "\n";
  }

  std::cout << "Training completed.\n";

  // Cleanup
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_labels));
  CUDA_CHECK(cudaFree(d_filter));
  CUDA_CHECK(cudaFree(d_biasConv));
  CUDA_CHECK(cudaFree(d_convOut));
  CUDA_CHECK(cudaFree(d_fcWeight));
  CUDA_CHECK(cudaFree(d_biasFC));
  CUDA_CHECK(cudaFree(d_fcOut));
  CUDA_CHECK(cudaFree(d_lossBatch));
  CUDA_CHECK(cudaFree(d_fcOutGrad));
  CUDA_CHECK(cudaFree(d_convOutGrad));
  CUDA_CHECK(cudaFree(d_filterGrad));
  CUDA_CHECK(cudaFree(d_biasConvGrad));
  CUDA_CHECK(cudaFree(d_fcWeightGrad));
  CUDA_CHECK(cudaFree(d_biasFCGrad));

  delete[] h_input;
  delete[] h_labels;

  return 0;
}
