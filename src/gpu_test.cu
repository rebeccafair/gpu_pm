#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#include "eventReader.h"
#include "patternReader.h"
#include "tools.h"

#include "gpu_test.h"

using namespace std;

void copyContextToGpu(const PatternContainer& p, const EventContainer& e, GpuContext& ctx);
__global__ void testKernel(const int *hashId, const unsigned int *hashIdEventBegin, int *hashId_out, int N);


void copyContextToGpu(const PatternContainer& p, const EventContainer& e, GpuContext& ctx) {
    cudaError_t err = cudaSuccess;

    // For all group/event begins, calculate indices that are pointed to
    vector<unsigned int> h_hashIdEventIndices = pointerToIndex(e.hashIdEventBegin, e.hashId);

    // Allocate space for hashIds on GPU
    size_t hashId_size = sizeof(int)*e.hashId.size();
    size_t hashIdEventIndices_size = sizeof(unsigned int)*h_hashIdEventIndices.size();
    err = cudaMalloc((void ** )&ctx.d_hashId, hashId_size);
    if (err != cudaSuccess) cerr << "Error: device memory not allocated for ctx.d_hashId" << endl;
    err = cudaMalloc((void ** )&ctx.d_hashIdEventIndices, hashIdEventIndices_size);
    if (err != cudaSuccess) cerr << "Error: device memory not allocated for ctx.d_hashIdEventIndices" << endl;

    // Copy hashIds to GPU
    err = cudaMemcpy(ctx.d_hashId, &e.hashId[0], hashId_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) cerr << "Error: hashId not copied to device" << endl;
    err = cudaMemcpy(ctx.d_hashIdEventIndices, &h_hashIdEventIndices[0], hashIdEventIndices_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) cerr << "Error: hashIdEventIndices not copied to device" << endl;

    // Calculate number of threads/blocks required
    int N = e.header.nEvents;
    int threadsPerBlock = 256;
    int blocksPerGrid = (N/threadsPerBlock) + 1; 

    // Allocate and initialise vector to store result
    vector<int> hashId_out(hashId_size);
    int* d_hashId_out;
    err = cudaMalloc((void ** )&d_hashId_out, hashId_size);
    if (err != cudaSuccess) cerr << "Error: device memory not allocated for d_hashId_out" << endl;
    err = cudaMemset(d_hashId_out, 0, hashId_size);
    if (err != cudaSuccess) cerr << "Error: d_hashId_out not initialised to zero" << endl;

    // Launch the kernel
    testKernel<<<blocksPerGrid, threadsPerBlock>>>(ctx.d_hashId, ctx.d_hashIdEventIndices, d_hashId_out, N);

    // Copy result back to host memory
    err = cudaMemcpy(&hashId_out[0], d_hashId_out, hashId_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) cerr << "Error: d_hashId_out not copied from device to host" << endl;

    // Free device memory
    err = cudaFree(ctx.d_hashId);
    if (err != cudaSuccess) cerr << "Failed to free device memory for d_hashId (error code " << cudaGetErrorString(err) << ")!" << endl;
    err = cudaFree(ctx.d_hashIdEventIndices);
    if (err != cudaSuccess) cerr << "Failed to free device memory for d_hashIdEventIndices (error code " << cudaGetErrorString(err) << ")!" << endl;
    err = cudaFree(ctx.d_hashId_out);
    if (err != cudaSuccess) cerr << "Failed to free device memory for d_hashId_out (error code " << cudaGetErrorString(err) << ")!" << endl;

};


__global__ void testKernel(const int *hashId, const unsigned int *hashIdEventBegin, int *hashId_out, int N) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N) {
        int nColl = hashIdEventBegin[i+1] - hashIdEventBegin[i];
        for (int j = 0; j < nColl; j++) {
              hashId_out[hashIdEventBegin[i] + j] = hashId[hashIdEventBegin[i] + j] + 1;
        }
    }

}
