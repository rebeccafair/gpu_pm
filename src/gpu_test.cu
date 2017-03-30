#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "eventReader.h"
#include "patternReader.h"
#include "tools.h"

#include "gpu_test.h"

using namespace std;

void copyContextToGpu(const PatternContainer& p, const EventContainer& e, GpuContext& ctx);
void runTestKernel(const PatternContainer& p, const EventContainer& e, GpuContext& ctx);
void deleteGpuContext(GpuContext& ctx);
__global__ void testKernel(const int *hashId, const unsigned int *hashIdEventBegin, int *hashId_out, int N);


void copyContextToGpu(const PatternContainer& p, const EventContainer& e, GpuContext& ctx) {
    cudaError_t err = cudaSuccess;

    // For all group/event begins, calculate indices that are pointed to
    vector<unsigned int> h_hitArrayGroupIndices = pointerToIndex(p.hitArrayGroupBegin, p.hitArray);
    vector<unsigned int> h_hashIdEventIndices = pointerToIndex(e.hashIdEventBegin, e.hashId);
    vector<unsigned int> h_nHitsEventIndices = pointerToIndex(e.nHitsEventBegin, e.nHits);
    vector<unsigned int> h_hitDataEventIndices = pointerToIndex(e.hitDataEventBegin, e.hitData);

    // Calculate size for all arrays that will be transferred
    size_t hashId_array_size = sizeof(int)*p.hashId_array.size();
    size_t hitArray_size = sizeof(unsigned char)*p.hitArray.size();
    size_t hitArrayGroupIndices_size = sizeof(unsigned int)*h_hitArrayGroupIndices.size();
    size_t hashId_size = sizeof(int)*e.hashId.size();
    size_t hashIdEventIndices_size = sizeof(unsigned int)*h_hashIdEventIndices.size();
    size_t nHits_size = sizeof(unsigned int)*e.nHits.size();
    size_t nHitsEventIndices_size = sizeof(unsigned int)*h_nHitsEventIndices.size();
    size_t hitData_size = sizeof(unsigned char)*e.hitData.size();
    size_t hitDataEventIndices_size = sizeof(unsigned int)*h_hitDataEventIndices.size();
    size_t nEventMatches_size = sizeof(int)*e.header.nEvents;

    // Create timer events
    cudaEvent_t start;
    err = cudaEventCreate(&start);
    if (err != cudaSuccess) cerr << "Error: failed to create timer start event\n" << cudaGetErrorString(err) << endl;
    cudaEvent_t stop;
    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) cerr << "Error: failed to create timer stop event\n" << cudaGetErrorString(err) << endl;

    // Record timer start event
    err = cudaEventRecord(start, NULL);
    if (err != cudaSuccess) cerr << "Error: failed to record timer start event\n" << cudaGetErrorString(err) << endl;

    // Allocate space for arrays on device
    err = cudaMalloc((void ** )&ctx.d_hashId_array, hashId_array_size);
    if (err != cudaSuccess) cerr << "Error: device memory not successfully allocated for d_hashId_array\n" << cudaGetErrorString(err) << endl;
    err = cudaMalloc((void ** )&ctx.d_hitArray, hitArray_size);
    if (err != cudaSuccess) cerr << "Error: device memory not successfully allocated for d_hitArray\n" << cudaGetErrorString(err) << endl;
    err = cudaMalloc((void ** )&ctx.d_hitArrayGroupIndices, hitArrayGroupIndices_size);
    if (err != cudaSuccess) cerr << "Error: device memory not successfully allocated for d_hitArrayGroupIndices\n" << cudaGetErrorString(err) << endl;
    err = cudaMalloc((void ** )&ctx.d_hashId, hashId_size);
    if (err != cudaSuccess) cerr << "Error: device memory not successfully allocated for d_hashId\n" << cudaGetErrorString(err) << endl;
    err = cudaMalloc((void ** )&ctx.d_hashIdEventIndices, hashIdEventIndices_size);
    if (err != cudaSuccess) cerr << "Error: device memory not successfully allocated for d_hashIdEventIndices\n" << cudaGetErrorString(err) << endl;
    err = cudaMalloc((void ** )&ctx.d_nHits, nHits_size);
    if (err != cudaSuccess) cerr << "Error: device memory not successfully allocated for d_nHits\n" << cudaGetErrorString(err) << endl;
    err = cudaMalloc((void ** )&ctx.d_nHitsEventIndices, nHitsEventIndices_size);
    if (err != cudaSuccess) cerr << "Error: device memory not successfully allocated for d_nHitsEventIndices\n" << cudaGetErrorString(err) << endl;
    err = cudaMalloc((void ** )&ctx.d_hitData, hitData_size);
    if (err != cudaSuccess) cerr << "Error: device memory not successfully allocated for d_hitData\n" << cudaGetErrorString(err) << endl;
    err = cudaMalloc((void ** )&ctx.d_hitDataEventIndices, hitDataEventIndices_size);
    if (err != cudaSuccess) cerr << "Error: device memory not successfully allocated for d_hitDataEventIndices\n" << cudaGetErrorString(err) << endl;
    err = cudaMalloc((void ** )&ctx.d_nEventMatches, nEventMatches_size);
    if (err != cudaSuccess) cerr << "Error: device memory not successfully allocated for d_nEventMatches\n" << cudaGetErrorString(err) << endl;

    // Copy input arrays to device
    err = cudaMemcpy(ctx.d_hashId_array, &p.hashId_array[0], hashId_array_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) cerr << "Error: hashId_array not copied to device\n" << cudaGetErrorString(err) << endl;
    err = cudaMemcpy(ctx.d_hitArray, &p.hitArray[0], hitArray_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) cerr << "Error: hitArray not copied to device\n" << cudaGetErrorString(err) << endl;
    err = cudaMemcpy(ctx.d_hitArrayGroupIndices, &h_hitArrayGroupIndices[0], hitArrayGroupIndices_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) cerr << "Error: hitArrayGroupIndices not copied to device\n" << cudaGetErrorString(err) << endl;
    err = cudaMemcpy(ctx.d_hashId, &e.hashId[0], hashId_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) cerr << "Error: hashId not copied to device\n" << cudaGetErrorString(err) << endl;
    err = cudaMemcpy(ctx.d_hashIdEventIndices, &h_hashIdEventIndices[0], hashIdEventIndices_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) cerr << "Error: hashIdEventIndices not copied to device\n" << cudaGetErrorString(err) << endl;
    err = cudaMemcpy(ctx.d_nHits, &e.nHits[0], nHits_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) cerr << "Error: nHits not copied to device\n" << cudaGetErrorString(err) << endl;
    err = cudaMemcpy(ctx.d_nHitsEventIndices, &h_nHitsEventIndices[0], nHitsEventIndices_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) cerr << "Error: nHitsEventIndices not copied to device\n" << cudaGetErrorString(err) << endl;
    err = cudaMemcpy(ctx.d_hitData, &e.hitData[0], hitData_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) cerr << "Error: hitData not copied to device\n" << cudaGetErrorString(err) << endl;
    err = cudaMemcpy(ctx.d_hitDataEventIndices, &h_hitDataEventIndices[0], hitDataEventIndices_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) cerr << "Error: hitDataEventIndices not copied to device\n" << cudaGetErrorString(err) << endl;

    // Initialise output arrays
    err = cudaMemset(ctx.d_nEventMatches, 0, nEventMatches_size);
    if (err != cudaSuccess) cerr << "Error: d_nEventMatches not initialised to zero" << endl;

    // Record timer stop event
    err = cudaEventRecord(stop, NULL);
    if (err != cudaSuccess) cerr << "Error: failed to record timer stop event\n" << cudaGetErrorString(err) << endl;
    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) cerr << "Error: failed to synchronize on stop event\n" << cudaGetErrorString(err) << endl;

    // Calculate elapsed time
    float msecTotal = 0.0f;
    err = cudaEventElapsedTime(&msecTotal, start, stop);
    if (err != cudaSuccess) cerr << "Error: failed to get elapsed time between events\n" << cudaGetErrorString(err) << endl;
    cout << "Allocated and copied arrays to device in " << msecTotal << " ms" << endl;
};


void runTestKernel(const PatternContainer& p, const EventContainer& e, GpuContext& ctx) {
    cudaError_t err = cudaSuccess;

    // Calculate number of threads/blocks required
    int N = e.header.nEvents;
    int threadsPerBlock = 256;
    int blocksPerGrid = (N/threadsPerBlock) + 1; 

    // Allocate and initialise vector to store result
    size_t nEventMatches_size = sizeof(int)*e.header.nEvents;
    vector<int> nEventMatches(nEventMatches_size);

    // Create timer events
    cudaEvent_t start;
    err = cudaEventCreate(&start);
    if (err != cudaSuccess) cerr << "Error: failed to create timer start event\n" << cudaGetErrorString(err) << endl;
    cudaEvent_t stop;
    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) cerr << "Error: failed to create timer stop event\n" << cudaGetErrorString(err) << endl;

    // Record timer start event
    err = cudaEventRecord(start, NULL);
    if (err != cudaSuccess) cerr << "Error: failed to record timer start event\n" << cudaGetErrorString(err) << endl;

    // Run the kernel nRepeats times
    int nRepeats = 100;
    for (int i = 0; i < nRepeats; i++) {
        testKernel<<<blocksPerGrid, threadsPerBlock>>>(ctx.d_hashId, ctx.d_hashIdEventIndices, ctx.d_nEventMatches, N);
    }

    // Record timer stop event
    err = cudaEventRecord(stop, NULL);
    if (err != cudaSuccess) cerr << "Error: failed to record timer stop event\n" << cudaGetErrorString(err) << endl;
    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) cerr << "Error: failed to synchronize on stop event\n" << cudaGetErrorString(err) << endl;

    // Calculate elapsed time
    float msecTotal = 0.0f;
    err = cudaEventElapsedTime(&msecTotal, start, stop);
    if (err != cudaSuccess) cerr << "Error: failed to get elapsed time between events\n" << cudaGetErrorString(err) << endl;
    cout << "Ran kernel " << nRepeats << " times in " << msecTotal << " ms" << endl;
    float msecPerEvent = msecTotal/nRepeats;
    cout << "Average kernel time is " << msecPerEvent << " ms" << endl;

    // Copy result back to host memory
    err = cudaMemcpy(&nEventMatches[0], ctx.d_nEventMatches, nEventMatches_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) cerr << "Error: d_nEventMatches not copied from device to host" << endl;

};


void deleteGpuContext(GpuContext& ctx) {
    cudaError_t err = cudaSuccess;

    // Create timer events
    cudaEvent_t start;
    err = cudaEventCreate(&start);
    if (err != cudaSuccess) cerr << "Error: failed to create timer start event\n" << cudaGetErrorString(err) << endl;
    cudaEvent_t stop;
    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) cerr << "Error: failed to create timer stop event\n" << cudaGetErrorString(err) << endl;

    // Record timer start event
    err = cudaEventRecord(start, NULL);
    if (err != cudaSuccess) cerr << "Error: failed to record timer start event\n" << cudaGetErrorString(err) << endl;

    // Free device memory
    err = cudaFree(ctx.d_hashId_array);
    if (err != cudaSuccess) cerr << "Error: failed to free device memory for d_hashId_array\n" << cudaGetErrorString(err) << endl;
    err = cudaFree(ctx.d_hitArray);
    if (err != cudaSuccess) cerr << "Error: failed to free device memory for d_hitArray\n" << cudaGetErrorString(err) << endl;
    err = cudaFree(ctx.d_hitArrayGroupIndices);
    if (err != cudaSuccess) cerr << "Error: failed to free device memory for d_hitArrayGroupIndices\n" << cudaGetErrorString(err) << endl;
    err = cudaFree(ctx.d_hashId);
    if (err != cudaSuccess) cerr << "Error: failed to free device memory for d_hashId\n" << cudaGetErrorString(err) << endl;
    err = cudaFree(ctx.d_hashIdEventIndices);
    if (err != cudaSuccess) cerr << "Error: failed to free device memory for d_hashIdEventIndices\n" << cudaGetErrorString(err) << endl;
    err = cudaFree(ctx.d_nHits);
    if (err != cudaSuccess) cerr << "Error: failed to free device memory for d_nHits\n" << cudaGetErrorString(err) << endl;
    err = cudaFree(ctx.d_nHitsEventIndices);
    if (err != cudaSuccess) cerr << "Error: failed to free device memory for d_nHitsEventIndices\n" << cudaGetErrorString(err) << endl;
    err = cudaFree(ctx.d_hitData);
    if (err != cudaSuccess) cerr << "Error: failed to free device memory for d_hitData\n" << cudaGetErrorString(err) << endl;
    err = cudaFree(ctx.d_hitDataEventIndices);
    if (err != cudaSuccess) cerr << "Error: failed to free device memory for d_hitDataEventIndices\n" << cudaGetErrorString(err) << endl;
    err = cudaFree(ctx.d_nEventMatches);
    if (err != cudaSuccess) cerr << "Error: failed to free device memory for d_nEventMatches\n" << cudaGetErrorString(err) << endl;

    // Record timer stop event
    err = cudaEventRecord(stop, NULL);
    if (err != cudaSuccess) cerr << "Error: failed to record timer stop event\n" << cudaGetErrorString(err) << endl;
    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) cerr << "Error: failed to synchronize on stop event\n" << cudaGetErrorString(err) << endl;

    // Calculate elapsed time
    float msecTotal = 0.0f;
    err = cudaEventElapsedTime(&msecTotal, start, stop);
    if (err != cudaSuccess) cerr << "Error: failed to get elapsed time between events\n" << cudaGetErrorString(err) << endl;

    cout << "Freed device memory in " << msecTotal << " ms" << endl;

};


__global__ void testKernel(const int *hashId, const unsigned int *hashIdEventBegin, int *nEventMatches, int N) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N) {
        int nColl = hashIdEventBegin[i+1] - hashIdEventBegin[i];
        for (int j = 0; j < nColl; j++) {
              if (hashId[hashIdEventBegin[i] + j] == 3005) {
                  nEventMatches[i]++;
              }
        }
    }

}
