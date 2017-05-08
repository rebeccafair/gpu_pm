#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "eventReader.h"
#include "patternReader.h"
#include "matchResults.h"
#include "tools.h"

#include "gpuMatcher.h"

using namespace std;

void createGpuContext(const PatternContainer& p, const EventContainer& e, GpuContext& ctx) {
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
    size_t matchingPattIds_size = sizeof(int)*10000;

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
    err = cudaMalloc((void ** )&ctx.d_matchingPattIds, matchingPattIds_size);
    if (err != cudaSuccess) cerr << "Error: device memory not successfully allocated for d_matchingPattIds\n" << cudaGetErrorString(err) << endl;
    err = cudaMalloc((void ** )&ctx.d_nMatches, sizeof(int));
    if (err != cudaSuccess) cerr << "Error: device memory not successfully allocated for d_nMatches\n" << cudaGetErrorString(err) << endl;

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
    err = cudaMemset(ctx.d_matchingPattIds, 0, matchingPattIds_size);
    if (err != cudaSuccess) cerr << "Error: d_matchingPattIds not initialised to zero" << endl;
    err = cudaMemset(ctx.d_nMatches, 0, sizeof(int));
    if (err != cudaSuccess) cerr << "Error: d_matchingPattIds not initialised to zero" << endl;

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


void runMatchByBlockSingle(const PatternContainer& p, const EventContainer& e, GpuContext& ctx, MatchResults& mr, int threadsPerBlock) {
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

    // Calculate number of blocks required
    int blocksPerGrid = p.header.nGroups;

    // Run kernel for each event
    int nPattMatchesSize = threadsPerBlock/p.header.nLayers*sizeof(unsigned int);
    for (int i = 0; i < e.header.nEvents; i++ ) {
    //for (int i = 100; i < 125; i++ ) {
        matchByBlockSingle<<<blocksPerGrid, threadsPerBlock, nPattMatchesSize>>>(ctx.d_hashId_array, ctx.d_hitArray, ctx.d_hitArrayGroupIndices,
                                                                                 ctx.d_hashId, ctx.d_hashIdEventIndices, ctx.d_nHits,
                                                                                 ctx.d_nHitsEventIndices, ctx.d_hitData, ctx.d_hitDataEventIndices,
                                                                                 ctx.d_matchingPattIds, ctx.d_nMatches, i);
    }
    cudaDeviceSynchronize();

    // Record timer stop event
    err = cudaEventRecord(stop, NULL);
    if (err != cudaSuccess) cerr << "Error: failed to record timer stop event\n" << cudaGetErrorString(err) << endl;
    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) cerr << "Error: failed to synchronize on stop event\n" << cudaGetErrorString(err) << endl;

    // Calculate elapsed time
    float msecTotal = 0.0f;
    err = cudaEventElapsedTime(&msecTotal, start, stop);
    if (err != cudaSuccess) cerr << "Error: failed to get elapsed time between events\n" << cudaGetErrorString(err) << endl;
    cout << "Ran kernel " << e.header.nEvents << " times in " << msecTotal << " ms" << endl;
    float msecPerEvent = msecTotal/e.header.nEvents;
    cout << "Average matchByBlockSingle kernel time with " << threadsPerBlock << " threads is " << msecPerEvent << " ms" << endl;

    // Copy result back to host memory
    err = cudaMemcpy(&mr.nMatches, ctx.d_nMatches, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) cerr << "Error: d_nMatches not copied from device to host" << endl;
    mr.patternIds.resize(mr.nMatches);
    err = cudaMemcpy(&mr.patternIds[0], ctx.d_matchingPattIds, mr.nMatches*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) cerr << "Error: d_matchingPattIds not copied from device to host" << endl;
};

void runMatchByBlockMulti(const PatternContainer& p, const EventContainer& e, GpuContext& ctx, MatchResults& mr, int threadsPerBlock, int nBlocks) {
    cudaError_t err = cudaSuccess;

    // Distribute groups to blocks according to number of blocks
    vector<int> blockBegin(nBlocks,-1);
    vector<int> nGroupsInBlock(nBlocks,0);
    vector<int> groups(p.header.nGroups,-1);
    distributeWork(nBlocks, p, blockBegin, nGroupsInBlock, groups);

    // Allocate and copy information about block/group assignments to device
    size_t blockBegin_size = sizeof(int)*blockBegin.size();
    err = cudaMalloc((void ** )&ctx.d_blockBegin, blockBegin_size);
    if (err != cudaSuccess) cerr << "Error: device memory not successfully allocated for d_blockBegin\n" << cudaGetErrorString(err) << endl;
    err = cudaMemcpy(ctx.d_blockBegin, &blockBegin[0], blockBegin_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) cerr << "Error: blockBegin not copied to device\n" << cudaGetErrorString(err) << endl;
    size_t nGroupsInBlock_size = sizeof(int)*nGroupsInBlock.size();
    err = cudaMalloc((void ** )&ctx.d_nGroupsInBlock, nGroupsInBlock_size);
    if (err != cudaSuccess) cerr << "Error: device memory not successfully allocated for d_nGroupsInBlock\n" << cudaGetErrorString(err) << endl;
    err = cudaMemcpy(ctx.d_nGroupsInBlock, &nGroupsInBlock[0], nGroupsInBlock_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) cerr << "Error: nGroupsInBlock not copied to device\n" << cudaGetErrorString(err) << endl;
    size_t groups_size = sizeof(int)*groups.size();
    err = cudaMalloc((void ** )&ctx.d_groups, groups_size);
    if (err != cudaSuccess) cerr << "Error: device memory not successfully allocated for d_groups\n" << cudaGetErrorString(err) << endl;
    err = cudaMemcpy(ctx.d_groups, &groups[0], groups_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) cerr << "Error: groups not copied to device\n" << cudaGetErrorString(err) << endl;

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

    // Run kernel for each event
    int nPattMatchesSize = threadsPerBlock/p.header.nLayers*sizeof(unsigned int);
    for (int i = 0; i < e.header.nEvents; i++ ) {
        matchByBlockMulti<<<nBlocks, threadsPerBlock, nPattMatchesSize>>>(ctx.d_hashId_array, ctx.d_hitArray, ctx.d_hitArrayGroupIndices,
                                                                          ctx.d_hashId, ctx.d_hashIdEventIndices, ctx.d_nHits,
                                                                          ctx.d_nHitsEventIndices, ctx.d_hitData, ctx.d_hitDataEventIndices,
                                                                          ctx.d_matchingPattIds, ctx.d_nMatches, i, ctx.d_blockBegin,
                                                                          ctx.d_nGroupsInBlock, ctx.d_groups);
    }
    cudaDeviceSynchronize();

    // Record timer stop event
    err = cudaEventRecord(stop, NULL);
    if (err != cudaSuccess) cerr << "Error: failed to record timer stop event\n" << cudaGetErrorString(err) << endl;
    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) cerr << "Error: failed to synchronize on stop event\n" << cudaGetErrorString(err) << endl;

    // Calculate elapsed time
    float msecTotal = 0.0f;
    err = cudaEventElapsedTime(&msecTotal, start, stop);
    if (err != cudaSuccess) cerr << "Error: failed to get elapsed time between events\n" << cudaGetErrorString(err) << endl;
    cout << "Ran kernel " << e.header.nEvents << " times in " << msecTotal << " ms" << endl;
    float msecPerEvent = msecTotal/e.header.nEvents;
    cout << "Average matchByBlockMulti kernel time with " << threadsPerBlock << " threads and " << nBlocks << " blocks is " << msecPerEvent << " ms" << endl;

    // Copy result back to host memory
    err = cudaMemcpy(&mr.nMatches, ctx.d_nMatches, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) cerr << "Error: d_nMatches not copied from device to host" << endl;
    mr.patternIds.resize(mr.nMatches);
    err = cudaMemcpy(&mr.patternIds[0], ctx.d_matchingPattIds, mr.nMatches*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) cerr << "Error: d_matchingPattIds not copied from device to host" << endl;

};

void distributeWork(int nBlocks, const PatternContainer& p, vector<int>& blockBegin, vector<int>& nGroupsInBlock, vector<int>& groups) {

    int maxPattsInBlock = (p.header.nHitPatt + nBlocks - 1)/nBlocks;

    cout << "Distributing work..." << endl;

    vector<int> nPattsInBlock(nBlocks,0);
    vector<int> assignedBlock(p.header.nGroups,-1);

    bool forward = true;
    int nextBlock = 0;
    // Loop through groups and determine number of groups in each block
    for (int g = 0; g < p.header.nGroups; g++) {
       // Loop over blocks and add group to block if number of patterns
       // in block is less than threshold. Alternately loop forward and
       // backward across patterns for better load balancing
       while ( assignedBlock[g] == -1) {
           if (forward) {
               for (int b = nextBlock; b < nBlocks; b++) {
                   if (nPattsInBlock[b] < maxPattsInBlock) {
                       nGroupsInBlock[b]++;
                       nPattsInBlock[b] += p.nPattInGrp[g];
                       assignedBlock[g] = b;
                       if (b != nBlocks - 1) { nextBlock = b + 1; } else { forward = !forward; }
                       break;
                   }
                   if (b == nBlocks - 1) { forward = !forward; } // Reverse loop if loop is completed with no assignments
               }
           } else {
               for (int b = nextBlock; b >= 0; b--) {
                   if (nPattsInBlock[b] < maxPattsInBlock) {
                       nGroupsInBlock[b]++;
                       nPattsInBlock[b] += p.nPattInGrp[g];
                       assignedBlock[g] = b;
                       if (b != 0) { nextBlock = b - 1; } else { forward = !forward; }
                       break;
                   }
                   if (b == 0) { forward = !forward; } // Reverse loop if loop is completed with no assignments
               }
           }
        }

    } // End loop over groups


    // Loop through blocks and point to first index in each group
    int nextIndex = 0;
    for (int b = 0; b < nBlocks; b++) {
        blockBegin[b] = nextIndex;
        nextIndex += nGroupsInBlock[b];
        nGroupsInBlock[b] = 0;
    }

    // Loop through groups again and assign them to correct block
    forward = true;
    nextBlock = 0;
    for (int g = 0; g < p.header.nGroups; g++) {
        int block = assignedBlock[g];
        groups[blockBegin[block] + nGroupsInBlock[block]] = g;
        nGroupsInBlock[block]++;
    }

    // Print assigned blocks
    /*for (int b = 0; b < nBlocks; b++) {
        cout << "block: " << b << " nGroupsInBlock: " << nGroupsInBlock[b] << " nPattsInBlock: " << nPattsInBlock[b] << endl;
        cout << "groups: ";
        for (int g = 0; g < nGroupsInBlock[b]; g++) {
            cout << groups[blockBegin[b] + g] << " ";
        }
        cout << endl;
    }
    */

};

void runMatchByLayer(const PatternContainer& p, const EventContainer& e, GpuContext& ctx, MatchResults& mr, int threadsPerBlock) {
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

    // Calculate number of blocks required
    int blocksPerGrid = p.header.nGroups;

    // Run kernel for each event
    int nPattMatchesSize = p.nPattInGrp[0]*sizeof(unsigned int);
    for (int i = 0; i < e.header.nEvents; i++ ) {
        matchByLayer<<<blocksPerGrid, threadsPerBlock, nPattMatchesSize>>>(ctx.d_hashId_array, ctx.d_hitArray, ctx.d_hitArrayGroupIndices,
                                                                           ctx.d_hashId, ctx.d_hashIdEventIndices, ctx.d_nHits,
                                                                           ctx.d_nHitsEventIndices, ctx.d_hitData, ctx.d_hitDataEventIndices,
                                                                           ctx.d_matchingPattIds, ctx.d_nMatches, i);
    }
    cudaDeviceSynchronize();

    // Record timer stop event
    err = cudaEventRecord(stop, NULL);
    if (err != cudaSuccess) cerr << "Error: failed to record timer stop event\n" << cudaGetErrorString(err) << endl;
    err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) cerr << "Error: failed to synchronize on stop event\n" << cudaGetErrorString(err) << endl;

    // Calculate elapsed time
    float msecTotal = 0.0f;
    err = cudaEventElapsedTime(&msecTotal, start, stop);
    if (err != cudaSuccess) cerr << "Error: failed to get elapsed time between events\n" << cudaGetErrorString(err) << endl;
    cout << "Ran kernel " << e.header.nEvents << " times in " << msecTotal << " ms" << endl;
    float msecPerEvent = msecTotal/e.header.nEvents;
    cout << "Average matchByLayer kernel time with " << threadsPerBlock << " threads is " << msecPerEvent << " ms" << endl;

    // Copy result back to host memory
    err = cudaMemcpy(&mr.nMatches, ctx.d_nMatches, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) cerr << "Error: d_nMatches not copied from device to host" << endl;
    mr.patternIds.resize(mr.nMatches);
    err = cudaMemcpy(&mr.patternIds[0], ctx.d_matchingPattIds, mr.nMatches*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) cerr << "Error: d_matchingPattIds not copied from device to host" << endl;

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
    err = cudaFree(ctx.d_matchingPattIds);
    if (err != cudaSuccess) cerr << "Error: failed to free device memory for d_matchingPattIds\n" << cudaGetErrorString(err) << endl;

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

__global__ void matchByBlockSingle(const int *hashId_array, const unsigned char *hitArray, const unsigned int *hitArrayGroupIndices, 
                                   const int *hashId, const unsigned int *hashIdEventIndices, const unsigned int *nHits,
                                   const unsigned int *nHitsEventIndices, const unsigned char *hitData, const unsigned int *hitDataEventIndices, 
                                   int *matchingPattIds, int *nMatches, const int eventId) {
    const int nLayers = 8;
    const int nRequiredMatches = 7;
    const int nMaxRows = 11;
    const int nMaxColumns = 3;
    const int maxDcBits = 2;
    int grp = blockIdx.x;
    int lyr = threadIdx.x%nLayers;
    int row = threadIdx.x/nLayers;
    int lyrHashId = hashId_array[grp*nLayers + lyr];

    __shared__ unsigned int nHashMatches; // Number of group hashIds that match event collection hashIds
    __shared__ int matchingCollections[nLayers]; // Records which collection matches the group hashId of a certain layer
    __shared__ unsigned int collectionHit[(maxDcBits + 1)*nLayers]; // Records hits as a bit array for a matching collection

    if (threadIdx.x == 0) {
        nHashMatches = 0;
    }
    __syncthreads();

    // Get first nLayers threads to check if any hashIds are wildcards
    if (threadIdx.x < nLayers) {
        // Initialise matchingCollections to -1
        matchingCollections[lyr] = -1;
        // Automatically match if layer is wildcard
        if (lyrHashId == -1) {
            atomicAdd(&nHashMatches,1);
         }
    }

    // Get each thread to compare one hashId with one collection from
    // an event to check for a match
    int nColl = hashIdEventIndices[eventId+1] - hashIdEventIndices[eventId];
    int nLoops = ((nLayers*nColl)/blockDim.x) + 1;
    for (int n = 0; n < nLoops; n++) {
        int coll = n*blockDim.x/nLayers + threadIdx.x/nLayers;
        if (coll < nColl) {
            if (lyrHashId != -1) {
                if (hashId[hashIdEventIndices[eventId] + coll] == lyrHashId) {
                    atomicExch(&matchingCollections[lyr],coll);
                    atomicAdd(&nHashMatches,1);
                }
            }
        }
    }
    __syncthreads();

    if (nHashMatches >= nRequiredMatches) {

        // Initialise collectionHit[]
        if (threadIdx.x < nLayers*(maxDcBits + 1)) {
            collectionHit[threadIdx.x] = 0;
        }
        __syncthreads();

        // Loop through collection hits to find collection hit data
        int matchingColl = matchingCollections[lyr];
        const unsigned char *pHitData = &hitData[hitDataEventIndices[eventId]];
        for (int coll = 0; coll < matchingColl; coll++) {
            pHitData += nHits[nHitsEventIndices[eventId] + coll];
        }

        // Put hits into bit arrays
        unsigned char isPixel = ((*pHitData >> 7) & 1); // If bit 7 is 1, element is pixel, otherwise strip
        if (threadIdx.x < nLayers) {
            if (matchingColl != -1) {
                for (int hit = 0; hit < nHits[nHitsEventIndices[eventId] + matchingColl]; hit++) {
                    unsigned char hitPos = ((*(pHitData + hit) >> 2) & 31); // Get superstrip value if strip, or row value if pixel. Occupies bits 2-6
                    for (int dcBits = 0; dcBits <= maxDcBits; dcBits++) {
                        unsigned char maskedHitPos = hitPos &  (~((1 << dcBits) - 1)); // Mask hit value depending on dc bits. e.g. if dcBits = 2 AND with 11100
                        if (isPixel) { 
                            maskedHitPos += (*(pHitData + hit) & 3)*nMaxRows; // If pixel, need to adjust value according to column
                        }
                        unsigned char bitMask = 1; // Get bitmask  e.g. ...01111 for dcBits = 2
                        if (dcBits == 1) {
                            bitMask = 3;
                        } else if (dcBits == 2) {
                            bitMask = 15;
                        }
                        collectionHit[dcBits*nLayers + lyr] |= (bitMask << maskedHitPos); 
                    }
                }
            }
        }

        // Loop as many times as necessary for all threads to cover all patterns
        int nPattInGrp = (hitArrayGroupIndices[grp + 1] - hitArrayGroupIndices[grp])/nLayers;
        nLoops = ((nPattInGrp*nLayers)/blockDim.x) + 1;
        extern __shared__ unsigned int nPattMatches[];

        for (int n = 0; n < nLoops; n++) {

            int pattNum = n*blockDim.x/nLayers + row;

            // Initialise nPattMatches to zero
            if (lyr == 0) {
                nPattMatches[row] = 0;
            }
            __syncthreads();

            // Only continue if thread isn't overflowing the number of patterns in the group
            if ( pattNum < nPattInGrp) {

                if (matchingColl != -1) {

                    // Get pattern hit data
                    unsigned char pattHit = hitArray[hitArrayGroupIndices[grp] + n*blockDim.x + threadIdx.x];
                    unsigned char hitPos = ((pattHit >> 2) & 31); // Get superstrip position if strip, or row if pixel. Occupies bits 2-6
                    unsigned char dcBits = (pattHit & 3);
                    if (dcBits == 3) { dcBits = 2; }
                    if (isPixel) {
                        unsigned char pattPixCol = hitPos/nMaxRows;
                        unsigned char pattPixRow = hitPos%nMaxRows;
                        hitPos = nMaxRows*pattPixCol + pattPixRow;
                    }
                    if ( ((1 << hitPos) & collectionHit[dcBits*nLayers + lyr]) > 0 ) {
                        atomicAdd(&nPattMatches[row],1);
                    }

                } else if ( lyrHashId == -1 ) {
                    atomicAdd(&nPattMatches[row],1);
                }

                __syncthreads();
                // Output matching pattern ids to array
                if (lyr == 0) {
                    if (nPattMatches[row] >= nRequiredMatches) {
                        int i = atomicAdd(nMatches,1);
                        int pattId = ((hitArrayGroupIndices[grp] - hitArrayGroupIndices[0])/nLayers) + pattNum;
                        matchingPattIds[i] = pattId;
                    }
                }

            } // End if ( pattNum < nPattInGrp)

        } // End nLoops

    } // End if (nHashMatches >= nRequiredMatches)

}

__global__ void matchByBlockMulti(const int *hashId_array, const unsigned char *hitArray, const unsigned int *hitArrayGroupIndices, 
                                  const int *hashId, const unsigned int *hashIdEventIndices, const unsigned int *nHits,
                                  const unsigned int *nHitsEventIndices, const unsigned char *hitData, const unsigned int *hitDataEventIndices, 
                                  int *matchingPattIds, int *nMatches, const int eventId, const int *blockBegin, const int *nGroupsInBlock,
                                  const int *groups) {

    const int nLayers = 8;
    const int maxGroupsInBlock = 60;
    const int nRequiredMatches = 7;
    const int nMaxRows = 11;

    __shared__ unsigned int nHashMatches[maxGroupsInBlock]; // Number of group hashIds that match event collection hashIds for each group
    __shared__ int matchingCollections[nLayers]; // Records which collection matches the group hashId of a certain layer

    // Initialise match counters
    if (threadIdx.x < nGroupsInBlock[blockIdx.x]) {
        nHashMatches[threadIdx.x] = 0;
    }
    __syncthreads();

    // Check if any hashIds are wildcards
    int lyr = threadIdx.x%nLayers;
    int nLoops = (nGroupsInBlock[blockIdx.x]*nLayers/blockDim.x) + 1;
    for (int n = 0; n < nLoops; n++) {
        int grpInBlock = (n*blockDim.x + threadIdx.x)/nLayers;
        if (grpInBlock < nGroupsInBlock[blockIdx.x]) {
            // Initialise matchingCollections to -1
            atomicExch(&matchingCollections[n*blockDim.x + threadIdx.x],-1);
            int grp = groups[blockBegin[blockIdx.x] + grpInBlock];
            int lyrHashId = hashId_array[grp*nLayers + lyr];
            // Automatically match if layer is wildcard
            if (lyrHashId == -1) {
                atomicAdd(&nHashMatches[grpInBlock],1);
             }
        }

    }

    __syncthreads();

    // Get each thread to compare one hashId from each group with one collection from
    // an event to check for a match
    int nColl = hashIdEventIndices[eventId+1] - hashIdEventIndices[eventId];
    nLoops = (nGroupsInBlock[blockIdx.x]*nLayers*nColl/blockDim.x) + 1;
    for (int n = 0; n < nLoops; n++) {
        int grpInBlock = (n*blockDim.x + threadIdx.x)/(nColl*nLayers);
        int coll = ((n*blockDim.x + threadIdx.x)/nLayers)%nColl;
        if (grpInBlock < nGroupsInBlock[blockIdx.x]) {
            int grp = groups[blockBegin[blockIdx.x] + grpInBlock];
            int lyrHashId = hashId_array[grp*nLayers + lyr];
            if (lyrHashId != -1) {
                if (hashId[hashIdEventIndices[eventId] + coll] == lyrHashId) {
                    atomicExch(&matchingCollections[grpInBlock*nLayers + lyr],coll);
                    atomicAdd(&nHashMatches[grpInBlock],1);
                }
            }
        }
    }
    __syncthreads();

    int row = threadIdx.x/nLayers;
    for (int i = 0; i < nGroupsInBlock[blockIdx.x]; i++) {
        int grp = groups[blockBegin[blockIdx.x] + i];
        int lyrHashId = hashId_array[grp*nLayers + lyr];
        if (nHashMatches[i] >= nRequiredMatches) {

            // Loop through collection hits to find collection hit data
            int matchingColl = matchingCollections[i*nLayers + lyr];
            const unsigned char *pHitData = &hitData[hitDataEventIndices[eventId]];
            for (int coll = 0; coll < matchingColl; coll++) {
                pHitData += nHits[nHitsEventIndices[eventId] + coll];
            }

            // Loop as many times as necessary for all threads to cover all patterns
            int nPattInGrp = (hitArrayGroupIndices[grp + 1] - hitArrayGroupIndices[grp])/nLayers;
            nLoops = ((nPattInGrp*nLayers)/blockDim.x) + 1;
            extern __shared__ unsigned int nPattMatches[];

            for (int n = 0; n < nLoops; n++) {

                int pattNum = n*blockDim.x/nLayers + row;

                // Initialise nPattMatches to zero
                if (lyr == 0) {
                    nPattMatches[row] = 0;
                }
                __syncthreads();

                // Only continue if thread isn't overflowing the number of patterns in the group
                if ( pattNum < nPattInGrp) {

                    if (matchingColl != -1) {
    
                        // Get pattern hit data
                        unsigned char pattHit = hitArray[hitArrayGroupIndices[grp] + n*blockDim.x + threadIdx.x];
                        // Decode pattern hit data
                        unsigned char dontCareBitmask = pattHit & 3;
                        if (dontCareBitmask == 2) { dontCareBitmask = 3; } 
                        unsigned char pattHitPos = ((pattHit >> 2) & 31);

                        // Loop through hits
                        for (int hit = 0; hit < nHits[nHitsEventIndices[eventId] + matchingColl]; hit++) {
                            unsigned char eventHitPos = (*(pHitData + hit) & 127);
                            unsigned char eventIsPixel = ((*(pHitData + hit) >> 7) & 1);
                            // Check if pixel or strip
                            if (eventIsPixel) {
                                // Pixel - decode pixel column number
                                unsigned char eventPixCol = (eventHitPos & 3);
                                unsigned char pattPixCol = pattHitPos/nMaxRows;
                                if ( eventPixCol == pattPixCol ) {
                                    // If pixel columns match, decode pixel row, mask with don't care bits and check
                                    // for a match
                                    unsigned char eventPixRow = (((eventHitPos >> 2) & 31) | dontCareBitmask);
                                    unsigned char pattPixRow = (pattHitPos%nMaxRows | dontCareBitmask);
                                    if ( eventPixRow == pattPixRow ) {
                                        atomicAdd(&nPattMatches[row],1);
                                        break;
                                    }
                                }
                            } else {
                                // Strip - decode superstrip values, mask with pattern don't care bits and check
                                // for a match
                                unsigned char eventSuperstrip = (((eventHitPos >> 2) & 31) | dontCareBitmask);
                                unsigned char pattSuperstrip = (pattHitPos | dontCareBitmask);
                                if ( eventSuperstrip == pattSuperstrip ) {
                                    atomicAdd(&nPattMatches[row],1);
                                    break;
                                }
                            }
                        } // End loop over hits

                    } else if ( lyrHashId == -1) {
                        atomicAdd(&nPattMatches[row],1);
                    }

                    __syncthreads();
                    // Output matching pattern ids to array
                    if (lyr == 0) {
                        if (nPattMatches[row] >= nRequiredMatches) {
                            int i = atomicAdd(nMatches,1);
                            int pattId = ((hitArrayGroupIndices[grp] - hitArrayGroupIndices[0])/nLayers) + pattNum;
                            matchingPattIds[i] = pattId;
                        }
                    }

                } // End if ( pattNum < nPattInGrp)

            } // End nLoops

        } // End if (nHashMatches >= nRequiredMatches)
    } // End loop over groups

}

__global__ void matchByLayer(const int *hashId_array, const unsigned char *hitArray, const unsigned int *hitArrayGroupIndices, 
                             const int *hashId, const unsigned int *hashIdEventIndices, const unsigned int *nHits,
                             const unsigned int *nHitsEventIndices, const unsigned char *hitData, const unsigned int *hitDataEventIndices, 
                             int *matchingPattIds, int *nMatches, const int eventId) {
    const int nLayers = 8;
    const int nRequiredMatches = 7;
    const int nMaxRows = 11;
    int grp = blockIdx.x;

    __shared__ unsigned int nHashMatches;
    __shared__ unsigned int nWildcards;

    if (threadIdx.x == 0) {
        nHashMatches = 0;
        nWildcards = 0;
    }
    __syncthreads();

    int lyrHashId = hashId_array[grp*nLayers + threadIdx.x%nLayers];
    // Get first nLayers threads to check if any hashIds are wildcards
    if (threadIdx.x < nLayers) {
        if (lyrHashId == -1) {
            // Automatically match if layer is wildcard
            atomicAdd(&nHashMatches,1);
            atomicAdd(&nWildcards,1);
         }
    }

    // Get each thread to compare one hashId with one collection from
    // an event to check for a match
    int nColl = hashIdEventIndices[eventId+1] - hashIdEventIndices[eventId];
    int nLoops = ((nLayers*nColl)/blockDim.x) + 1;
    for (int n = 0; n < nLoops; n++) {
        int coll = (n*blockDim.x + threadIdx.x)/nLayers;
        if (coll < nColl) {
            if (lyrHashId != -1) {
                if (hashId[hashIdEventIndices[eventId] + coll] == lyrHashId) {
                    atomicAdd(&nHashMatches,1);
                }
            }
        }
    }
    __syncthreads();

    // If there are enough hashId matches, loop through patterns in group
    if (nHashMatches >= nRequiredMatches) {
        int nPattInGrp = (hitArrayGroupIndices[grp + 1] - hitArrayGroupIndices[grp])/nLayers;
        extern __shared__ unsigned int nPattMatches[];

        // Initialise nPattMatches to zero
        nLoops = nPattInGrp/blockDim.x + 1;
        for (int n= 0; n < nLoops; n++) {
            int pattNum = n*blockDim.x + threadIdx.x;
            if (pattNum < nPattInGrp) {
                nPattMatches[pattNum] = 0;
            }
        }
        __syncthreads();

        // Loop as many times as necessary for all threads to cover all patterns
        nLoops = ((nPattInGrp*nLayers)/blockDim.x) + 1;
        for (int n = 0; n < nLoops; n++) {
            int pattNum = (n*blockDim.x + threadIdx.x)%nPattInGrp;
            int lyr = (n*blockDim.x + threadIdx.x)/nPattInGrp;

            // Only continue if thread isn't overflowing the number of layers
            if ( lyr < nLayers) {
                lyrHashId = hashId_array[grp*nLayers + lyr];

                // Automatically match if wildcard layer
                if (lyrHashId == -1) {
                    atomicAdd(&nPattMatches[pattNum],1);
                } else {
                    // Get pattern hit data
                    unsigned char pattHit = hitArray[hitArrayGroupIndices[grp] + pattNum*nLayers + lyr];// n*blockDim.x + threadIdx.x];
                    // Decode pattern hit data
                    unsigned char dontCareBitmask = pattHit & 3;
                        if (dontCareBitmask == 2) { dontCareBitmask = 3; } 
                    unsigned char pattHitPos = ((pattHit >> 2) & 31);

                    // Loop through collections looking for hashId match
                    int nColl = hashIdEventIndices[eventId+1] - hashIdEventIndices[eventId];
                    const unsigned char *pHitData = &hitData[hitDataEventIndices[eventId]];
                    for (int coll = 0; coll < nColl; coll++) {
                        if (hashId[hashIdEventIndices[eventId] + coll] == lyrHashId) {
                            // Once the matching collection has been found, loop through hits
                            for (int hit = 0; hit < nHits[nHitsEventIndices[eventId] + coll]; hit++) {
                                unsigned char eventHitPos = (*(pHitData + hit) & 127);
                                unsigned char eventIsPixel = ((*(pHitData + hit) >> 7) & 1);
                                // Check if pixel or strip
                                if (eventIsPixel) {
                                    // Pixel - decode pixel column number
                                    unsigned char eventPixCol = (eventHitPos & 3);
                                    unsigned char pattPixCol = pattHitPos/nMaxRows;
                                    if ( eventPixCol == pattPixCol ) {
                                        // If pixel columns match, decode pixel row, mask with don't care bits and check
                                        // for a match
                                        unsigned char eventPixRow = (((eventHitPos >> 2) & 31) | dontCareBitmask);
                                        unsigned char pattPixRow = (pattHitPos%nMaxRows | dontCareBitmask);
                                        if ( eventPixRow == pattPixRow ) {
                                                atomicAdd(&nPattMatches[pattNum],1);
                                                break;
                                        }
                                    }
                                } else {
                                    // Strip - decode superstrip values, mask with pattern don't care bits and check
                                    // for a match
                                    unsigned char eventSuperstrip = (((eventHitPos >> 2) & 31) | dontCareBitmask);
                                    unsigned char pattSuperstrip = (pattHitPos | dontCareBitmask);
                                    if ( eventSuperstrip == pattSuperstrip ) {
                                        atomicAdd(&nPattMatches[pattNum],1);
                                        break;
                                    }
                                }
                            }
                            break; // Break once the matching collection has been checked
                        }
                        pHitData += nHits[nHitsEventIndices[eventId] + coll];
                    }
                }
            } // End if lyr < nLayers
        } // End loop over patterns

        __syncthreads();
        // Output matching pattern ids to array
        nLoops = nPattInGrp/blockDim.x + 1;
        for (int n = 0; n < nLoops; n++) {
            int pattNum = n*blockDim.x + threadIdx.x;
            if (pattNum < nPattInGrp) {
                if (nPattMatches[pattNum] >= nRequiredMatches) {
                    int i = atomicAdd(nMatches,1);
                    int pattId = ((hitArrayGroupIndices[grp] - hitArrayGroupIndices[0])/nLayers) + pattNum;
                    matchingPattIds[i] = pattId;
                }
            }
        }

    }
}

