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


void runMatchByBlock(const PatternContainer& p, const EventContainer& e, GpuContext& ctx, MatchResults& mr, int threadsPerBlock) {
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
        matchByBlock<<<blocksPerGrid, threadsPerBlock, nPattMatchesSize>>>(ctx.d_hashId_array, ctx.d_hitArray, ctx.d_hitArrayGroupIndices,
                                                                           ctx.d_hashId, ctx.d_hashIdEventIndices, ctx.d_nHits,
                                                                           ctx.d_nHitsEventIndices, ctx.d_hitData, ctx.d_hitDataEventIndices,
                                                                           ctx.d_matchingPattIds, ctx.d_nMatches, p.header.nGroups, p.header.nLayers, i);
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
    cout << "Average matchByBlock kernel time with " << threadsPerBlock << " threads is " << msecPerEvent << " ms" << endl;

    // Copy result back to host memory
    err = cudaMemcpy(&mr.nMatches, ctx.d_nMatches, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) cerr << "Error: d_nMatches not copied from device to host" << endl;
    mr.patternIds.resize(mr.nMatches);
    err = cudaMemcpy(&mr.patternIds[0], ctx.d_matchingPattIds, mr.nMatches*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) cerr << "Error: d_matchingPattIds not copied from device to host" << endl;

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
                                                                           ctx.d_matchingPattIds, ctx.d_nMatches, p.header.nGroups, p.header.nLayers, i);
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

__global__ void matchByBlock(const int *hashId_array, const unsigned char *hitArray, const unsigned int *hitArrayGroupIndices, 
                             const int *hashId, const unsigned int *hashIdEventIndices, const unsigned int *nHits,
                             const unsigned int *nHitsEventIndices, const unsigned char *hitData, const unsigned int *hitDataEventIndices, 
                             int *matchingPattIds, int *nMatches, const int nGroups, const int nLayers, const int eventId) {
    int nRequiredMatches = 7;
    int nMaxRows = 22;
    int grp = blockIdx.x;
    int lyr = threadIdx.x%nLayers;
    int row = threadIdx.x/nLayers;

    __shared__ unsigned int nHashMatches;
    __shared__ unsigned int nWildcards;

    int lyrHashId = hashId_array[grp*nLayers + lyr];
    // Get first nLayers threads to check the group hashIds and check if they are
    // a potential match for this event
    if (threadIdx.x < nLayers) {
        if (lyrHashId == -1) {
            // Automatically match if layer is wildcard
            atomicAdd(&nHashMatches,1);
            atomicAdd(&nWildcards,1);
        } else {
            // Otherwise loop through collections looking for match
            int nColl = hashIdEventIndices[eventId+1] - hashIdEventIndices[eventId];
            for (int coll = 0; coll < nColl; coll++) {
                if (hashId[hashIdEventIndices[eventId] + coll] == lyrHashId) {
                    atomicAdd(&nHashMatches,1);
                    // Break out of collection if a match is found
                    break;
                }
            }
        }

    }
    __syncthreads();

    // If there are enough hashId matches, loop through patterns in group
    if (nHashMatches >= nRequiredMatches) {
        int nPattInGrp = (hitArrayGroupIndices[grp + 1] - hitArrayGroupIndices[grp])/nLayers;
        int nLoops = ((nPattInGrp*nLayers)/blockDim.x) + 1;
        // Loop as many times as necessary for all threads to cover all patterns
        extern __shared__ unsigned int nPattMatches[];

        for (int n = 0; n < nLoops; n++) {
            int pattNum = n*blockDim.x/nLayers + row;

            // Only continue if thread isn't overflowing the number of patterns in the group
            if ( pattNum < nPattInGrp) {
                // Initialise nPattMatches to zero
                if (lyr == 0) {
                    nPattMatches[row] = 0;
                }
                __syncthreads();
    
                // Automatically match if wildcard layer
                if (lyrHashId == -1) {
                    atomicAdd(&nPattMatches[row],1);
                } else {
                    // Get pattern hit data
                    unsigned char pattHit = hitArray[hitArrayGroupIndices[grp] + n*blockDim.x + threadIdx.x];
                    // Decode pattern hit data
                    unsigned char dontCareBitmask = pattHit & 3;
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
                            }
                            break; // Break once the matching collection has been checked
                        }
                        pHitData += nHits[nHitsEventIndices[eventId] + coll];
                    }
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
            }
        } // End loop over patterns

    }
}

__global__ void matchByLayer(const int *hashId_array, const unsigned char *hitArray, const unsigned int *hitArrayGroupIndices, 
                             const int *hashId, const unsigned int *hashIdEventIndices, const unsigned int *nHits,
                             const unsigned int *nHitsEventIndices, const unsigned char *hitData, const unsigned int *hitDataEventIndices, 
                             int *matchingPattIds, int *nMatches, const int nGroups, const int nLayers, const int eventId) {
    int nRequiredMatches = 7;
    int nMaxRows = 22;
    int grp = blockIdx.x;

    __shared__ unsigned int nHashMatches;
    __shared__ unsigned int nWildcards;

    // Get first nLayers threads to check the group hashIds and check if they are
    // a potential match for this event
    if (threadIdx.x < nLayers) {
        int grpCheckHashId = hashId_array[grp*nLayers + threadIdx.x];
        if (grpCheckHashId == -1) {
            // Automatically match if layer is wildcard
            atomicAdd(&nHashMatches,1);
            atomicAdd(&nWildcards,1);
        } else {
            // Otherwise loop through collections looking for match
            int nColl = hashIdEventIndices[eventId+1] - hashIdEventIndices[eventId];
            for (int coll = 0; coll < nColl; coll++) {
                if (hashId[hashIdEventIndices[eventId] + coll] == grpCheckHashId) {
                    atomicAdd(&nHashMatches,1);
                    // Break out of collection if a match is found
                    break;
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
        int mLoops = nPattInGrp/blockDim.x + 1;
        for (int m = 0; m < mLoops; m++) {
            int pattNum = m*blockDim.x + threadIdx.x;
            if (pattNum < nPattInGrp) {
                nPattMatches[pattNum] = 0;
            }
        }
        __syncthreads();

        // Loop as many times as necessary for all threads to cover all patterns
        int nLoops = ((nPattInGrp*nLayers)/blockDim.x) + 1;
        for (int n = 0; n < nLoops; n++) {
            int pattNum = (n*blockDim.x + threadIdx.x)%nPattInGrp;
            int lyr = (n*blockDim.x + threadIdx.x)/nPattInGrp;

            // Only continue if thread isn't overflowing the number of layers
            if ( lyr < nLayers) {
                int lyrHashId = hashId_array[grp*nLayers + lyr];

                // Automatically match if wildcard layer
                if (lyrHashId == -1) {
                    atomicAdd(&nPattMatches[pattNum],1);
                } else {
                    // Get pattern hit data
                    unsigned char pattHit = hitArray[hitArrayGroupIndices[grp] + pattNum*nLayers + lyr];// n*blockDim.x + threadIdx.x];
                    // Decode pattern hit data
                    unsigned char dontCareBitmask = pattHit & 3;
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
        //int mLoops = nPattInGrp/blockDim.x + 1;
        // Output matching pattern ids to array
        for (int m = 0; m < mLoops; m++) {
            int pattNum = m*blockDim.x + threadIdx.x;
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

