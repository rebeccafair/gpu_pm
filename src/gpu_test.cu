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

void copyContextToGpu(const PatternContainer, const EventContainer, GpuContext);
void runTestKernel(const PatternContainer, const EventContainer, GpuContext);
void deleteGpuContext(GpuContext);
__global__ void testKernel(const int *hashId_array, const unsigned char *hitArray,
                           const unsigned int *hitArrayGroupIndices, const int *hashId,
                           const unsigned int *hashIdEventIndices, const unsigned int *nHits,
                           const unsigned int *nHitsEventIndices, const unsigned char *hitData,
                           const unsigned int *hitDataEventIndices, int *matchingPattIds,
                           int *nMatches, const int nGroups, const int nLayers, const int eventId);


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
    size_t matchingPattIds_size = sizeof(int)*1000;

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


void runTestKernel(const PatternContainer& p, const EventContainer& e, GpuContext& ctx) {
    cudaError_t err = cudaSuccess;

    // Calculate number of threads/blocks required
    //int N = e.header.nEvents;
    int threadsPerBlock =256;
    //int threadsPerBlock = p.header.nLayers;
    //int blocksPerGrid = (N/threadsPerBlock) + 1; 
    int blocksPerGrid = p.header.nGroups;

    // Allocate and initialise vector to store result
    size_t matchingPattIds_size = sizeof(int)*1000;
    vector<int> matchingPattIds(matchingPattIds_size);
    int nMatches;

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
    int nEvents = 1;
    int nPattMatchesSize = threadsPerBlock/p.header.nLayers;
    //for (int i = 0; i < e.header.nEvents; i++ ) {
    for (int i =0; i < nEvents; i++ ) {
        int j = 1;
        testKernel<<<blocksPerGrid, threadsPerBlock, nPattMatchesSize>>>(ctx.d_hashId_array, ctx.d_hitArray, ctx.d_hitArrayGroupIndices,
                                                       ctx.d_hashId, ctx.d_hashIdEventIndices, ctx.d_nHits,
                                                       ctx.d_nHitsEventIndices, ctx.d_hitData, ctx.d_hitDataEventIndices,
                                                       ctx.d_matchingPattIds, ctx.d_nMatches, p.header.nGroups, p.header.nLayers, j);
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
    cout << "Ran kernel " << nEvents << " times in " << msecTotal << " ms" << endl;
    float msecPerEvent = msecTotal/nEvents;
    cout << "Average kernel time is " << msecPerEvent << " ms" << endl;

    // Copy result back to host memory
    err = cudaMemcpy(&matchingPattIds[0], ctx.d_matchingPattIds, matchingPattIds_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) cerr << "Error: d_matchingPattIds not copied from device to host" << endl;
    err = cudaMemcpy(&nMatches, ctx.d_nMatches, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) cerr << "Error: d_nMatches not copied from device to host" << endl;

    for (int i = 0; i < nMatches; i++) {
        cout << "Matching id: " << matchingPattIds[i] << endl;
    }

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

__global__ void testKernel(const int *hashId_array, const unsigned char *hitArray, const unsigned int *hitArrayGroupIndices, 
                           const int *hashId, const unsigned int *hashIdEventIndices, const unsigned int *nHits,
                           const unsigned int *nHitsEventIndices, const unsigned char *hitData, const unsigned int *hitDataEventIndices, 
                           int *matchingPattIds, int *nMatches, const int nGroups, const int nLayers, const int eventId) {
    int nRequiredMatches = 7;
    int nMaxRows = 22;
    //int i = blockDim.x * blockIdx.x + threadIdx.x;
    int grp = blockIdx.x;
    int lyr = threadIdx.x%nLayers;
    int row = threadIdx.x/nLayers;
    //const unsigned char *pHitData = &hitData[hitDataEventIndices[eventId]];

    __shared__ unsigned int nHashMatches;
    __shared__ unsigned int nWildcards;

    __syncthreads();

    int lyrHashId = hashId_array[grp*nLayers + (threadIdx.x%nLayers)];
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
        //if(threadIdx.x == 0) { printf("Match! For group %i, nPatt: %i\n",grp,nPattInGrp); }
        int nLoops = ((nPattInGrp*nLayers)/blockDim.x) + 1;
        // Loop as many times as necessary for all threads to cover all patterns
        for (int i = 0; i < nLoops; i++) {
            extern __shared__ unsigned int nPattMatches[];

            // Only continue if all patterns haven't yet been covered
            if (i*blockDim.x + threadIdx.x < nPattInGrp) {
                // Initialise nPattMatches to zero
                if (threadIdx.x%nLayers == 0) {
                    nPattMatches[threadIdx.x/nLayers] = 0;
                }
                __syncthreads();
    
                // Automatically match if wildcard layer
                if (lyrHashId == -1) {
                    atomicAdd(&nPattMatches[threadIdx.x/nLayers],1);
                } else {
                    // Get pattern hit data
                    unsigned char pattHit = hitArray[hitArrayGroupIndices[grp] + i*blockDim.x + threadIdx.x];
                    // Decode pattern hit data
                    unsigned char pattDontCareBits = pattHit & 3;
                    unsigned char dontCareBitmask = (7 >> (3 - pattDontCareBits));
                    unsigned char pattHitPos = ((pattHit >> 2) & 63);

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
                                                atomicAdd(&nPattMatches[threadIdx.x/nLayers],1);
                                                //printf("Strip match found grp %i patt %i lyr %i, event %i coll %i, nPattMatches: %i\n", grp, (i*blockDim.x + threadIdx.x)/nLayers, threadIdx.x%nLayers, eventId, coll, nPattMatches[threadIdx.x/nLayers] );
                                                break;
                                        }
                                    }
                                } else {
                                    // Strip - decode superstrip values, mask with pattern don't care bits and check
                                    // for a match
                                    unsigned char eventSuperstrip = (((eventHitPos >> 2) & 31) | dontCareBitmask);
                                    unsigned char pattSuperstrip = (pattHitPos | dontCareBitmask);
                                    if ( eventSuperstrip == pattSuperstrip ) {
                                        atomicAdd(&nPattMatches[threadIdx.x/nLayers],1);
                                        //printf("Strip match found grp %i patt %i lyr %i, event %i coll %i, nPattMatches: %i\n", grp, (i*blockDim.x + threadIdx.x)/nLayers, threadIdx.x%nLayers, eventId, coll, nPattMatches[threadIdx.x/nLayers] );
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
                if (threadIdx.x%nLayers == 0) {
                    if (nPattMatches[threadIdx.x/nLayers] >= nRequiredMatches) {
                        printf("Match found for grp %i patt %i pattId %i, nMatches %i\n", grp, (i*blockDim.x + threadIdx.x)/nLayers, ((hitArrayGroupIndices[grp] - hitArrayGroupIndices[0])/nLayers) + (i*blockDim.x + threadIdx.x)/nLayers, nMatches);
                        int i = atomicAdd(nMatches,1);
                        matchingPattIds[i] = ((hitArrayGroupIndices[grp] - hitArrayGroupIndices[0])/nLayers) + (i*blockDim.x + threadIdx.x)/nLayers;
                    }
                }
            }
        
        } // End loop over patterns

    }
}
