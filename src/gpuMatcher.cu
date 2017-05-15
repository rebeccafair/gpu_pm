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

    // Get detector elements that are in patterns (used for creating bit arrays)
    vector <short> hashIdToIndex;
    int nDetectorElems = 0;
    int maxHashId = 50000;
    patternHashIdToIndex(p, maxHashId, hashIdToIndex, nDetectorElems);

    // Calculate size for all arrays that will be transferred
    size_t hashId_array_size = sizeof(int)*p.hashId_array.size();
    size_t hitArray_size = sizeof(unsigned char)*p.hitArray.size();
    size_t hitArrayGroupIndices_size = sizeof(unsigned int)*h_hitArrayGroupIndices.size();
    size_t hashIdToIndex_size = sizeof(short)*hashIdToIndex.size();
    size_t bitArray_size = sizeof(unsigned int)*3*nDetectorElems;
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
    err = cudaMalloc((void ** )&ctx.d_hashIdToIndex, hashIdToIndex_size);
    if (err != cudaSuccess) cerr << "Error: device memory not successfully allocated for d_hashIdToIndex\n" << cudaGetErrorString(err) << endl;
    err = cudaMalloc((void ** )&ctx.d_bitArray, bitArray_size);
    if (err != cudaSuccess) cerr << "Error: device memory not successfully allocated for d_bitArray\n" << cudaGetErrorString(err) << endl;
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
    err = cudaMemcpy(ctx.d_hashIdToIndex, &hashIdToIndex[0], hashIdToIndex_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) cerr << "Error: hashIdToIndex not copied to device\n" << cudaGetErrorString(err) << endl;

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


void runGpuMatching(const PatternContainer& p, const EventContainer& e, GpuContext& ctx, MatchResults& mr, int threadsPerBlock, int nBlocks) {
    cudaError_t err = cudaSuccess;

    int maxNGroupsInBlock = 0;
    if (nBlocks) {
        // Distribute groups to blocks according to number of blocks
        vector<int> blockBegin(nBlocks,-1);
        vector<int> nGroupsInBlock(nBlocks,0);
        vector<int> groups(p.header.nGroups,-1);
        distributeWork(nBlocks, p, blockBegin, nGroupsInBlock, groups, maxNGroupsInBlock);

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
    }

    // Get nDetectorElems
    vector <short> hashIdToIndex;
    int nDetectorElems = 0;
    int maxHashId = 50000;
    patternHashIdToIndex(p, maxHashId, hashIdToIndex, nDetectorElems);

    // Create events for timing bit array creation and kernel
    cudaEvent_t kernelStart, bitStart;
    err = cudaEventCreate(&kernelStart);
    err = cudaEventCreate(&bitStart);
    if (err != cudaSuccess) cerr << "Error: failed to create timer start events\n" << cudaGetErrorString(err) << endl;
    cudaEvent_t kernelStop, bitStop;
    err = cudaEventCreate(&kernelStop);
    err = cudaEventCreate(&bitStop);
    if (err != cudaSuccess) cerr << "Error: failed to create timer stop events\n" << cudaGetErrorString(err) << endl;

    // Total timers
    float kernelTotal = 0.0f;
    float bitArrTotal = 0.0f;

    // Calculate bit arrays and run kernel for each event
    if (nBlocks) {
        for (int i = 0; i < e.header.nEvents; i++ ) {

            // Calculate bit array and record timers
            err = cudaEventRecord(bitStart, 0);
            if (err != cudaSuccess) cerr << "Error: failed to record bit start event\n" << cudaGetErrorString(err) << endl;
            vector<unsigned int> bitArray = createBitArray(p, e, hashIdToIndex, nDetectorElems, i);
            err = cudaMemcpy(ctx.d_bitArray, &bitArray[0], sizeof(unsigned int)*bitArray.size(), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) cerr << "Error: bitArray not copied to device\n" << cudaGetErrorString(err) << endl;
            err = cudaEventRecord(bitStop, 0);
            if (err != cudaSuccess) cerr << "Error: failed to record bit stop event\n" << cudaGetErrorString(err) << endl;

            // Run kernel to calculate matches and record timers
            err = cudaEventRecord(kernelStart, 0);
            if (err != cudaSuccess) cerr << "Error: failed to record kernel start event\n" << cudaGetErrorString(err) << endl;
            matchByBlockMulti<<<nBlocks, threadsPerBlock>>>(ctx.d_hashId_array, ctx.d_hitArray, ctx.d_hitArrayGroupIndices,
                                                            ctx.d_bitArray, ctx.d_hashIdToIndex, nDetectorElems,
                                                            ctx.d_matchingPattIds, ctx.d_nMatches, i, ctx.d_blockBegin,
                                                            ctx.d_nGroupsInBlock, ctx.d_groups);
            err = cudaEventRecord(kernelStop, 0);
            if (err != cudaSuccess) cerr << "Error: failed to record kernel stop event\n" << cudaGetErrorString(err) << endl;
            err = cudaEventSynchronize(kernelStop);
            if (err != cudaSuccess) cerr << "Error: failed to synchronize on stop event\n" << cudaGetErrorString(err) << endl;

            // Calculate elapsed time
            float bitArrTime = 0.0f;
            float kernelTime = 0.0f;
            err = cudaEventElapsedTime(&bitArrTime, bitStart, bitStop);
            if (err != cudaSuccess) cerr << "Error: failed to get elapsed time between events\n" << cudaGetErrorString(err) << endl;
            bitArrTotal += bitArrTime;
            err = cudaEventElapsedTime(&kernelTime, kernelStart, kernelStop);
            if (err != cudaSuccess) cerr << "Error: failed to get elapsed time between events\n" << cudaGetErrorString(err) << endl;
            kernelTotal += kernelTime;

        }
    } else {
        for (int i = 0; i < e.header.nEvents; i++ ) {

            // Calculate bit array and record timers
            err = cudaEventRecord(bitStart, 0);
            if (err != cudaSuccess) cerr << "Error: failed to record bit start event\n" << cudaGetErrorString(err) << endl;
            vector<unsigned int> bitArray = createBitArray(p, e, hashIdToIndex, nDetectorElems, i);
            err = cudaMemcpy(ctx.d_bitArray, &bitArray[0], sizeof(unsigned int)*bitArray.size(), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) cerr << "Error: bitArray not copied to device\n" << cudaGetErrorString(err) << endl;
            err = cudaEventRecord(bitStop, 0);
            if (err != cudaSuccess) cerr << "Error: failed to record bit stop event\n" << cudaGetErrorString(err) << endl;

            // Run kernel to calculate matches and record timers
            err = cudaEventRecord(kernelStart, 0);
            if (err != cudaSuccess) cerr << "Error: failed to record kernel start event\n" << cudaGetErrorString(err) << endl;
            matchByBlockSingle<<<p.header.nGroups, threadsPerBlock>>>(ctx.d_hashId_array, ctx.d_hitArray, ctx.d_hitArrayGroupIndices,
                                                                                     ctx.d_bitArray, ctx.d_hashIdToIndex, nDetectorElems,
                                                                                     ctx.d_matchingPattIds, ctx.d_nMatches, i);
            err = cudaEventRecord(kernelStop, 0);
            if (err != cudaSuccess) cerr << "Error: failed to record kernel stop event\n" << cudaGetErrorString(err) << endl;
            err = cudaEventSynchronize(kernelStop);
            if (err != cudaSuccess) cerr << "Error: failed to synchronize on stop event\n" << cudaGetErrorString(err) << endl;

            // Calculate elapsed time
            float bitArrTime = 0.0f;
            float kernelTime = 0.0f;
            err = cudaEventElapsedTime(&bitArrTime, bitStart, bitStop);
            if (err != cudaSuccess) cerr << "Error: failed to get elapsed time between events\n" << cudaGetErrorString(err) << endl;
            bitArrTotal += bitArrTime;
            err = cudaEventElapsedTime(&kernelTime, kernelStart, kernelStop);
            if (err != cudaSuccess) cerr << "Error: failed to get elapsed time between events\n" << cudaGetErrorString(err) << endl;
            kernelTotal += kernelTime;
        }

    }
    cudaDeviceSynchronize();

    // Calculate and output time per event
    cout << "Ran kernel " << e.header.nEvents << " times in " << kernelTotal << " ms" << endl;
    cout << "Created bit arrays " << e.header.nEvents << " times in " << bitArrTotal << " ms" << endl;
    float msecPerEvent = kernelTotal/e.header.nEvents;
    if (nBlocks) {
        cout << "Average GPU matching kernel time with " << threadsPerBlock << " threads and " << nBlocks << " blocks is " << msecPerEvent << " ms" << endl;
        cout << "Average bit array creation time is " << bitArrTotal/e.header.nEvents << " ms" << endl;
    } else {
        cout << "Average GPU matching kernel time with " << threadsPerBlock << " threads is " << msecPerEvent << " ms" << endl;
        cout << "Average bit array creation time is " << bitArrTotal/e.header.nEvents << " ms" << endl;
    }

    // Copy result back to host memory
    err = cudaMemcpy(&mr.nMatches, ctx.d_nMatches, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) cerr << "Error: d_nMatches not copied from device to host" << endl;
    mr.patternIds.resize(mr.nMatches);
    err = cudaMemcpy(&mr.patternIds[0], ctx.d_matchingPattIds, mr.nMatches*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) cerr << "Error: d_matchingPattIds not copied from device to host" << endl;

};

void patternHashIdToIndex(const PatternContainer& p, const int maxHashId, vector<short>& hashIdToIndex, int& nDetectorElemsInPatt) {

    hashIdToIndex.assign(maxHashId,-1);
    nDetectorElemsInPatt = 0;

    for (int grp = 0; grp < p.header.nGroups; grp++) {
        for (int lyr = 0; lyr < p.header.nLayers; lyr++) {
            int hashId = p.hashId_array[grp*p.header.nLayers + lyr];
            if (hashId != -1 && hashIdToIndex[hashId] == -1) {
               hashIdToIndex[hashId] = nDetectorElemsInPatt;
               nDetectorElemsInPatt++;
            }
        }
    }
}

vector<unsigned int> createBitArray(const PatternContainer& p, const EventContainer& e, const vector<short>& hashIdToIndex, int nDetectorElemsInPatt, int eventId) {

    int nMaxRows = 11;
    int maxDcBits = 2;
    int columnOffset = 4; // When encoding pixel into bitArray, need to offset columns so that
                          // only hits in the same column will match.
    const unsigned char *pHitData = e.hitDataEventBegin[eventId];
    vector<unsigned int> bitArray(nDetectorElemsInPatt*(maxDcBits + 1),0);

    for (int coll = 0; coll < e.nCollections[eventId]; coll++) {
        int hashId = *(e.hashIdEventBegin[eventId] + coll);
        // Check if hashId for this collection is found in pattern set
        if (hashIdToIndex[hashId] != -1) {

            // Put hits into bit arrays
            unsigned char isPixel = ((*pHitData >> 7) & 1); // If bit 7 is 1, element is pixel, otherwise strip
            for (int hit = 0; hit < *(e.nHitsEventBegin[eventId] + coll); hit++) {
                unsigned char hitPos = ((*(pHitData + hit) >> 2) & 31); // Get superstrip value if strip, or row value if pixel. Occupies bits 2-6

                for (int dcBits = 0; dcBits <= maxDcBits; dcBits++) {
                    unsigned char maskedHitPos = hitPos &  (~((1 << dcBits) - 1)); // Mask hit value depending on dc bits. e.g. if dcBits = 2 AND with 11100
                    if (isPixel) {
                        maskedHitPos += (*(pHitData + hit) & 3)*(nMaxRows + columnOffset); // If pixel, need to adjust value according to column
                    }
                    unsigned char bitMask = 1; // Get bitmask  e.g. ...01111 for dcBits = 2
                    if (dcBits == 1) {
                        bitMask = 3;
                    } else if (dcBits == 2) {
                        bitMask = 15;
                    }
                    bitArray[dcBits*nDetectorElemsInPatt + hashIdToIndex[hashId]] |= (bitMask << maskedHitPos);
                }
            } // End loop over hits

        }
        pHitData += *(e.nHitsEventBegin[eventId] + coll); // Update pointer to hit data
    } // End loop over collections

    return bitArray;
}

void distributeWork(int nBlocks, const PatternContainer& p, vector<int>& blockBegin, vector<int>& nGroupsInBlock, vector<int>& groups, int& maxNGroupsInBlock) {

    int maxPattsInBlock = (p.header.nHitPatt + nBlocks - 1)/nBlocks;

    //cout << "Distributing work..." << endl;

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
        if (nGroupsInBlock[b] > maxNGroupsInBlock) { maxNGroupsInBlock = nGroupsInBlock[b]; }
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
    }*/

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
    err = cudaFree(ctx.d_bitArray);
    if (err != cudaSuccess) cerr << "Error: failed to free device memory for d_bitArray\n" << cudaGetErrorString(err) << endl;
    err = cudaFree(ctx.d_hashIdToIndex);
    if (err != cudaSuccess) cerr << "Error: failed to free device memory for d_hashIdToIndex\n" << cudaGetErrorString(err) << endl;
    err = cudaFree(ctx.d_matchingPattIds);
    if (err != cudaSuccess) cerr << "Error: failed to free device memory for d_matchingPattIds\n" << cudaGetErrorString(err) << endl;

    // Free memory associated with distributing groups to blocks, if allocated
    err = cudaFree(ctx.d_blockBegin);
    if (err != cudaSuccess && err != cudaErrorInvalidDevicePointer) cerr << "Error: failed to free device memory for d_blockBegin\n" << cudaGetErrorString(err) << endl;
    err = cudaFree(ctx.d_nGroupsInBlock);
    if (err != cudaSuccess && err != cudaErrorInvalidDevicePointer) cerr << "Error: failed to free device memory for d_nGroupsInBlock\n" << cudaGetErrorString(err) << endl;
    err = cudaFree(ctx.d_groups);
    if (err != cudaSuccess && err != cudaErrorInvalidDevicePointer) cerr << "Error: failed to free device memory for d_groups\n" << cudaGetErrorString(err) << endl;

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
                                   const unsigned int *bitArray, const short *hashIdToIndex, const int nDetectorElemsInPatt,
                                   int *matchingPattIds, int *nMatches, const int eventId) {
    const int nLayers = 8;
    const int nRequiredMatches = 7;
    const int nMaxRows = 11;
    int columnOffset = 4; // When encoding pixel into bitArray, need to offset columns so that
                          // only hits in the same column will match.

    int grp = blockIdx.x;
    int lyr = threadIdx.x%nLayers;

    __shared__ unsigned int nHashMatches; // Number of group hashIds that match event collection hashIds
    __shared__ int bitArrayIndex[nLayers]; // The index in the bit array corresponding to the hashId for a certain layer
    __shared__ int lyrHashIds[nLayers];

    if (threadIdx.x == 0) {
        nHashMatches = 0;
    }
    __syncthreads();

    // Get first nLayers threads to set bit array index, lyrHashId, and check if there
    // are enough matches in the group if bit array > 0 there are hits for that 
    // detector element for this event
    if (threadIdx.x < nLayers) {
        int lyrHashId = hashId_array[grp*nLayers + lyr];
        lyrHashIds[lyr] = lyrHashId;
        bitArrayIndex[lyr] = hashIdToIndex[lyrHashId];
        if (lyrHashId == -1 || bitArray[bitArrayIndex[lyr]] > 0) {
            atomicAdd(&nHashMatches,1);
         }
    }

    __syncthreads();


    if (nHashMatches >= nRequiredMatches) {

        // Put relevant bit array elements in shared memory to reduce memory latency
        __shared__ unsigned int sharedBitArray[nLayers*3];
        int row = threadIdx.x/nLayers;
        if (threadIdx.x < nLayers*3) {
            sharedBitArray[threadIdx.x] = bitArray[row*nDetectorElemsInPatt + bitArrayIndex[lyr]];
        }
        __syncthreads();

        // Loop as many times as necessary for all threads to cover all patterns
        int nPattInGrp = (hitArrayGroupIndices[grp + 1] - hitArrayGroupIndices[grp])/nLayers;
        for (int patt = threadIdx.x; patt < nPattInGrp; patt += blockDim.x) {
            int pattMatches = 0;

            for (int l = 0; l < nLayers; l++) {
                int lyrHashId = lyrHashIds[l];

                // Check for wildcard
                if (lyrHashId == -1) {
                    pattMatches++;
                // Else check if this event has a collection with the right hashId
                } else if (sharedBitArray[l] > 0) {
                    // Get pattern hit data
                    unsigned char pattHit = hitArray[hitArrayGroupIndices[grp] + patt*nLayers + l];
                    unsigned char isPixel = ((pattHit >> 7) & 1);
                    unsigned char hitPos = ((pattHit >> 2) & 31); // Get superstrip position if strip, or row if pixel. Occupies bits 2-6
                    unsigned char dcBits = (pattHit & 3);
                    if (dcBits == 3) { dcBits = 2; }
                    if (isPixel) {
                        unsigned char pattPixCol = hitPos/nMaxRows;
                        unsigned char pattPixRow = hitPos%nMaxRows;
                        hitPos = (nMaxRows + columnOffset)*pattPixCol + pattPixRow;
                    }
                    if ( ((1 << hitPos) & sharedBitArray[dcBits*nLayers + l]) > 0 ) {
                        pattMatches++;
                    }

                }
                if ( pattMatches + nLayers - l + 1 < nRequiredMatches ) {
                    break;
                }

            }
            // Output matching pattern ids to array
            if (pattMatches >= nRequiredMatches) {
                int i = atomicAdd(nMatches,1);
                int pattId = ((hitArrayGroupIndices[grp] - hitArrayGroupIndices[0])/nLayers) + patt;
                matchingPattIds[i] = pattId;
            }

        }
    }

}

__global__ void matchByBlockMulti(const int *hashId_array, const unsigned char *hitArray, const unsigned int *hitArrayGroupIndices, 
                                  const unsigned int *bitArray, const short *hashIdToIndex, const int nDetectorElemsInPatt,
                                  int *matchingPattIds, int *nMatches, const int eventId, const int *blockBegin, const int *nGroupsInBlock,
                                  const int *groups) {

    const int nLayers = 8;
    const int nRequiredMatches = 7;
    const int nMaxRows = 11;
    const int maxGroupsInBlock = 1000;
    int columnOffset = 4; // When encoding pixel into bitArray, need to offset columns so that
                          // only hits in the same column will match.
    int nGroups = nGroupsInBlock[blockIdx.x];

    __shared__ unsigned int nHashMatches[maxGroupsInBlock]; // Number of group hashIds that match event collection hashIds for each group
    __shared__ unsigned int sharedGroups[maxGroupsInBlock]; // Array of groups to be handled by this block

    // Initialise match counters
    if (threadIdx.x < nGroups) {
        nHashMatches[threadIdx.x] = 0;
        sharedGroups[nGroups + threadIdx.x] = groups[blockBegin[blockIdx.x] + threadIdx.x];
    }
    __syncthreads();

    // Loop over groups in block and check if there are enough matches in the group
    // if bit array > 0 there are hits for that detector element for this event
    int lyr = threadIdx.x%nLayers;
    int nLoops = (nGroups*nLayers/blockDim.x) + 1;
    for (int n = 0; n < nLoops; n++) {
        int grpInBlock = (n*blockDim.x + threadIdx.x)/nLayers;
        if (grpInBlock < nGroups) {
            int grp = sharedGroups[nGroups + grpInBlock];
            int lyrHashId = hashId_array[grp*nLayers + lyr];
            // Automatically match if layer is wildcard
            if (lyrHashId == -1 || bitArray[hashIdToIndex[lyrHashId]] > 0) {
                atomicAdd(&nHashMatches[grpInBlock],1);
             }
        }

    }

    __syncthreads();

    for (int i = 0; i < nGroups; i++) {
        if (nHashMatches[i] >= nRequiredMatches) {
            int grp = sharedGroups[nGroups + i];
            int row = threadIdx.x/nLayers;
            int lyrHashId = hashId_array[grp*nLayers + lyr];

            // Loop as many times as necessary for all threads to cover all patterns
            int nPattInGrp = (hitArrayGroupIndices[grp + 1] - hitArrayGroupIndices[grp])/nLayers;
            nLoops = ((nPattInGrp*nLayers)/blockDim.x) + 1;
            __shared__ unsigned int nPattMatches[128]; // Max size of patt matches = maxNThreads/nLayers = 1024/8

            for (int n = 0; n < nLoops; n++) {

                int pattNum = n*blockDim.x/nLayers + row;

                // Initialise nPattMatches to zero
                if (lyr == 0) {
                    nPattMatches[row] = 0;
                }
                __syncthreads();

                // Only continue if thread isn't overflowing the number of patterns in the group
                if ( pattNum < nPattInGrp) {

                    // Check for wildcard
                    if (lyrHashId == -1) {
                        atomicAdd(&nPattMatches[row],1);
                    // Else check if this event has a collection with the right hashId 
                    } else if (hashIdToIndex[lyrHashId] != -1) {

                        // Get pattern hit data
                        unsigned char pattHit = hitArray[hitArrayGroupIndices[grp] + n*blockDim.x + threadIdx.x];
                        unsigned char isPixel = ((pattHit >> 7) & 1);
                        unsigned char hitPos = ((pattHit >> 2) & 31); // Get superstrip position if strip, or row if pixel. Occupies bits 2-6
                        unsigned char dcBits = (pattHit & 3);
                        if (dcBits == 3) { dcBits = 2; }
                        if (isPixel) {
                            unsigned char pattPixCol = hitPos/nMaxRows;
                            unsigned char pattPixRow = hitPos%nMaxRows;
                            hitPos = (nMaxRows + columnOffset)*pattPixCol + pattPixRow;
                        }
                        if ( ((1 << hitPos) & bitArray[dcBits*nDetectorElemsInPatt + hashIdToIndex[lyrHashId]]) > 0 ) {
                            atomicAdd(&nPattMatches[row],1);
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

                } // End if ( pattNum < nPattInGrp)

            } // End nLoops

        } // End if (nHashMatches >= nRequiredMatches)
    } // End loop over groups*/

}
