#ifndef __GPU_TEST_H__
#define __GPU_TEST_H__

#include "eventReader.h"
#include "patternReader.h"

struct GpuContext {

    // Pattern vars
    int *d_nPattInGrp;
    int *d_hashId_array;
    unsigned char *d_hitArray;
    unsigned int *d_hitArrayGroupIndices;

    // Event vars
    unsigned int *d_nCollections;
    int *d_hashId;
    unsigned int *d_hashIdEventIndices;
    unsigned int *d_nHits;
    unsigned int *d_nHitsEventIndices;
    unsigned char *d_hitData;
    unsigned int *d_hitDataEventIndices;

    // Outs
    int *d_hashId_out;
};

void copyContextToGpu(const PatternContainer& p, const EventContainer& e, GpuContext& ctx);

__global__ void testKernel(const int *hashId, const unsigned int *hashIdEventBegin, int *hashId_out, int N);

#endif

