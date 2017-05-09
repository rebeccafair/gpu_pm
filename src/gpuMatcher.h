#ifndef __GPU_TEST_H__
#define __GPU_TEST_H__

#include "eventReader.h"
#include "patternReader.h"
#include "matchResults.h"

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
    // Bit array inputs
    short *d_hashIdToIndex;
    unsigned int *d_bitArray;

    // Work distribution vars
    int *d_blockBegin;
    int *d_nGroupsInBlock;
    int *d_groups;

    // Outs
    int *d_nMatches;
    int *d_matchingPattIds;
};

void createGpuContext(const PatternContainer& p, const EventContainer& e, GpuContext& ctx);

void runMatchByBlockSingle(const PatternContainer& p, const EventContainer& e, GpuContext& ctx, MatchResults& mr, int threadsPerBlock);

void runMatchByBlockMulti(const PatternContainer& p, const EventContainer& e, GpuContext& ctx, MatchResults& mr, int threadsPerBlock, int nBlocks);

void patternHashIdToIndex(const PatternContainer& p, const int maxHashId, vector<short>& hashIdToIndex, int& nDetectorElemsInPatt);

vector<unsigned int> createBitArray(const PatternContainer& p, const EventContainer& e, const vector<short>& hashIdToIndex, int nDetectorElemsInPatt, int eventId);

void distributeWork(int nBlocks, const PatternContainer& p, vector<int>& blockBegin, vector<int>& nGroupsInBlock, vector<int>& groups);

void runMatchByLayer(const PatternContainer& p, const EventContainer& e, GpuContext& ctx, MatchResults& mr, int threadsPerBlock);

void deleteGpuContext(GpuContext& ctx);

__global__ void matchByBlockSingle(const int *hashId_array, const unsigned char *hitArray,
                                   const unsigned int *hitArrayGroupIndices, const int *hashId,
                                   const unsigned int *hashIdEventIndices, const unsigned int *nHits,
                                   const unsigned int *nHitsEventIndices, const unsigned char *hitData,
                                   const unsigned int *hitDataEventIndices, const unsigned int *bitArray,
                                   const short *hashIdToArray, int nDetectorElemsInPatt, int *matchingPattIds,
                                   int *nMatches, const int eventId);

__global__ void matchByBlockMulti(const int *hashId_array, const unsigned char *hitArray,
                                  const unsigned int *hitArrayGroupIndices, const int *hashId,
                                  const unsigned int *hashIdEventIndices, const unsigned int *nHits,
                                  const unsigned int *nHitsEventIndices, const unsigned char *hitData,
                                  const unsigned int *hitDataEventIndices, int *matchingPattIds,
                                  int *nMatches, const int eventId, const int *blockBegin, const int *nGroupsInBlock,
                                  const int *groups);

__global__ void matchByLayer(const int *hashId_array, const unsigned char *hitArray,
                             const unsigned int *hitArrayGroupIndices, const int *hashId,
                             const unsigned int *hashIdEventIndices, const unsigned int *nHits,
                             const unsigned int *nHitsEventIndices, const unsigned char *hitData,
                             const unsigned int *hitDataEventIndices, int *matchingPattIds,
                             int *nMatches, const int eventId);

#endif

