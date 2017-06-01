#ifndef __GPU_TEST_H__
#define __GPU_TEST_H__

#include "eventReader.h"
#include "patternReader.h"
#include "matchResults.h"

// Contains vars to be copied to GPU
struct GpuContext {

    // Pattern vars
    int *d_nPattInGrp;
    int *d_hashId_array;
    unsigned char *d_hitArray;
    unsigned int *d_hitArrayGroupIndices;

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

// Allocates memory and copies arrays to GPU
void createGpuContext(const PatternContainer& p, const EventContainer& e, GpuContext& ctx);

// Calls GPU kernel and copies results from GPU
void runGpuMatching(const PatternContainer& p, const EventContainer& e, GpuContext& ctx, MatchResults& mr, int threadsPerBlock, int nBlocks = 0);

// For detector elements in the pattern set, translates detector element hash ID (0-~50000) to an index (0-~350). Required for bit array creation
void patternHashIdToIndex(const PatternContainer& p, const int maxHashId, vector<short>& hashIdToIndex, int& nDetectorElemsInPatt);

// Creates bit arrays
vector<unsigned int> createBitArray(const PatternContainer& p, const EventContainer& e, const vector<short>& hashIdToIndex, int nDetectorElemsInPatt, int eventId);

// Distributes pattern groups to blocks for use by matchByBlockMulti
void distributeWork(int nBlocks, const PatternContainer& p, vector<int>& blockBegin, vector<int>& nGroupsInBlock, vector<int>& groups, int& nMaxGroupsInBlock);

// Deallocates GPU memory
void deleteGpuContext(GpuContext& ctx);

// GPU kernel to match patterns to hits. Matches 1 pattern group per CUDA block
__global__ void matchByBlockSingle(const int *hashId_array, const unsigned char *hitArray,
                                   const unsigned int *hitArrayGroupIndices, const unsigned int *bitArray,
                                   const short *hashIdToArray, int nDetectorElemsInPatt, int *matchingPattIds,
                                   int *nMatches, const int eventId);

// GPU kernel to match patterns to hits. Matches multiple pattern groups per CUDA block, specified by distributeWork
__global__ void matchByBlockMulti(const int *hashId_array, const unsigned char *hitArray,
                                  const unsigned int *hitArrayGroupIndices, unsigned int *bitArray,
                                  const short *hashIdToIndex, const int nDetectorElemsInPatt, int *matchingPattIds,
                                  int *nMatches, const int eventId, const int *blockBegin, const int *nGroupsInBlock,
                                  const int *groups);

#endif

