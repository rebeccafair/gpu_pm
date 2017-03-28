#ifndef PATTERN_READER_H_
#define PATTERN_READER_H_

// patternReader.h
// Reads and prints patterns from a binary pattern file

#include <string>
#include <vector>

using namespace std;

struct PatternHeader {
    unsigned int nHitPatt;
    int nGroups;
    unsigned int nLayers;
};

struct PatternContainer {
    PatternHeader header;
    vector<int> nPattInGrp;
    vector<int> hashId_array;
    vector<unsigned short> layerSet;
    vector<unsigned short*> layerSetGroupBegin;
    vector<unsigned char> hitArray;
    vector<unsigned char*> hitArrayGroupBegin;
};

void readPatterns(string patternFile, PatternContainer& p);
void printPatterns(const PatternContainer& p);

#endif
