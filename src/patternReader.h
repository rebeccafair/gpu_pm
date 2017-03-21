#ifndef PATTERN_READER_H_
#define PATTERN_READER_H_

// patternReader.h
// Reads and prints patterns from a binary pattern file

#include <string>
#include <vector>

using namespace std;

void readPatterns(string patternFile);
void printPatterns();

struct PatternHeader {
    unsigned int nHitPatt;
    int nGroups;
    unsigned int nLayers;
};

extern PatternHeader patternHeader;
extern vector<int> nPattInGrp;
extern vector<int> hashId_array;
extern vector<unsigned short> layerSet;
extern vector<unsigned short*> layerSetGroupBegin;
extern vector<unsigned char> hitArray;
extern vector<unsigned char*> hitArrayGroupBegin;

#endif
