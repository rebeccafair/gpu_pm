#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <bitset>
#include <cstdlib>
#include <chrono>

#include "common.h"
#include "patternReader.h"

using namespace std;

void readPatterns(string patternFile);
void printPatterns();

PatternHeader patternHeader;
vector<int> hashId_array;
vector<unsigned short> layerSet;
vector<unsigned short*> layerSetGroupBegin;
vector<unsigned char> hitArray;
vector<unsigned char*> hitArrayGroupBegin;

// Read patterns from a binary pattern file and put them into global variables
// 
// The pattern file is structured:
// Header -> Group record -> Pattern record
//
// Header - contains general information about the file 
//     Fields:
//         nHitPatt - number of patterns in the file
//         nGroups - number of groups in the file
//         nLayers - number of detector layers
//
// Group record - contains patterns for a certain set of detector elements
//     Fields:
//         hashIdArray[nLayers] - an array of unique IDs describing the detector elements
//         nPattInGrp - number of patterns in this group
//
// Pattern record - describes a pattern of hits
//     Fields:
//         layerSet - a bitmask detailing the "wildcard" layers
//         hitArray[nLayers] - an array detailing the hit position on each element
void readPatterns(string patternFile){
    ifstream input(patternFile.c_str(),ifstream::binary);
    int nPattInGrp;

    if (input) {
        auto t_begin = Clock::now();
        cout << "\nReading pattern file " << patternFile << endl;

        // Read header 
        input.read((char*)&patternHeader, sizeof(patternHeader));

        // Resize vectors according to header values
        hashId_array.resize(patternHeader.nGroups*patternHeader.nLayers);
        layerSet.resize(patternHeader.nHitPatt);
        layerSetGroupBegin.resize(patternHeader.nGroups);
        hitArray.resize(patternHeader.nHitPatt*patternHeader.nLayers);
        hitArrayGroupBegin.resize(patternHeader.nGroups);

        // Initialise pointers to beginning of layerSet and hitArray vectors
        unsigned short* currentLayerSet = &layerSet[0];
        unsigned char* currentHitArray = &hitArray[0];

        // Loop through groups
        for (int i = 0; i < patternHeader.nGroups; i++) {
            // Set pointers to the beginning of current group in layerSet and hitArray vectors
            layerSetGroupBegin[i] = currentLayerSet;
            hitArrayGroupBegin[i] = currentHitArray;

            // Read group fields
            input.read((char*)&hashId_array[i*patternHeader.nLayers], sizeof(int)*patternHeader.nLayers);
            input.read((char*)&nPattInGrp, sizeof(nPattInGrp));

            //Loop through patterns
	    for (int j = 0; j < nPattInGrp; j++) {
 
                // Read pattern fields and advance pointers
                input.read((char*)currentLayerSet, sizeof(unsigned short));
                currentLayerSet++;
                input.read((char*)currentHitArray, patternHeader.nLayers);
                currentHitArray += patternHeader.nLayers;
            }
        }
        auto t_end = Clock::now();
        if (input.peek() == EOF) {
            cout << "Finished reading " << patternFile << " in " 
                 << chrono::duration_cast<std::chrono::milliseconds>(t_end - t_begin).count() << " ms" << endl;
            input.close();
        } else {
            cerr << "Error: Finished reading groups but did not reach end of file" << endl;
            input.close();
            exit(EXIT_FAILURE);
        }
        input.close();

    } else {
        cerr << "Error reading pattern file " << patternFile << endl;
    }
}

void printPatterns() {

    int nPattInGrp;
    unsigned short* layerSetGroupEnd;

    // Print header
    cout << "\nnHitPatt: " << patternHeader.nHitPatt;
    cout << " nGroups: " << patternHeader.nGroups;
    cout << " nLayers: " << patternHeader.nLayers << endl;

    // Loop through groups
    for (int g = 0; g < patternHeader.nGroups; g++) {
        cout << "\nPrinting group " << g + 1 << " of " << patternHeader.nGroups << endl;
        cout << "hashId_array: ";
        for (int i = 0; i < patternHeader.nLayers; i++) {
            cout << " " << hashId_array[g*patternHeader.nLayers + i];
        }
        cout << endl;

        layerSetGroupEnd = (g + 1 < patternHeader.nGroups) ? layerSetGroupBegin[g+1] : &layerSet.back() + 1;
        nPattInGrp = layerSetGroupEnd - layerSetGroupBegin[g];
        cout << "nPattInGrp: " << nPattInGrp << endl;

        // Loop through patterns
        for (int p = 0; p < nPattInGrp; p++) {
            cout << "Printing pattern " << p + 1 << " of " << nPattInGrp << "(Group " << g + 1 << ")" << endl;
            cout << "layerSet: " << bitset<16>(*(layerSetGroupBegin[g] + p)) << endl;
            cout << "hitArray:";
            for (int i = 0; i < patternHeader.nLayers; i++) {
                cout << " " << bitset<8>(*(hitArrayGroupBegin[g] + p*patternHeader.nLayers + i));
            }
            cout << endl;
        }
    }

}

