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

void readPatterns(string patternFile, PatternContainer& p);
void printPatterns(const PatternContainer& p);

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
void readPatterns(string patternFile, PatternContainer& p){
    ifstream input(patternFile.c_str(),ifstream::binary);

    if (input) {
        auto t_begin = Clock::now();
        cout << "\nReading pattern file " << patternFile << endl;

        // Read header 
        input.read((char*)&p.header, sizeof(p.header));

        // Resize vectors according to header values
        p.nPattInGrp.resize(p.header.nGroups);
        p.hashId_array.resize(p.header.nGroups*p.header.nLayers);
        p.layerSet.resize(p.header.nHitPatt);
        p.layerSetGroupBegin.resize(p.header.nGroups);
        p.hitArray.resize(p.header.nHitPatt*p.header.nLayers);
        p.hitArrayGroupBegin.resize(p.header.nGroups);

        // Initialise pointers to beginning of layerSet and hitArray vectors
        unsigned short* currentLayerSet = &p.layerSet[0];
        unsigned char* currentHitArray = &p.hitArray[0];

        // Loop through groups
        for (int i = 0; i < p.header.nGroups; i++) {
            // Set pointers to the beginning of current group in layerSet and hitArray vectors
            p.layerSetGroupBegin[i] = currentLayerSet;
            p.hitArrayGroupBegin[i] = currentHitArray;

            // Read group fields
            input.read((char*)&p.hashId_array[i*p.header.nLayers], sizeof(int)*p.header.nLayers);
            input.read((char*)&p.nPattInGrp[i], sizeof(int));

            //Loop through patterns
	    for (int j = 0; j < p.nPattInGrp[i]; j++) {
 
                // Read pattern fields and advance pointers
                input.read((char*)currentLayerSet, sizeof(unsigned short));
                currentLayerSet++;
                input.read((char*)currentHitArray, p.header.nLayers);
                currentHitArray += p.header.nLayers;
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
        exit(EXIT_FAILURE);
    }
}

void printPatterns(const PatternContainer& p) {

    // Print header
    cout << "\nnHitPatt: " << p.header.nHitPatt;
    cout << " nGroups: " << p.header.nGroups;
    cout << " nLayers: " << p.header.nLayers << endl;

    // Loop through groups
    for (int grp = 0; grp < p.header.nGroups; grp++) {
        cout << "\nPrinting group " << grp + 1 << " of " << p.header.nGroups << endl;
        cout << "hashId_array: ";
        for (int i = 0; i < p.header.nLayers; i++) {
            cout << " " << p.hashId_array[grp*p.header.nLayers + i];
        }
        cout << endl;

        cout << "nPattInGrp: " << p.nPattInGrp[grp] << endl;

        // Loop through patterns
        for (int patt = 0; patt < p.nPattInGrp[grp]; patt++) {
            cout << "Printing pattern " << patt + 1 << " of " << p.nPattInGrp[grp] << "(Group " << grp + 1 << ")" << endl;
            cout << "layerSet: " << bitset<16>(*(p.layerSetGroupBegin[grp] + patt)) << endl;
            cout << "hitArray:";
            for (int i = 0; i < p.header.nLayers; i++) {
                cout << " " << bitset<8>(*(p.hitArrayGroupBegin[grp] + patt*p.header.nLayers + i));
            }
            cout << endl;
        }
    }

}

