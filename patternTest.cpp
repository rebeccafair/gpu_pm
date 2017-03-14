#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <bitset>

using namespace std;

void readPatterns(string patternFile);
void printPatterns();
void readEvents(string eventFile);
void printEvents();

struct PatternHeader {
    unsigned int nHitPatt;
    int nGroups;
    unsigned int nLayers;
};

struct EventHeader {
    int nEvents;
};

PatternHeader patternHeader;
vector<int> hashId_array;
vector<unsigned short> layerSet;
vector<unsigned short*> layerSetGroupBegin;
vector<unsigned char> hitArray;
vector<unsigned char*> hitArrayGroupBegin;

EventHeader eventHeader;
vector<int> eventId;
vector<int> hashId;
vector<unsigned int> nCollections;
vector<unsigned int> nHits;
vector<unsigned char> hitData;
vector<unsigned int> subEventId;
vector<unsigned int> barCode;

int main(int argc, char* argv[]) {

    string patternFile = "pattern_groups.bin";
    string eventFile = "single_track_hit_events.bin";
    char opt;

    while((opt = getopt(argc,argv,"e:p:")) != EOF) {
        switch(opt)
        {
            case 'e': eventFile = optarg; break;
            case 'p': patternFile = optarg; break;
            default: cerr << "Invalid argument" << endl; exit(EXIT_FAILURE);
        }
    }

    readPatterns(patternFile);
    printPatterns();
    readEvents(eventFile);
    printEvents();

    return 0;
}

void readPatterns(string patternFile){
    ifstream input(patternFile.c_str(),ifstream::binary);
    int nPattInGrp;

    if (input) {
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
        if (input.peek() == EOF) {
            cout << "Finished reading " << patternFile << endl;
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

void readEvents(string eventFile) {
    ifstream input(eventFile.c_str(),ifstream::binary);

    if (input) {
        cout << "Reading event file " << eventFile << endl;
 
        // Read header 
        input.read((char*)&eventHeader, sizeof(eventHeader));

        // Resize vectors according to header value
        eventId.resize(eventHeader.nEvents);
        nCollections.resize(eventHeader.nEvents);

        // Loop through events
        for (int i = 0; i < eventHeader.nEvents; i++) {

            // Read event fields
            input.read((char*)&eventId[i], sizeof(int));
            input.read((char*)&nCollections[i], sizeof(unsigned int));

            // Reserve memory for collection fields
            int* temp_hashId = new int[nCollections[i]];
            unsigned int* temp_nHits = new unsigned int[nCollections[i]];

            // Loop through collections
	    for (int j = 0; j < nCollections[i]; j++) {

                // Read collection fields
                input.read((char*)&temp_hashId[j], sizeof(int));
                input.read((char*)&temp_nHits[j], sizeof(unsigned int));

                // Reserve memory for hit fields
                unsigned char* temp_hitData = new unsigned char[temp_nHits[j]];
                unsigned int* temp_subEventId = new unsigned int[temp_nHits[j]];
                unsigned int* temp_barCode = new unsigned int[temp_nHits[j]];

                // Loop through hits
                for (int k = 0; k < temp_nHits[j]; k++) {

                    // Read hit fields
                    input.read((char*)&temp_hitData[k], sizeof(unsigned char));
                    input.read((char*)&temp_subEventId[k], sizeof(unsigned int));
                    input.read((char*)&temp_barCode[k], sizeof(unsigned int));
                }
                // Insert hit fields to global arrays
                hitData.insert(hitData.end(), temp_hitData, temp_hitData + temp_nHits[j]);
                subEventId.insert(subEventId.end(), temp_subEventId, temp_subEventId + temp_nHits[j]);
                barCode.insert(barCode.end(), temp_barCode, temp_barCode + temp_nHits[j]);
            }
            // Insert collection fields to global arrays
            hashId.insert(hashId.end(), temp_hashId, temp_hashId + nCollections[i]);
            nHits.insert(nHits.end(), temp_nHits, temp_nHits + nCollections[i]);
        }
        if (input.peek() == EOF) {
            cout << "Finished reading " << eventFile << endl;
            input.close();
        } else {
            cerr << "Error: Finished reading events but did not reach end of file" << endl;
            input.close();
            exit(EXIT_FAILURE);
        }
        input.close();

    } else {
        cerr << "Error reading event file " << eventFile << endl;
    }
}

void printEvents() {

    int* pHashId = &hashId[0];
    unsigned int* pNHits = &nHits[0];
    unsigned char* pHitData = &hitData[0];
    unsigned int* pSubEventId = &subEventId[0];
    unsigned int* pBarCode = &barCode[0];

    cout << "\nPrinting events..." << endl;

    // Print header
    cout << "\nnEvents: " << eventHeader.nEvents;

    // Loop through events
    for (int i = 0; i < eventHeader.nEvents; i++) {
        cout << "\nPrinting event " << i + 1 << " of " << eventHeader.nEvents << endl;
        cout << "EventId: " << eventId[i] << endl;
        cout << "nCollections: " << nCollections[i] << endl;

        // Loop through collections
        for (int j = 0; j < nCollections[i]; j++) {
            cout << "\nPrinting collection " << j + 1 << " of " << nCollections[i] << " (Event " << i + 1 << ")" << endl;

            cout << "hashId: " << *(pHashId) << endl;
            cout << "nHits: " << *(pNHits) << endl;

            // Loop through hits
            for (int k = 0; k < *(pNHits); k++) {
                cout << "hitData: " << bitset<8>(*(pHitData)) << endl;
                cout << "subEventId: " << *(pSubEventId) << endl;
                cout << "barCode: " << *(pBarCode) << endl;

                pHitData++;
                pSubEventId++;
                pBarCode++;
            }
            pHashId++;
            pNHits++;

        }
    }

}

