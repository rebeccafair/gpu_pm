#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <string>
#include <bitset>
#include <unistd.h>

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

vector<int> hashId_array;
vector<unsigned short> layerSet;
vector<unsigned short*> layerSetGroupBegin;
vector<unsigned char> hitArray;
vector<unsigned char*> hitArrayGroupBegin;
PatternHeader patternHeader;

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

    return 0;
}

void readPatterns(string patternFile){
    ifstream input(patternFile.c_str(),ifstream::binary);
    ostream_iterator<int> int_out (cout, " ");
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
            cout << "Reading pattern " << p + 1 << " of " << nPattInGrp << "(Group " << g + 1 << ")" << endl;
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
    ostream_iterator<int> int_out (cout, " ");
    EventHeader header;

    if (input) {
        cout << "\nReading event file " << eventFile << endl;
 
        // Read header 
        input.read((char*)&header, sizeof(header));
        cout << "nEvents: " << header.nEvents << endl;

        // Declare vectors and temp variables
        // Event record
        vector<int> eventId;
        int temp_eventId;
        vector<unsigned int> nCollections;
        unsigned int temp_nCollections;
        // Hit collection record
        vector<int> hashId;
        int temp_hashId;
        vector<unsigned int> nHits;
        unsigned int temp_nHits;
        // Hit record
        vector<unsigned char> hitData;
        unsigned char temp_hitData;
        vector<unsigned int> subEventId;
        unsigned int temp_subEventId;
        vector<unsigned int> barCode;
        unsigned int temp_barCode;

        int eventCount;
        int collectionCount;
        int hitCount;

        // Loop through events
        eventCount = 0;
        for (int i = 0; i < header.nEvents; i++) {
            eventCount++;
            cout << "\nReading event " << i + 1 << " of " << header.nEvents << endl;

            // Read and print eventId
            input.read((char*)&temp_eventId, sizeof(temp_eventId));
            cout << "eventId: " << temp_eventId << endl;
            eventId.insert(eventId.end(), temp_eventId);
            // Read and print nCollections
            input.read((char*)&temp_nCollections, sizeof(temp_nCollections));
            cout << "nCollections: " << temp_nCollections << endl;
            nCollections.insert(nCollections.end(), temp_nCollections);

            // Loop through collections
            collectionCount = 0;
	    for (int j = 0; j < temp_nCollections; j++) {
                collectionCount++;
                cout << "\nReading collection " << j + 1 << " of " << temp_nCollections << " (Event " << i + 1 << ")" << endl;
 
                // Read and print hashId
                input.read((char*)&temp_hashId, sizeof(temp_hashId));
                cout << "hashId: " << temp_hashId << endl;
                hashId.insert(hashId.end(), temp_hashId);
                // Read and print nHits
                input.read((char*)&temp_nHits, sizeof(temp_nHits));
                cout << "nHits: " << temp_nHits << endl;
                nHits.insert(nHits.end(), temp_nHits);

                // Loop through hits
                hitCount = 0;
                for (int k = 0; k < temp_nHits; k++) {
                    hitCount++;

                    // Read and print hitData
                    input.read((char*)&temp_hitData, sizeof(temp_hitData));
                    cout << "hitData: " << bitset<8>(temp_hitData) << endl;
                    hitData.insert(hitData.end(), temp_hitData);
                    // Read and print subEventId
                    input.read((char*)&temp_subEventId, sizeof(temp_subEventId));
                    cout << "subEventId: " << temp_subEventId << endl;
                    subEventId.insert(subEventId.end(), temp_subEventId);
                    // Read and print barCode
                    input.read((char*)&temp_barCode, sizeof(temp_barCode));
                    cout << "barCode: " << bitset<16>(temp_barCode) << endl;
                    barCode.insert(barCode.end(), temp_barCode);
                }
                if (hitCount != temp_nHits) {
                    cerr << "Error: Number of hits read is incorrect for event " << i + 1 << " collection " << j + 1 << endl;
                    input.close();
                    exit(EXIT_FAILURE);
                }
            }
            if (collectionCount != temp_nCollections) {
                cerr << "Error: Number of collections read is incorrect for event " << i + 1 << endl;
                input.close();
                exit(EXIT_FAILURE);
            }
        }
        if (eventCount != header.nEvents) {
            cerr << "Error: Number of events read is incorrect" << endl;
            input.close();
            exit(EXIT_FAILURE);
        }

        if (input.peek() == EOF) {
            cout << "\nFinished reading " << eventFile << endl;
            input.close();
        } else {
            cerr << "\nError: Finished reading events but did not reach end of file" << endl;
            input.close();
            exit(EXIT_FAILURE);
        }
        input.close();

    } else {
        cerr << "Error reading event file " << eventFile << endl;
    }
}
