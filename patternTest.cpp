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
void readEvents(string eventFile);

struct PatternHeader {
    unsigned int nHitPatt;
    int nGroups;
    unsigned int nLayers;
};

struct EventHeader {
    int nEvents;
};


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
    readEvents(eventFile);

    return 0;
}


void readPatterns(string patternFile){
    ifstream input(patternFile.c_str(),ifstream::binary);
    ostream_iterator<int> int_out (cout, " ");
    PatternHeader header;

    if (input) {
        cout << "\nReading pattern file " << patternFile << endl;
 
        // Read header 
        input.read((char*)&header, sizeof(header));
        cout << "\nnHitPatt: " << header.nHitPatt;
        cout << " nGroups: " << header.nGroups;
        cout << " nLayers: " << header.nLayers << endl;

        // Declare vectors and temp variables
        // Group record
        vector<int> hashId_array;
        vector<int> temp_hashId_array(header.nLayers);
        vector<int> nPattInGrp;
        int temp_nPattInGrp;
        // Pattern record
        vector<unsigned short> layerSet;
        unsigned short temp_layerSet;
        vector<unsigned char> hitArray;
        vector<unsigned char> temp_hitArray(header.nLayers);

        int groupCount;
        int patternCount;

        // Loop through groups
        groupCount = 0;
        for (int i = 0; i < header.nGroups; i++) {
            groupCount++;
            cout << "\nReading group " << i + 1 << " of " << header.nGroups << endl;

            // Read and print hashId_array
            input.read((char*)&temp_hashId_array.front(), sizeof(int)*header.nLayers);
            cout << "hashId_array: ";
            copy(temp_hashId_array.begin(), temp_hashId_array.end(), int_out);
            cout << endl;
            hashId_array.insert(hashId_array.end(), temp_hashId_array.begin(), temp_hashId_array.end());

            // Read and print nPattInGrp
            input.read((char*)&temp_nPattInGrp, sizeof(temp_nPattInGrp));
            cout << "nPattInGrp: " << temp_nPattInGrp << endl;
            nPattInGrp.insert(nPattInGrp.end(), temp_nPattInGrp);

            //Loop through patterns
            patternCount = 0;
	    for (int j = 0; j < temp_nPattInGrp; j++) {
                patternCount++;
                cout << "Reading pattern " << j + 1 << " of " << temp_nPattInGrp << " (Group " << i + 1 << ")" << endl;
 
                // Read and print layerSet
                input.read((char*)&temp_layerSet, sizeof(temp_layerSet));
                cout << "layerSet: " << bitset<16>(temp_layerSet) << endl;
                layerSet.insert(layerSet.end(), temp_layerSet);
            
                // Read and print hitArray
                input.read((char*)&temp_hitArray.front(), header.nLayers);
                cout << "hitArray: ";
                //copy(temp_hitArray.begin(), temp_hitArray.end(), int_out);
                for (int k = 0; k < header.nLayers; k++) {
                    cout << bitset<8>(temp_hitArray[k]) << ' ';
                }
                cout << endl;
                hitArray.insert(hitArray.end(), temp_hitArray.begin(), temp_hitArray.end());
            }
            if (patternCount != temp_nPattInGrp) {
                cerr << "Error: Number of patterns read is incorrect for group " << i + 1 << endl;
                input.close();
                exit(EXIT_FAILURE);
            }
        }
        if (groupCount != header.nGroups) {
            cerr << "Error: Number of groups read is incorrect" << endl;
            input.close();
            exit(EXIT_FAILURE);
        }

        if (input.peek() == EOF) {
            cout << "\nFinished reading " << patternFile << endl;
            input.close();
        } else {
            cerr << "\nError: Finished reading groups but did not reach end of file" << endl;
            input.close();
            exit(EXIT_FAILURE);
        }
        input.close();

    } else {
        cerr << "Error reading pattern file " << patternFile << endl;
    }
}
void readEvents(string eventFile){
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
