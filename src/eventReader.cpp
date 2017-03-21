#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <bitset>
#include <cstdlib>
#include <chrono>

#include "common.h"
#include "eventReader.h"

using namespace std;

void readEvents(string eventFile);
void printEvents();

EventHeader eventHeader;
vector<int> eventId;
vector<int> hashId;
vector<unsigned int> nCollections;
vector<unsigned int> nHits;
vector<unsigned char> hitData;
vector<unsigned char*> hitDataEventBegin;
vector<unsigned int> subEventId;
vector<unsigned int> barCode;

// Read events from a binary event file and put them into global variables
// 
// The event file is structured:
// Header -> Event record -> Hit collection record -> Hit record
//
// Header - contains general information about the file 
//     Fields:
//         nEvents - number of events in the file
//
// Event record - contains collections of hits for a particular event
//     Fields:
//         eventId - unique ID for this event
//         nCollections - number of hit collections in this event
//
// Hit collection record - contains hits for a certain detector element
//     Fields:
//         hashId - unique ID describing the detector element
//         nHits - number of hits on this element
//
// Hit record - contains information about a hit
//     Fields:
//         hitData - a bitmask detailing hit position
//         subEventId - unique ID describing the sub event
//         barCode - unique ID describing the sub event particle
void readEvents(string eventFile) {
    ifstream input(eventFile.c_str(),ifstream::binary);

    if (input) {
        auto t_begin = Clock::now();
        cout << "Reading event file " << eventFile << endl;
 
        // Read header 
        input.read((char*)&eventHeader, sizeof(eventHeader));

        // Resize vectors according to header value
        eventId.resize(eventHeader.nEvents);
        nCollections.resize(eventHeader.nEvents);
        hitDataEventBegin.resize(eventHeader.nEvents);

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

        // Loop through groups, collections and events again to point pointers to the
        // beginning of each event in hitData
        unsigned int* pNHits = &nHits[0];
        unsigned char* pHitData = &hitData[0];
        for (int i = 0; i < eventHeader.nEvents; i++) {
            hitDataEventBegin[i] = pHitData;
            for (int j = 0; j < nCollections[i]; j++) {
                for (int k = 0; k < *pNHits; k++) {
                    pHitData++;
                }
                pNHits++;
            }
        }

        auto t_end = Clock::now();
        if (input.peek() == EOF) {
            cout << "Finished reading " << eventFile << " in "
                 <<  chrono::duration_cast<std::chrono::milliseconds>(t_end - t_begin).count() << " ms" << endl;
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

