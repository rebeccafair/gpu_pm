#include <iostream>
#include <vector>

#include "common.h"
#include "eventReader.h"
#include "patternReader.h"

#include "matchPatterns.h"

using namespace std;

void match();

void match() {

    int* pHashId;
    int* pHashIdEventBegin;
    unsigned int* pNHits = &nHits[0];
    unsigned char* pHitData = &hitData[0];

    int nMatchingDetectorElems;
    int nRequiredMatches = 7;
    vector<int> matchingGroups;

    auto t_begin = Clock::now();

    // Loop through events
    for (int event = 0; event < eventHeader.nEvents; event++) {
        //cout << "Comparing event " << event + 1 << endl;

        pHashIdEventBegin = (event == 0) ? &hashId[0] : pHashIdEventBegin + nCollections[event-1];
        matchingGroups.clear();

        // Determine which groups contain the correct detector
        // elements to be a potential match
        for (int grp = 0; grp < patternHeader.nGroups; grp++) {
            nMatchingDetectorElems = 0;
            // Loop through hashId layers in group
            for (int lyr = 0; lyr < patternHeader.nLayers; lyr++) {
                //cout << "Checking layer " << lyr << " from group " << grp + 1 << " hashId: " << hashId_array[grp*patternHeader.nLayers + lyr] << endl;

                // Check if layer is wildcard
                if ( hashId_array[grp*patternHeader.nLayers + lyr] == -1) {
                    //cout << "Wildcard on layer " << lyr << " from group " << grp + 1 << endl;
                    nMatchingDetectorElems++;
                // If not wildcard, check collection hash IDs for a match
                } else {
                    for (int j = 0; j < nCollections[event]; j++) {
                        //cout << "Checking hashId: " << *(pHashIdEventBegin + j) << " from collection " << j + 1 << endl;
                        if ( hashId_array[grp*patternHeader.nLayers + lyr] == *(pHashIdEventBegin + j) ) {
                            nMatchingDetectorElems++;
                            //cout << "hashId match! Breaking out of collection" << endl;
                            break;
                        }
                    }
                }

                // Break out of group if a match is impossible
                if ( nMatchingDetectorElems + patternHeader.nLayers - lyr <= nRequiredMatches ) {
                    //cout << "Impossible to match, breaking out of group" << endl;
                    break;
                // Break out if enough matching ids have already been found
                } else if ( nMatchingDetectorElems >= nRequiredMatches ) {
                    //cout << "Match found! Breaking out of loop" << endl;
                }

            }
            //cout << "For group " << grp + 1 << ", " << nMatchingDetectorElems << " matching detector elements" << endl;
            if ( nMatchingDetectorElems >= nRequiredMatches ) {
                 //cout << "Found matching group! Group " << grp + 1 << endl;
                 matchingGroups.push_back(grp);
            }
        }
        cout << "Matching groups for event " << event + 1 << ": ";
        for (int i = 0; i < matchingGroups.size(); i++) {
            cout << matchingGroups[i] << " ";
        }
        cout << "\n" << endl;
    }
    auto t_end = Clock::now();
    cout << "Matching completed in " << chrono::duration_cast<std::chrono::milliseconds>(t_end - t_begin).count() << " ms" << endl;
}
