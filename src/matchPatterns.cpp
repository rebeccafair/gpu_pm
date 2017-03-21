#include <iostream>
#include <vector>
#include <bitset>
#include <algorithm>

#include "common.h"
#include "eventReader.h"
#include "patternReader.h"

#include "matchPatterns.h"

using namespace std;

void match();

void match() {

    int* pHashId;
    int* pHashIdEventBegin;
    unsigned int* pNHitsEventBegin;
    unsigned char* hitDataCollBegin;

    int nMatchingDetectorElems;
    int nMatchingPatterns;
    int nRequiredMatches = 7;
    int nMaxRows = 22;
    vector<int> nEventMatches(eventHeader.nEvents);
    vector<int> matchingGroups;

    auto t_begin = Clock::now();

    // Loop through events
    for (int event = 0; event < eventHeader.nEvents; event++) {
        cout << "Comparing event " << event + 1 << endl;
        pHashIdEventBegin = (event == 0) ? &hashId[0] : pHashIdEventBegin + nCollections[event-1];
        pNHitsEventBegin = (event == 0) ? &nHits[0] : pNHitsEventBegin + nCollections[event-1];
        matchingGroups.clear();

        // Determine which groups contain the correct detector
        // elements to be a potential match
        for (int grp = 0; grp < patternHeader.nGroups; grp++) {
            nMatchingDetectorElems = 0;
            // Loop through hashId layers in group
            for (int lyr = 0; lyr < patternHeader.nLayers; lyr++) {
                // Check if layer is wildcard
                if ( hashId_array[grp*patternHeader.nLayers + lyr] == -1) {
                    nMatchingDetectorElems++;
                // If not wildcard, check collection hash IDs for a match
                } else {
                    for (int coll = 0; coll < nCollections[event]; coll++) {
                        if ( hashId_array[grp*patternHeader.nLayers + lyr] == *(pHashIdEventBegin + coll) ) {
                            nMatchingDetectorElems++;
                            // Break out of collection loop if match is found
                            break;
                        }
                    }
                }

                // Break out of group if a match is impossible
                if ( nMatchingDetectorElems <= lyr ) {
                    break;
                }
            }
            if ( nMatchingDetectorElems >= nRequiredMatches ) {
                 matchingGroups.push_back(grp);
            }
        }

        // For each matching group, loop through layers 
        for (int i = 0; i < matchingGroups.size(); i++) {
            vector<int> nPattMatches(nPattInGrp[i]);
            int grp = matchingGroups[i];
            for (int lyr = 0; lyr < patternHeader.nLayers; lyr++) {
                hitDataCollBegin = hitDataEventBegin[event];

                // Check for wildcard layer
                if ( hashId_array[grp*patternHeader.nLayers + lyr] == -1 ) {
                    transform(begin(nPattMatches), end(nPattMatches), begin(nPattMatches), [](int x){return x + 1;});

                // Otherwise find collection with matching hashId
                } else {
                    for (int coll = 0; coll < nCollections[event]; coll++) {
                        if (hashId_array[grp*patternHeader.nLayers + lyr] == *(pHashIdEventBegin + coll)) {
                            // Loop through hits in matching collections
                            for (int hit = 0; hit < *(pNHitsEventBegin + coll); hit++) {
                                // Decode hit data
                                unsigned char eventHitPos = (*(hitDataCollBegin + hit) & 127);
                                unsigned char eventIsPixel = ((*(hitDataCollBegin + hit) >> 7) & 1);
                                // Loop through patterns
                                for (int patt = 0; patt < nPattInGrp[grp]; patt++) {
                                    // Decode pattern data
                                    unsigned char pattDontCareBits = *(hitArrayGroupBegin[grp] + patternHeader.nLayers*patt + lyr) & 3;
                                    unsigned char pattHitPos = ((*(hitArrayGroupBegin[grp] + patternHeader.nLayers*patt + lyr) >> 2) & 63);

                                    // Check if pixel or strip 
                                    if (eventIsPixel) {
                                        // Pixel - decode pixel column number
                                        unsigned char eventPixCol = (eventHitPos & 3);
                                        unsigned char pattPixCol = pattHitPos/nMaxRows;
                                        if ( eventPixCol == pattPixCol ) {
                                            // If pixel columns match, decode pixel row and check whether they
                                            // are within dontCareBits of each other
                                            unsigned char eventPixRow = ((eventHitPos >> 2) & 31);
                                            unsigned char pattPixRow = pattHitPos%nMaxRows;
                                            if ( abs(eventPixRow - pattPixRow) <= pattDontCareBits ) {
                                                nPattMatches[patt]++;
                                            }
                                        }
                                    } else {
                                        // Strip - decode superstrip value
                                        unsigned char eventSuperstrip = ((eventHitPos >> 2) & 31);
                                        // Check if the hit positions of the pattern and event are within DontCareBits of each other
                                        if ( abs(eventSuperstrip - pattHitPos) <= pattDontCareBits ) {
                                            nPattMatches[patt]++;
                                        }
                                    }
                                }
                            }
                        }
                        hitDataCollBegin += *(pNHitsEventBegin + coll);
                    }
                }
            }
            for (int patt = 0; patt < nPattInGrp[grp]; patt++) {
                if (nPattMatches[patt] > nRequiredMatches) {
                    nEventMatches[event]++;  
                }
            }
       }
    }

    for (int event = 0; event < eventHeader.nEvents; event++) {
        cout << "Matching patterns for event " << event + 1 << ": " << nEventMatches[event] << endl;
    }
    auto t_end = Clock::now();
    cout << "Matching completed in " << chrono::duration_cast<std::chrono::milliseconds>(t_end - t_begin).count() << " ms" << endl;
}
