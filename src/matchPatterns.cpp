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

    int* pHashIdEventBegin;
    unsigned int* pNHitsEventBegin;
    unsigned char* hitDataCollBegin;

    int nRequiredMatches = 7;
    int nMaxRows = 22;
    vector<int> nEventMatches(eventHeader.nEvents);

    auto t_begin = Clock::now();

    // Loop through events
    for (int event = 0; event < eventHeader.nEvents; event++) {
        pHashIdEventBegin = (event == 0) ? &hashId[0] : pHashIdEventBegin + nCollections[event-1];
        pNHitsEventBegin = (event == 0) ? &nHits[0] : pNHitsEventBegin + nCollections[event-1];
        vector<int> matchingGroups;

        // Determine which groups contain the correct detector
        // elements to be a potential match
        for (int grp = 0; grp < patternHeader.nGroups; grp++) {
            int nMatchingDetectorElems = 0;
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
            vector<int> matchingPatterns;
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

                            // Loop through hits and create variables
                            vector<unsigned char> eventHitPos(*(pNHitsEventBegin + coll));
                            vector<unsigned char> eventIsPixel(*(pNHitsEventBegin + coll));
                            for (int hit = 0; hit <  *(pNHitsEventBegin + coll); hit++) {
                                eventHitPos[hit] = (*(hitDataCollBegin + hit) & 127);
                                eventIsPixel[hit] = ((*(hitDataCollBegin + hit) >> 7) & 1);
                            }

                            // Loop through potential matching patterns
                            int nMatchingPatterns = ( lyr == 0 ) ? nPattInGrp[grp] : matchingPatterns.size();
                            for (int j = 0; j < nMatchingPatterns; j++) {
                                int patt = ( lyr == 0 ) ? j : matchingPatterns[j];
                                // Decode pattern data
                                unsigned char pattDontCareBits = *(hitArrayGroupBegin[grp] + patternHeader.nLayers*patt + lyr) & 3;
                                unsigned char dontCareBitmask = (7 >> (3 - pattDontCareBits));
                                unsigned char pattHitPos = ((*(hitArrayGroupBegin[grp] + patternHeader.nLayers*patt + lyr) >> 2) & 63);

                                // Loop through hits in matching collections
                                for (int hit = 0; hit < *(pNHitsEventBegin + coll); hit++) {
                                    // Check if pixel or strip 
                                    if (eventIsPixel[hit]) {
                                        // Pixel - decode pixel column number
                                        unsigned char eventPixCol = (eventHitPos[hit] & 3);
                                        unsigned char pattPixCol = pattHitPos/nMaxRows;
                                        if ( eventPixCol == pattPixCol ) {
                                            // If pixel columns match, decode pixel row, mask with don't care bits and check
                                            // for a match
                                            unsigned char eventPixRow = (((eventHitPos[hit] >> 2) & 31) | dontCareBitmask);
                                            unsigned char pattPixRow = (pattHitPos%nMaxRows | dontCareBitmask);
                                            if ( eventPixRow == pattPixRow ) {
                                            //cout << "Match found for event " << event + 1 << " hashId " << *(pHashIdEventBegin + coll)
                                            //     << " grp " << grp + 1 << " patt " << patt + 1 << " pattHit: "
                                            //     << bitset<8>(*(hitArrayGroupBegin[grp] + patternHeader.nLayers*patt + lyr))
                                            //     << " collHit: " << bitset<8>(*(hitDataCollBegin + hit)) << endl;
                                                nPattMatches[patt]++;
                                                break;
                                            }
                                        }
                                    } else {
                                        // Strip - decode superstrip values, mask with pattern don't care bits and check
                                        // for a match
                                        unsigned char eventSuperstrip = (((eventHitPos[hit] >> 2) & 31) | dontCareBitmask);
                                        unsigned char pattSuperstrip = (pattHitPos | dontCareBitmask);
                                        if ( eventSuperstrip == pattSuperstrip ) {
                                            //cout << "Match found for event " << event + 1 << " hashId " << *(pHashIdEventBegin + coll)
                                            //     << " grp " << grp + 1 << " patt " << patt + 1 << " pattHit: "
                                            //     << bitset<8>(*(hitArrayGroupBegin[grp] + patternHeader.nLayers*patt + lyr))
                                            //     << " collHit: " << bitset<8>(*(hitDataCollBegin + hit)) << endl;
                                            nPattMatches[patt]++;
                                            break;
                                        }
                                    }
                                } // End loop through hits
                            } // End loop through patterns
                        }
                        hitDataCollBegin += *(pNHitsEventBegin + coll);
                    } // End loop through collections
                }
                // Create vector containing only patterns that still have a chance of matching
                vector<int> tmp_matchingPatterns;
                int nMatchingPatterns = ( lyr == 0 ) ? nPattInGrp[grp] : matchingPatterns.size();
                for (int j = 0; j < nMatchingPatterns; j++) {
                    int patt = ( lyr == 0 ) ? j : matchingPatterns[j];
                    if (nPattMatches[patt] + patternHeader.nLayers - lyr > nRequiredMatches) {
                        tmp_matchingPatterns.push_back(patt);
                    }
                }
                matchingPatterns = tmp_matchingPatterns;
            } // End loop through layers

            // Get total number of matches for this event
            for (int patt = 0; patt < nPattInGrp[grp]; patt++) {
                if (nPattMatches[patt] >= nRequiredMatches) {
                    //cout << "Match found, event: " << event + 1 << " grp: " << grp + 1 << " patt: " << patt + 1 << endl;
                    nEventMatches[event]++;
                }
            }
       } // End loop through matching groups
    } // End loop through events
    auto t_end = Clock::now();

    int totalMatches = 0;
    for (int event = 0; event < eventHeader.nEvents; event++) {
        totalMatches += nEventMatches[event];
        cout << "Matching patterns for event " << event + 1 << ": " << nEventMatches[event] << endl;
    }
    cout << "Total matches: " << totalMatches << endl;
    cout << "Matching completed in " << chrono::duration_cast<std::chrono::milliseconds>(t_end - t_begin).count() << " ms" << endl;
}

