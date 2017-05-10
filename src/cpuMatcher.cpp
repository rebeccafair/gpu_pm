#include <iostream>
#include <vector>
#include <bitset>
#include <algorithm>
#include <map>

#include "common.h"
#include "eventReader.h"
#include "patternReader.h"
#include "matchResults.h"

#include "cpuMatcher.h"

using namespace std;

void matchByEvents(const PatternContainer& p, const EventContainer& e, MatchResults& mr) {

    unsigned char* hitDataCollBegin;

    int nRequiredMatches = 7;
    int nMaxRows = 11;
    int maxDcBits = 2;
    int columnOffset = 4; // When encoding pixel into bitArray, need to offset columns so that
                          // only hits in the same column will match.
    vector<int> nEventMatches(e.header.nEvents);

    auto t_begin = Clock::now();

    // Get list of detector elements that are in the pattern set
    map<int,int> hashIdToIndex;
    int nDetectorElemsInPatt = 0;

    for (int grp = 0; grp < p.header.nGroups; grp++) {
        for (int lyr = 0; lyr < p.header.nLayers; lyr++) {
            int hashId = p.hashId_array[grp*p.header.nLayers + lyr];
            if (hashId != -1 && hashIdToIndex.find(hashId) == hashIdToIndex.end()) {
               hashIdToIndex[hashId] = nDetectorElemsInPatt;
               nDetectorElemsInPatt++;
            }
        }
    }

    // Loop through events
    for (int event = 0; event < e.header.nEvents; event++) {
        vector<int> matchingGroups;

        // For each detector element in the pattern set, create a
        // bit array describing hits for this event
        const unsigned char *pHitData = e.hitDataEventBegin[event];
        vector<unsigned int> bitArray(nDetectorElemsInPatt*(maxDcBits + 1),0);

        for (int coll = 0; coll < e.nCollections[event]; coll++) {
            int hashId = *(e.hashIdEventBegin[event] + coll);
            // Check if hashId for this collection is found in pattern set
            if (hashIdToIndex.find(hashId) != hashIdToIndex.end()) {

                // Put hits into bit arrays
                unsigned char isPixel = ((*pHitData >> 7) & 1); // If bit 7 is 1, element is pixel, otherwise strip
                for (int hit = 0; hit < *(e.nHitsEventBegin[event] + coll); hit++) {
                    unsigned char hitPos = ((*(pHitData + hit) >> 2) & 31); // Get superstrip value if strip, or row value if pixel. Occupies bits 2-6

                    for (int dcBits = 0; dcBits <= maxDcBits; dcBits++) {
                        unsigned char maskedHitPos = hitPos &  (~((1 << dcBits) - 1)); // Mask hit value depending on dc bits. e.g. if dcBits = 2 AND with 11100
                        if (isPixel) {
                            maskedHitPos += (*(pHitData + hit) & 3)*(nMaxRows + columnOffset); // If pixel, need to adjust value according to column
                        }
                        unsigned char bitMask = 1; // Get bitmask  e.g. ...01111 for dcBits = 2
                        if (dcBits == 1) {
                            bitMask = 3;
                        } else if (dcBits == 2) {
                            bitMask = 15;
                        }
                        bitArray[dcBits*nDetectorElemsInPatt + hashIdToIndex[hashId]] |= (bitMask << maskedHitPos);
                    }
                } // End loop over hits
            }
            pHitData += *(e.nHitsEventBegin[event] + coll); // Update pointer to hit data
        } // End loop over collections

        // Determine which groups contain the correct detector
        // elements to be a potential match
        for (int grp = 0; grp < p.header.nGroups; grp++) {
            int nMatches = 0;
            // Loop through hashId layers in group
            for (int lyr = 0; lyr < p.header.nLayers; lyr++) {
                int hashId =  p.hashId_array[grp*p.header.nLayers + lyr];
                // Check if layer is wildcard
                if ( hashId == -1) {
                    nMatches++;
                // Otherwise check if this detector element has any hits in this event
                } else if (bitArray[hashIdToIndex[hashId]] > 0) {
                    nMatches++;
                }

                // Break out of group if a match is impossible
                if ( nMatches + p.header.nLayers - lyr + 1 < nRequiredMatches ) {
                    break;
                }
            }
            if ( nMatches >= nRequiredMatches ) {
                 matchingGroups.push_back(grp);
            }
        }

        /*cout << "Matching groups for event " << event << ": ";
        for (int i = 0; i < matchingGroups.size(); i++) {
            cout << matchingGroups[i] << " ";
        }
        cout << endl;*/

        // For each matching group, loop through layers 
        for (int i = 0; i < matchingGroups.size(); i++) {
            vector<int> matchingPatterns;
            vector<int> nMatches(p.nPattInGrp[i]);
            int grp = matchingGroups[i];
            for (int lyr = 0; lyr < p.header.nLayers; lyr++) {

                int hashId =  p.hashId_array[grp*p.header.nLayers + lyr];
                if (hashId == -1) {
                    transform(begin(nMatches), end(nMatches), begin(nMatches), [](int x){return x + 1;});
                } else if ( bitArray[hashIdToIndex[hashId]] > 0) {
                    // Loop through potential matching patterns
                    int nMatchingPatterns = ( lyr == 0 ) ? p.nPattInGrp[grp] : matchingPatterns.size();
                    for (int j = 0; j < nMatchingPatterns; j++) {
                        int patt = ( lyr == 0 ) ? j : matchingPatterns[j];
                        // Decode pattern data
                        unsigned char pattHit = *(p.hitArrayGroupBegin[grp] + p.header.nLayers*patt + lyr);
                        unsigned char isPixel = ((pattHit >> 7) & 1);
                        unsigned char hitPos = ((pattHit >> 2) & 31); // Get superstrip position if strip, or row if pixel. Occupies bits 2-6
                        unsigned char dcBits = (pattHit & 3);
                        if (dcBits == 3) { dcBits = 2; }
                        if (isPixel) {
                            unsigned char pattPixCol = hitPos/nMaxRows;
                            unsigned char pattPixRow = hitPos%nMaxRows;
                            hitPos = (nMaxRows + columnOffset)*pattPixCol + pattPixRow;
                        }
                        if ( ((1 << hitPos) & bitArray[dcBits*nDetectorElemsInPatt + hashIdToIndex[hashId]]) > 0 ) {
                            nMatches[patt]++;
                        }

                   }

                }

                // Create vector containing only patterns that still have a chance of matching
                vector<int> tmp_matchingPatterns;
                int nMatchingPatterns = ( lyr == 0 ) ? p.nPattInGrp[grp] : matchingPatterns.size();
                for (int j = 0; j < nMatchingPatterns; j++) {
                    int patt = ( lyr == 0 ) ? j : matchingPatterns[j];
                    if (nMatches[patt] + p.header.nLayers - lyr > nRequiredMatches) {
                        tmp_matchingPatterns.push_back(patt);
                    }
                }
                matchingPatterns = tmp_matchingPatterns;

            } // End loop over layers

            // Get total number of matches for this event
            for (int patt = 0; patt < p.nPattInGrp[grp]; patt++) {
                if (nMatches[patt] >= nRequiredMatches) {
                    //cout << "Match found, event: " << event << " grp: " << grp << " patt: " << patt << endl;
                    mr.patternIds.push_back(((p.hitArrayGroupBegin[grp] - p.hitArrayGroupBegin[0])/p.header.nLayers) + patt);
                    nEventMatches[event]++;
                }
            }

        } // End loop over matching groups
    } // End loop through events
    auto t_end = Clock::now();

    mr.nMatches = 0;
    for (int event = 0; event < e.header.nEvents; event++) {
        /*cout << "Matching patterns for event " << event << ": " << nEventMatches[event] << endl;
        if (nEventMatches[event] > 0) {
            cout << "Matching pattern ids:";
            for (int patt = 0; patt < nEventMatches[event]; patt++) {
                cout << " " << mr.patternIds[mr.nMatches + patt];
            }
            cout << endl;
        }*/
        mr.nMatches += nEventMatches[event];
    }
    cout << "Total matches: " << mr.nMatches << endl;
    cout << "CPU Matching completed in " << chrono::duration_cast<std::chrono::milliseconds>(t_end - t_begin).count() << " ms" << endl;
}

void matchByPatterns(const PatternContainer& p, const EventContainer& e, MatchResults& mr) {

    unsigned char* hitDataCollBegin;

    int nRequiredMatches = 7;
    int nMaxRows = 11;
    vector<int> nEventMatches(e.header.nEvents);
    //vector<int> matchingPattIds;

    auto t_begin = Clock::now();

    // Loop through groups
    for (int grp = 0; grp < p.header.nGroups; grp++) {
        vector<int> matchingEvents;
        // Determine which events contain collections with the correct
        // hashIds to be a potential match
        for (int event = 0; event < e.header.nEvents; event++) {
            int nMatches = 0;
            // Loop through layers
            for (int lyr = 0; lyr < p.header.nLayers; lyr ++) {
                // Check for wildcard layer
                if ( p.hashId_array[grp*p.header.nLayers + lyr] == -1) {
                    nMatches++;
                // If not wildcard, check collection hashIds for a match
                } else {
                    for (int coll = 0; coll < e.nCollections[event]; coll++) {
                        if ( p.hashId_array[grp*p.header.nLayers + lyr] == *(e.hashIdEventBegin[event] + coll) ) {
                            nMatches++;
                            // Break out of collection loop if match is found
                            break;
                        }
                     }
                }
                // Break out of event if a match is impossible
                if ( nMatches + p.header.nLayers - lyr + 1 < nRequiredMatches ) {
                    break;
                }
            }
            if ( nMatches >= nRequiredMatches) {
                matchingEvents.push_back(event);
            }
        } // End loop through events

        //cout << "Matching events for group " << grp << ": ";
        //for (int i = 0; i < matchingEvents.size(); i++) {
        //    cout << matchingEvents[i] << " ";
        //}
        //cout << endl;

        // Loop through potential matching events
        for (int i = 0; i < matchingEvents.size(); i++) {
            int event = matchingEvents[i];
            vector<int> nMatches(p.nPattInGrp[grp]);
            vector<int> matchingPatterns;

            // Loop through layers
            for (int lyr = 0; lyr < p.header.nLayers; lyr++) {
                hitDataCollBegin = e.hitDataEventBegin[event];
                // Check for wildcard layer
                if ( p.hashId_array[grp*p.header.nLayers + lyr] == -1 ) {
                    transform(begin(nMatches), end(nMatches), begin(nMatches), [](int x){return x + 1;});
                } else {
                    // Otherwise loop through collections searching for hashId match
                    for (int coll = 0; coll < e.nCollections[event]; coll++) {
                        if (p.hashId_array[grp*p.header.nLayers + lyr] == *(e.hashIdEventBegin[event] + coll) ) {
                            // Loop through hits and decode hit data
                            vector<unsigned char> eventHitPos(*(e.nHitsEventBegin[event] + coll));
                            vector<unsigned char> eventIsPixel(*(e.nHitsEventBegin[event] + coll));
                            for (int hit = 0; hit <  *(e.nHitsEventBegin[event] + coll); hit++) {
                                eventHitPos[hit] = (*(hitDataCollBegin + hit) & 127);
                                eventIsPixel[hit] = ((*(hitDataCollBegin + hit) >> 7) & 1);
                            }
                            // Loop through potential matching patterns
                            int nMatchingPatterns = ( lyr == 0 ) ? p.nPattInGrp[grp] : matchingPatterns.size();
                            for (int j = 0; j < nMatchingPatterns; j++) {
                                int patt = ( lyr == 0 ) ? j : matchingPatterns[j];
                                // Decode pattern data
                                unsigned char dontCareBitmask = *(p.hitArrayGroupBegin[grp] + p.header.nLayers*patt + lyr) & 3;
                                if (dontCareBitmask == 2) { dontCareBitmask = 3; }
                                unsigned char pattHitPos = ((*(p.hitArrayGroupBegin[grp] + p.header.nLayers*patt + lyr) >> 2) & 31);

                                // Loop through hits in matching collections
                                for (int hit = 0; hit < *(e.nHitsEventBegin[event] + coll); hit++) {
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
                                            //cout << "Match found for event " << event << " grp " << grp <<
                                            //        " hashId " << *(e.hashIdEventBegin[event] + coll) << " grp " << grp << " patt " << patt << " pattHit: "
                                            //     << bitset<8>(*(p.hitArrayGroupBegin[grp] + p.header.nLayers*patt + lyr))
                                            //     << " collHit: " << bitset<8>(*(hitDataCollBegin + hit)) << " nMatches: " << nMatches[patt] << endl;
                                                nMatches[patt]++;
                                                break;
                                            }
                                        }
                                    } else {
                                        // Strip - decode superstrip values, mask with pattern don't care bits and check
                                        // for a match
                                        unsigned char eventSuperstrip = (((eventHitPos[hit] >> 2) & 31) | dontCareBitmask);
                                        unsigned char pattSuperstrip = (pattHitPos | dontCareBitmask);
                                        if ( eventSuperstrip == pattSuperstrip ) {
                                            //cout << "Match found for event " << event << " grp " << grp <<
                                            //        " hashId " << *(e.hashIdEventBegin[event] + coll) << " grp " << grp << " patt " << patt << " pattHit: "
                                            //     << bitset<8>(*(p.hitArrayGroupBegin[grp] + p.header.nLayers*patt + lyr))
                                            //     << " collHit: " << bitset<8>(*(hitDataCollBegin + hit)) << " nMatches: " << nMatches[patt] << endl;
                                            nMatches[patt]++;
                                            break;
                                        }
                                    }
                                } // End loop through hits
                            } // End loop through patterns

                        }
                        hitDataCollBegin += *(e.nHitsEventBegin[event] + coll);
                    } // End loop through collections
                }
                // Create vector containing only patterns that still have a chance of matching
                vector<int> tmp_matchingPatterns;
                int nMatchingPatterns = ( lyr == 0 ) ? p.nPattInGrp[grp] : matchingPatterns.size();
                for (int j = 0; j < nMatchingPatterns; j++) {
                    int patt = ( lyr == 0 ) ? j : matchingPatterns[j];
                    if (nMatches[patt] + p.header.nLayers - lyr > nRequiredMatches) {
                        tmp_matchingPatterns.push_back(patt);
                    }
                }
                matchingPatterns = tmp_matchingPatterns;
            } // End loop through layers
            // Get total number of matches for this event
            for (int patt = 0; patt < p.nPattInGrp[grp]; patt++) {
                if (nMatches[patt] >= nRequiredMatches) {
                    //cout << "Match found, event: " << event << " grp: " << grp << " patt: " << patt << " pattId: " << ((p.hitArrayGroupBegin[grp] - p.hitArrayGroupBegin[0])/p.header.nLayers) + patt << endl;
                    mr.patternIds.push_back(((p.hitArrayGroupBegin[grp] - p.hitArrayGroupBegin[0])/p.header.nLayers) + patt);
                    nEventMatches[event]++;
                }
            }
        } // End loop through events


    } // End loop through groups
    auto t_end = Clock::now();

    mr.nMatches = 0;
    for (int event = 0; event < e.header.nEvents; event++) {
        /*cout << "Matching patterns for event " << event << ": " << nEventMatches[event] << endl;
        if (nEventMatches[event] > 0) {
            //cout << "Matching pattern ids:";
            for (int patt = 0; patt < nEventMatches[event]; patt++) {
                cout << " " << mr.patternIds[mr.nMatches + patt];
            }
            cout << endl;
        }*/
        mr.nMatches += nEventMatches[event];
    }
    cout << "Total matches: " << mr.nMatches << endl;
    cout << "CPU Matching completed in " << chrono::duration_cast<std::chrono::milliseconds>(t_end - t_begin).count() << " ms" << endl;
}
