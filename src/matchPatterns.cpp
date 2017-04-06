#include <iostream>
#include <vector>
#include <bitset>
#include <algorithm>

#include "common.h"
#include "eventReader.h"
#include "patternReader.h"

#include "matchPatterns.h"

using namespace std;

void matchByEvents(const PatternContainer& p, const EventContainer& e);
void matchByPatterns(const PatternContainer& p, const EventContainer& e);

void matchByEvents(const PatternContainer& p, const EventContainer& e) {

    unsigned char* hitDataCollBegin;

    int nRequiredMatches = 7;
    int nMaxRows = 22;
    vector<int> nEventMatches(e.header.nEvents);
    vector<int> matchingPattIds;

    auto t_begin = Clock::now();

    // Loop through events
    for (int event = 0; event < e.header.nEvents; event++) {
        vector<int> matchingGroups;

        // Determine which groups contain the correct detector
        // elements to be a potential match
        for (int grp = 0; grp < p.header.nGroups; grp++) {
            int nMatches = 0;
            // Loop through hashId layers in group
            for (int lyr = 0; lyr < p.header.nLayers; lyr++) {
                // Check if layer is wildcard
                if ( p.hashId_array[grp*p.header.nLayers + lyr] == -1) {
                    nMatches++;
                // If not wildcard, check collection hash IDs for a match
                } else {
                    for (int coll = 0; coll < e.nCollections[event]; coll++) {
                        if ( p.hashId_array[grp*p.header.nLayers + lyr] == *(e.hashIdEventBegin[event] + coll) ) {
                            nMatches++;
                            // Break out of collection loop if match is found
                            break;
                        }
                    }
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

        cout << "Matching groups for event " << event + 1 << ": ";
        for (int i = 0; i < matchingGroups.size(); i++) {
            cout << matchingGroups[i] + 1 << " ";
        }
        cout << endl;

        // For each matching group, loop through layers 
        for (int i = 0; i < matchingGroups.size(); i++) {
            vector<int> matchingPatterns;
            vector<int> nMatches(p.nPattInGrp[i]);
            int grp = matchingGroups[i];
            for (int lyr = 0; lyr < p.header.nLayers; lyr++) {
                hitDataCollBegin = e.hitDataEventBegin[event];

                // Check for wildcard layer
                if ( p.hashId_array[grp*p.header.nLayers + lyr] == -1 ) {
                    transform(begin(nMatches), end(nMatches), begin(nMatches), [](int x){return x + 1;});

                // Otherwise find collection with matching hashId
                } else {
                    for (int coll = 0; coll < e.nCollections[event]; coll++) {
                        if (p.hashId_array[grp*p.header.nLayers + lyr] == *(e.hashIdEventBegin[event] + coll)) {

                            // Loop through hits and create variables
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
                                unsigned char pattDontCareBits = *(p.hitArrayGroupBegin[grp] + p.header.nLayers*patt + lyr) & 3;
                                unsigned char dontCareBitmask = (7 >> (3 - pattDontCareBits));
                                unsigned char pattHitPos = ((*(p.hitArrayGroupBegin[grp] + p.header.nLayers*patt + lyr) >> 2) & 63);

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
                                                //cout << "Match found for event " << event + 1 << " grp " << grp + 1 <<
                                                //" hashId " << *(e.hashIdEventBegin[event] + coll) << " grp " << grp + 1 << " patt " << patt + 1 << " pattHit: "
                                                //<< bitset<8>(*(p.hitArrayGroupBegin[grp] + p.header.nLayers*patt + lyr))
                                                //<< " collHit: " << bitset<8>(*(hitDataCollBegin + hit)) << " nMatches: " << nMatches[patt] << endl;
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
                                            //cout << "Match found for event " << event + 1 << " grp " << grp + 1 <<
                                            //" hashId " << *(e.hashIdEventBegin[event] + coll) << " grp " << grp + 1 << " patt " << patt + 1 << " pattHit: "
                                            //<< bitset<8>(*(p.hitArrayGroupBegin[grp] + p.header.nLayers*patt + lyr))
                                            //<< " collHit: " << bitset<8>(*(hitDataCollBegin + hit)) << " nMatches: " << nMatches[patt] << endl;
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
                    //cout << "Match found, event: " << event + 1 << " grp: " << grp + 1 << " patt: " << patt + 1 << endl;
                    matchingPattIds.push_back(((p.hitArrayGroupBegin[grp] - p.hitArrayGroupBegin[0])/p.header.nLayers) + patt);
                    nEventMatches[event]++;
                }
            }
       } // End loop through matching groups
    } // End loop through events
    auto t_end = Clock::now();

    int totalMatches = 0;
    for (int event = 0; event < e.header.nEvents; event++) {
        cout << "Matching patterns for event " << event + 1 << ": " << nEventMatches[event] << endl;
        if (nEventMatches[event] > 0) {
            cout << "Matching pattern ids:";
            for (int patt = 0; patt < nEventMatches[event]; patt++) {
                cout << " " << matchingPattIds[totalMatches + patt];
            }
            cout << endl;
        }
        totalMatches += nEventMatches[event];
    }
    cout << "Total matches: " << totalMatches << endl;
    cout << "Matching completed in " << chrono::duration_cast<std::chrono::milliseconds>(t_end - t_begin).count() << " ms" << endl;
}

void matchByPatterns(const PatternContainer& p, const EventContainer& e) {

    unsigned char* hitDataCollBegin;

    int nRequiredMatches = 7;
    int nMaxRows = 22;
    vector<int> nEventMatches(e.header.nEvents);
    vector<int> matchingPattIds;

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

        //cout << "Matching events for group " << grp + 1 << ": ";
        //for (int i = 0; i < matchingEvents.size(); i++) {
        //    cout << matchingEvents[i] + 1<< " ";
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
                                unsigned char pattDontCareBits = *(p.hitArrayGroupBegin[grp] + p.header.nLayers*patt + lyr) & 3;
                                unsigned char dontCareBitmask = (7 >> (3 - pattDontCareBits));
                                unsigned char pattHitPos = ((*(p.hitArrayGroupBegin[grp] + p.header.nLayers*patt + lyr) >> 2) & 63);

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
                                            //cout << "Match found for event " << event + 1 << " grp " << grp + 1 <<
                                            //        " hashId " << *(e.hashIdEventBegin[event] + coll) << " grp " << grp + 1 << " patt " << patt + 1 << " pattHit: "
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
                                            //cout << "Match found for event " << event + 1 << " grp " << grp + 1 <<
                                            //        " hashId " << *(e.hashIdEventBegin[event] + coll) << " grp " << grp + 1 << " patt " << patt + 1 << " pattHit: "
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
                    //cout << "Match found, event: " << event + 1 << " grp: " << grp + 1 << " patt: " << patt + 1 << " pattId: " << ((p.hitArrayGroupBegin[grp] - p.hitArrayGroupBegin[0])/p.header.nLayers) + patt << endl;
                    matchingPattIds.push_back(((p.hitArrayGroupBegin[grp] - p.hitArrayGroupBegin[0])/p.header.nLayers) + patt);
                    nEventMatches[event]++;
                }
            }
        } // End loop through events


    } // End loop through groups
    auto t_end = Clock::now();

    int totalMatches = 0;
    for (int event = 0; event < e.header.nEvents; event++) {
        cout << "Matching patterns for event " << event + 1 << ": " << nEventMatches[event] << endl;
        if (nEventMatches[event] > 0) {
            cout << "Matching pattern ids:";
            for (int patt = 0; patt < nEventMatches[event]; patt++) {
                cout << " " << matchingPattIds[totalMatches + patt];
            }
            cout << endl;
        }
        totalMatches += nEventMatches[event];
    }
    cout << "Total matches: " << totalMatches << endl;
    cout << "Matching completed in " << chrono::duration_cast<std::chrono::milliseconds>(t_end - t_begin).count() << " ms" << endl;
}
