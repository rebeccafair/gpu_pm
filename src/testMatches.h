#ifndef TEST_MATCHES_H_
#define TEST_MATCHES_H_

// testMatches.h
// Compares 2 sets of pattern matching results 

#include <iostream>
#include <vector>
#include <algorithm>
#include <stdlib.h>

#include "matchResults.h"

using namespace std;

inline void compareMatches(MatchResults mr1, MatchResults mr2) {
    cout << "Comparing match results..." << endl;

     // Reorder patternIds
     sort(mr1.patternIds.begin(), mr1.patternIds.end());
     sort(mr2.patternIds.begin(), mr2.patternIds.end());

     // Compare patternIds and record index where the comparison failed
     int failedIndex = -1;
     for (int i = 0; i < mr1.nMatches; i++) {
          if (mr1.patternIds[i] != mr2.patternIds[i]) {
              failedIndex = i; 
              break;
          }
     }

    // Print test results
    if (mr1.nMatches == mr2.nMatches) {
        cout << "nMatches comparison passed" << endl;

        if (failedIndex != -1) {
              cerr << "patternIds comparison failed on index " << failedIndex << ": " << mr1.patternIds[failedIndex]
                   << " != " << mr2.patternIds[failedIndex] << endl;
              cerr << "Exiting..." << endl;
              exit(EXIT_FAILURE);
        }
    } else {
        cerr << "nMatches comparison failed: " << mr1.nMatches << " != " << mr2.nMatches << endl;
        cerr << "Failed on patternId " << mr1.patternIds[failedIndex] << " != " << mr2.patternIds[failedIndex] << endl;
        cerr << "Exiting..." << endl;
        exit(EXIT_FAILURE);
    }

     cout << "patternIds comparison passed" << endl;
     cout << "Comparison of match results passed!" << endl;

} 

#endif
