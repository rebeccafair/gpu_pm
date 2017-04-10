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
    if (mr1.nMatches == mr2.nMatches) {
        cout << "nMatches comparison passed" << endl; 
    } else {
        cerr << "nMatches comparison failed: " << mr1.nMatches << " != " << mr2.nMatches << endl;
        cerr << "Exiting..." << endl;
        exit(EXIT_FAILURE);
    }

     //Reorder and compare patternIds
     sort(mr1.patternIds.begin(), mr1.patternIds.end());
     sort(mr2.patternIds.begin(), mr2.patternIds.end());

     for (int i = 0; i < mr1.nMatches; i++) {
          if (mr1.patternIds[i] != mr2.patternIds[i]) {
              cerr << "patternIds comparison failed on index " << i << ": " << mr1.patternIds[i] 
                   << " != " << mr2.patternIds[i] << endl;
              cerr << "Exiting..." << endl;
              exit(EXIT_FAILURE);
          }
     }
     cout << "patternIds comparison passed" << endl;
     cout << "Comparison of match results passed!" << endl;

} 

#endif
