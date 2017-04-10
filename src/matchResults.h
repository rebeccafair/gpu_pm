#ifndef MATCH_RESULTS_H_
#define MATCH_RESULTS_H_

// matchResults.h
// Defines struct to store output from pattern matching 

#include <iostream>
#include <vector>

using namespace std;

struct MatchResults {
    int nMatches;
    vector<int> patternIds;
};

inline void printMatchResults(const MatchResults& mr) {
    cout << "Printing match results..." << endl;
    cout << mr.nMatches << " matches found" << endl;
    cout << "Matching ids:" << endl;
    for (int i = 0; i < mr.nMatches; i++) {
        cout << mr.patternIds[i] << endl;
    }
    cout << "Finished printing match results" << endl;
}

#endif
