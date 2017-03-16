#include <iostream>
#include <vector>
#include <string>

#include "eventReader.h"
#include "patternReader.h"
#include "matchPatterns.h"

using namespace std;

int main(int argc, char* argv[]) {

    string patternFile = "inputs/pattern_groups.bin";
    string eventFile = "inputs/single_track_hit_events.bin";
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
    //printPatterns();
    readEvents(eventFile);
    //printEvents();

    match();

    return 0;
}

