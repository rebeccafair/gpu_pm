#include <string>
#include <iostream>
#include <getopt.h>
#include <cuda_runtime.h>

#include "eventReader.h"
#include "patternReader.h"
#include "cpuMatcher.h"
#include "gpuMatcher.h"
#include "matchResults.h"
#include "testMatches.h"

using namespace std;

int main(int argc, char* argv[]) {


    string patternFile = "inputs/pattern_groups.bin";
    string eventFile = "inputs/single_track_hit_events.bin";
    int nThreads = 64;
    int nBlocks = 0;
    char opt;

    while((opt = getopt(argc,argv,"e:p:t:b:")) != -1) {
        switch(opt)
        {
            case 'e': eventFile = optarg; break;
            case 'p': patternFile = optarg; break;
            case 't': nThreads = atoi(optarg); break;
            case 'b': nBlocks = atoi(optarg); break;
            default: cerr << "Invalid argument" << endl; exit(EXIT_FAILURE);
        }
    }

    // Read pattern data
    PatternContainer p;
    readPatterns(patternFile, p);
    //printPatterns(p);

    // Read event data
    EventContainer e;
    readEvents(eventFile, e);
    //printEvents(e);

    // Perform cpu pattern matching
    MatchResults cpuResults;
    matchByEvents(p, e, cpuResults);
    //matchByPatterns(p, e, cpuResults);
    //printMatchResults(cpuResults);

    // Perform gpu setup and pattern matching with multiple groups per block
    GpuContext ctx1;
    MatchResults gpuResults1;
    createGpuContext(p, e, ctx1);
    runGpuMatching(p, e, ctx1, gpuResults1, nThreads, nBlocks);
    deleteGpuContext(ctx1);

    // Perform gpu setup and pattern matching by pattern block
    GpuContext ctx2;
    MatchResults gpuResults2;
    createGpuContext(p, e, ctx2);
    runGpuMatching(p, e, ctx2, gpuResults2, nThreads);
    deleteGpuContext(ctx2);

    // Compare cpu/gpu results
    compareMatches(cpuResults, gpuResults1);
    compareMatches(cpuResults, gpuResults2);

    return 0;
}
