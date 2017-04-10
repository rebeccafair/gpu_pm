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
    char opt;

    while((opt = getopt(argc,argv,"e:p:")) != -1) {
        switch(opt)
        {
            case 'e': eventFile = optarg; break;
            case 'p': patternFile = optarg; break;
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

    // Perform gpu setup and pattern matching
    GpuContext ctx;
    MatchResults gpuResults;
    copyContextToGpu(p, e, ctx);
    runMatchKernel(p, e, ctx, gpuResults);
    //printMatchResults(gpuResults);
    deleteGpuContext(ctx);

    // Compare cpu/gpu results
    compareMatches(cpuResults, gpuResults);

    return 0;
}
