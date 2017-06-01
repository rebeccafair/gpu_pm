#ifndef MATCH_PATTERNS_H_
#define MATCH_PATTERNS_H_

// matchPatterns.h
// Searches for matches between events and patterns

#include "patternReader.h"
#include "eventReader.h"
#include "matchResults.h"

// Loops over patterns first and matches each pattern with event hits
void matchByPatterns(const PatternContainer& p, const EventContainer& e, MatchResults& mr);

// Loops over events first and matches patterns with each event
void matchByEvents(const PatternContainer& p, const EventContainer& e, MatchResults& mr);

#endif
