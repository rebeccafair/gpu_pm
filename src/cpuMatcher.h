#ifndef MATCH_PATTERNS_H_
#define MATCH_PATTERNS_H_

// matchPatterns.h
// Searches for matches between events and patterns

#include "patternReader.h"
#include "eventReader.h"
#include "matchResults.h"

void matchByPatterns(const PatternContainer& p, const EventContainer& e, MatchResults& mr);
void matchByEvents(const PatternContainer& p, const EventContainer& e, MatchResults& mr);

#endif
