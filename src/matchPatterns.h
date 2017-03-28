#ifndef MATCH_PATTERNS_H_
#define MATCH_PATTERNS_H_

// matchPatterns.h
// Searches for matches between events and patterns

#include "patternReader.h"
#include "eventReader.h"

void matchByPatterns(const PatternContainer& p, const EventContainer& e);
void matchByEvents(const PatternContainer& p, const EventContainer& e);

#endif
