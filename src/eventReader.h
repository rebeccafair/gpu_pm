#ifndef EVENT_READER_H_
#define EVENT_READER_H_

// eventReader.h
// Reads and prints data from a binary event file

#include <string>
#include <vector>

using namespace std;

void readEvents(string eventFile);
void printEvents();

struct EventHeader {
    int nEvents;
};

extern EventHeader eventHeader;
extern vector<int> eventId;
extern vector<unsigned int> nCollections;
extern vector<int> hashId;
extern vector<int*> hashIdEventBegin;
extern vector<unsigned int> nHits;
extern vector<unsigned int*> nHitsEventBegin;
extern vector<unsigned char> hitData;
extern vector<unsigned char*> hitDataEventBegin;
extern vector<unsigned int> subEventId;
extern vector<unsigned int> barCode;

#endif
