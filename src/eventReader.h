#ifndef EVENT_READER_H_
#define EVENT_READER_H_

// eventReader.h
// Reads and prints data from a binary event file

#include <string>
#include <vector>

using namespace std;

struct EventHeader {
    int nEvents;
};

struct EventContainer {
    EventHeader header;
    vector<int> eventId;
    vector<unsigned int> nCollections;
    vector<int> hashId;
    vector<int*> hashIdEventBegin;
    vector<unsigned int> nHits;
    vector<unsigned int*> nHitsEventBegin;
    vector<unsigned char> hitData;
    vector<unsigned char*> hitDataEventBegin;
    vector<unsigned int> subEventId;
    vector<unsigned int> barCode;
};

void readEvents(string eventFile, EventContainer& e);
void printEvents(const EventContainer& e);

#endif
