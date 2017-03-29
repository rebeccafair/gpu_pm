#ifndef __TOOLS_H__
#define __TOOLS_H__

#include <vector>

// This function turns a vector of pointers (pVec) pointing at another 
// vector (vec) into a vector of indexes. This is required for accessing
// certain elements of vec on a GPU without using pointers. The resultant 
// vector of indices is 1 element larger than pVec, because the last element
// points the the last element of vec. This allows easy calculation of the number
// of collections/patterns in an event/group without having to transfer
// nCollections or nPattInGrp onto the GPU
template <typename T>
vector<unsigned int> pointerToIndex(const vector<T*>& pVec, const vector<T>& vec) {
    vector<unsigned int> indices(pVec.size() + 1);
    for (int i = 0; i < pVec.size(); i++) {
        indices[i] = pVec[i] - pVec[0];
    }
    indices[pVec.size()] = vec.size();

    return indices;
}

#endif
