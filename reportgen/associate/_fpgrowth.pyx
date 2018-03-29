#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: cdivision=True
#cython: embedsignature=True
#cython: language_level=3
#cython: language=c++

from libcpp.set cimport set as cppset
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.unordered_map cimport unordered_map as hashmap


cdef int _BUCKETING_FEW_ITEMS = 16
BUCKETING_FEW_ITEMS = _BUCKETING_FEW_ITEMS

ctypedef cppset[int] itemset_t
ctypedef vector[pair[itemset_t, int]] itemsets_t


cpdef itemsets_t bucketing_count(list db,
                                 cppset[int] frequent_items,
                                 int min_support):
    """ The bucketing count operation. """
    cdef:
        int i, j, k = frequent_items.size()

        vector[int] inv_map = vector[int]()
        hashmap[int, int] fwd_map = hashmap[int, int]()
        int index = 0

        vector[int] buckets = vector[int](2**k, 0)
        pair[int, vector[int]] transaction
        int tid = 0
        int item

        int count
        itemset_t result
        itemsets_t results = itemsets_t()

    # Forward and inverse mapping of frequent_items to [0, n_items)
    for item in frequent_items:
        inv_map.push_back(item)
        fwd_map[item] = index
        index += 1
    # Project transactions
    for transaction in db:
        tid = 0
        for item in transaction.second:
            if not frequent_items.count(item): continue
            tid |= 1 << fwd_map.at(item)
        buckets[tid] += transaction.first
    # Aggregate bucketing counts ([2], Figure 5)
    for i in range(0, k):
        i = 1 << i
        for j in range(1 << k):
            if j & i == 0:
                buckets[j] += buckets[j + i]
    # Count results
    for tid in range(1, <int>buckets.size()):
        count = buckets[tid]
        if count >= min_support:
            result = itemset_t()
            for i in range(_BUCKETING_FEW_ITEMS):
                if tid & 1 << i:
                    result.insert(inv_map[i])
            results.push_back(pair[itemset_t, int](result, count))
    return results
