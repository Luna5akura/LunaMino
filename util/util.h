// util/util.h
#ifndef UTIL_H
#define UTIL_H
#include <stdint.h>
typedef int Bool;
#define TRUE 1
#define FALSE 0
extern unsigned long _rng_next;
static inline void magic_srandom(unsigned int seed) {
    if (seed == 0) seed = 1;
    _rng_next = seed;
}
static inline long int magic_random() {
    _rng_next = _rng_next * 1103515245 + 12345;
    return (unsigned long)(_rng_next >> 16) & 0x7FFF;
}
#endif