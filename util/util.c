#include <time.h>

static unsigned long next = 1;

void srandom(unsigned int seed) {
    if (seed == 0) {
        seed = (unsigned int)time(NULL);
    }
    next = seed;
}

long int random() {
    next = next * 1103515245 + 12345;
    return (unsigned long)(next >> 16) & 0x7FFF;
}
