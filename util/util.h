// util/util.h

typedef enum {
    FALSE = 0,
    TRUE = 1
} Bool;

long int random();
void srandom(unsigned int seed);