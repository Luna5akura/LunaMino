static unsigned long next = 1;

void srandom(unsigned int seed) {
    next = seed;
}

long int random() {
    next = next * 1103515245 + 12345;
    return (unsigned long)(next >> 16) & 0x7FFF;
}
