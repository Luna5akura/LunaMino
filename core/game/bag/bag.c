// core/game/bag/bag.c

#include "bag.h"
// 如果 _rng_next 没有在其他地方定义，可以在这里定义
unsigned long _rng_next = 1; 

static void _shuffle(Bag* bag) {
    for (int i = 0; i < 7; i++) {
        bag->sequence[i] = (PieceType)i;
    }
    for (int i = 6; i > 0; i--) {
        int j = magic_random() % (i + 1);
        
        PieceType temp = bag->sequence[i];
        bag->sequence[i] = bag->sequence[j];
        bag->sequence[j] = temp;
    }
}

void bag_init(Bag* bag) {
    _shuffle(bag);
    bag->head = 0;
}

PieceType bag_next(Bag* bag) {
    PieceType piece = bag->sequence[bag->head];
    
    bag->head++;
    if (bag->head >= 7) {
        _shuffle(bag);
        bag->head = 0;
    }
    
    return piece;
}