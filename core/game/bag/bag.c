// core/game/bag/bag.c

#include "bag.h"
#include <stdlib.h>

void shuffle_sequence(PieceType sequence[7]) {
    for (int i = 0; i < 7; i++) {
        sequence[i] = (PieceType)i;
    }

    for (int i = 6; i > 0; i--) {
        int j = random() % (i + 1);
        PieceType temp = sequence[i];
        sequence[i] = sequence[j];
        sequence[j] = temp;
    }
}

Bag* init_bag() {
    Bag* bag = (Bag*)malloc(sizeof(Bag));
    shuffle_sequence(bag->sequence);
    bag->current = 0;
    return bag;
}

void free_bag(Bag* bag) {
    free(bag);
}

PieceType bag_next_piece(Bag* bag) {
    bag->current = bag->current + 1;
    if (bag->current >= 7) {
        shuffle_sequence(bag->sequence);
        bag->current = 0;
    }
    return bag->sequence[bag->current];
}

