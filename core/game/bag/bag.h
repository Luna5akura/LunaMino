// core/game/bag/bag.h

#ifndef BAG_H
#define BAG_H

#include "../../piece/piece.h"

typedef struct {
    int current;
    PieceType sequence[7];
} Bag;

Bag* init_bag();
PieceType bag_next_piece(Bag* bag);

#endif