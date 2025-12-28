// core/game/bag/bag.h

#ifndef BAG_H
#define BAG_H

#include "../../piece/piece.h"
#include "../../../util/util.h"

typedef struct {
    PieceType sequence[7];
    int head;
} Bag;

void bag_init(Bag* bag);
PieceType bag_next(Bag* bag);
static inline void bag_copy(Bag* dest, const Bag* src) {
    *dest = *src;
}

#endif