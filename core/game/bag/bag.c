// core/game/bag/bag.c

#include "bag.h"
#include <stdlib.h>
#include <string.h>

void shuffle_sequence(PieceType sequence[7]) {
    // 1. 先填充 0-6
    for (int i = 0; i < 7; i++) {
        sequence[i] = (PieceType)i;
    }

    // 2. Fisher-Yates 洗牌
    for (int i = 6; i > 0; i--) {
        // 【修复】使用 rand() 代替 random()，并强制取正，防止负数索引越界
        int r = rand(); 
        if (r < 0) r = -r;
        int j = r % (i + 1);

        PieceType temp = sequence[i];
        sequence[i] = sequence[j];
        sequence[j] = temp;
    }
}

Bag* init_bag() {
    Bag* bag = (Bag*)malloc(sizeof(Bag));
    // 【修复】清零内存
    memset(bag, 0, sizeof(Bag));
    
    shuffle_sequence(bag->sequence);
    bag->current = 0;
    return bag;
}

void free_bag(Bag* bag) {
    if (bag) free(bag);
}

Bag* copy_bag(Bag* bag) {
    if (!bag) return NULL;
    Bag* new_bag = (Bag*)malloc(sizeof(Bag));
    memcpy(new_bag, bag, sizeof(Bag));
    return new_bag;
}

PieceType bag_next_piece(Bag* bag) {
    if (!bag) return (PieceType)0;

    bag->current = bag->current + 1;
    if (bag->current >= 7) {
        shuffle_sequence(bag->sequence);
        bag->current = 0;
    }
    
    // 【安全检查】
    if (bag->current < 0 || bag->current > 6) bag->current = 0;
    
    PieceType t = bag->sequence[bag->current];
    // 再次检查类型范围
    if (t < 0 || t > 6) t = (PieceType)0;
    
    return t;
}