// core/game/bag/bag.h
#ifndef BAG_H
#define BAG_H

#include <stdint.h>
#include "../../piece/piece.h"
#include "../../../util/util.h"

// 保持 8 字节极简结构
typedef struct {
    int8_t sequence[7]; 
    int8_t head;
} Bag;

// --- 核心修改：函数签名增加 rng_state ---
void bag_init(Bag* bag, uint32_t* rng_state);
PieceType bag_next(Bag* bag, uint32_t* rng_state);

static inline void bag_copy(Bag* dest, const Bag* src) {
    *dest = *src;
}

#endif