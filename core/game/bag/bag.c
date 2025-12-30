// core/game/bag/bag.c
#include "bag.h"

// 内部函数也需要接收 rng_state
static void _fill_and_shuffle(Bag* bag, uint32_t* rng_state) {
    // 1. 填充
    bag->sequence[0] = 0;
    bag->sequence[1] = 1;
    bag->sequence[2] = 2;
    bag->sequence[3] = 3;
    bag->sequence[4] = 4;
    bag->sequence[5] = 5;
    bag->sequence[6] = 6;
    // 2. 洗牌 (Fisher-Yates)
    for (int i = 6; i > 0; i--) {
        // --- 优化修改：用乘法+移位替换 %，减少偏差和开销 ---
        uint16_t r = magic_random(rng_state);
        int j = ((uint32_t)r * (i + 1)) >> 15;
       
        int8_t temp = bag->sequence[i];
        bag->sequence[i] = bag->sequence[j];
        bag->sequence[j] = temp;
    }
}

// 初始化时需要洗牌，所以需要 rng_state
void bag_init(Bag* bag, uint32_t* rng_state) {
    _fill_and_shuffle(bag, rng_state);
    bag->head = 0;
}

PieceType bag_next(Bag* bag, uint32_t* rng_state) {
    if (bag->head >= 7) {
        // 重新洗牌，传入状态
        _fill_and_shuffle(bag, rng_state);
        bag->head = 0;
    }
   
    PieceType piece = (PieceType)bag->sequence[bag->head];
    bag->head++;
   
    return piece;
}