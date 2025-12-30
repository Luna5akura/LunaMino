// util/util.h
#ifndef UTIL_H
#define UTIL_H

#include <stdint.h>
#include <stdbool.h>

// --- 核心修改：无状态随机数函数 ---

// 这里的 state 指针指向 GameState 中的 rng_state
// 返回 0 ~ 32767
static inline uint16_t util_rand_next(uint32_t* state) {
    // 线性同余发生器 (LCG)
    *state = (*state) * 1103515245 + 12345;
    return (uint16_t)((*state >> 16) & 0x7FFF);
}

static inline int32_t util_max(int32_t a, int32_t b) {
    return (a > b) ? a : b;
}

static inline int32_t util_min(int32_t a, int32_t b) {
    return (a < b) ? a : b;
}

#endif