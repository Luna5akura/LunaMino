// core/util/util.c
#include "util.h"

// 定义全局随机数状态
// 初始值设为 1，防止未初始化调用时全 0
uint32_t _rng_next = 1;