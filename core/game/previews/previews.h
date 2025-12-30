// core/game/previews/previews.h

#ifndef PREVIEWS_H
#define PREVIEWS_H

#include <stdint.h>
#include <string.h> // for memset
#include "../../piece/piece.h"

#define MAX_PREVIEW_CAPACITY 6

// 优化：压缩为 8 字节
// memory layout: [p0, p1, p2, p3, p4, p5, head, cap]
typedef struct {
    int8_t pieces[MAX_PREVIEW_CAPACITY];
    int8_t head;
    int8_t capacity;
} Previews;

// 初始化（需要重置内存，保留在 .c 中或这里均可，这里放 .c 以保持 API 风格）
void previews_init(Previews* p, int length);

// 极速复制
static inline void previews_copy(Previews* dest, const Previews* src) {
    *dest = *src; // 8-byte copy, compiles to a single MOV instruction
}

// 获取并替换下一个方块 (Hot Path: Inline)
static inline PieceType previews_next(Previews* p, PieceType input) {
    // 1. 获取当前队首 (即将进入游戏的方块)
    PieceType result = (PieceType)p->pieces[p->head];
    
    // 2. 覆盖写入新方块 (环形队列特性：Head 位置弹出后，正好是新的队尾位置)
    p->pieces[p->head] = (int8_t)input;
    
    // 3. 移动指针 (避免使用 % 操作符)
    p->head++;
    if (p->head >= p->capacity) {
        p->head = 0;
    }
    
    return result;
}

// 查看预览 (Hot Path: Inline)
static inline PieceType previews_peek(const Previews* p, int index) {
    // 安全检查：如果请求索引超出容量，返回非法值或 0
    if (index >= p->capacity) return (PieceType)0;
    
    // 计算物理索引
    // 逻辑索引 index 0 对应 physical[head]
    int8_t actual_index = p->head + (int8_t)index;
    
    // 优化：避免模运算
    // 因为 index < capacity，所以 actual_index 最大为 2*capacity - 1
    // 简单的减法比 % 快得多
    if (actual_index >= p->capacity) {
        actual_index -= p->capacity;
    }
    
    return (PieceType)p->pieces[actual_index];
}

#endif