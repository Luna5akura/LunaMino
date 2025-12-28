// core/game/previews/previews.h

#ifndef PREVIEWS_H
#define PREVIEWS_H

#include "../../piece/piece.h"

#define MAX_PREVIEW_CAPACITY 6

typedef struct {
    PieceType pieces[MAX_PREVIEW_CAPACITY];
    int head;
    int count;
    int capacity;
} Previews;

void previews_init(Previews* p, int length);

// 获取并替换下一个方块
// input: 从 Bag 中拿出的新方块，放入队尾
// return: 从队首弹出的方块，进入游戏
PieceType previews_next(Previews* p, PieceType input);

// 辅助：查看第 N 个预览方块（不弹出）- 用于 UI 渲染或 AI 观察
// index 0 是下一个即将出来的方块
PieceType previews_peek(const Previews* p, int index);

static inline void previews_copy(Previews* dest, const Previews* src) {
    *dest = *src;
}

#endif