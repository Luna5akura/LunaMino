// core/game/previews/previews.c

#include "previews.h"
#include <string.h>

void previews_init(Previews* p, int length) {
    if (length > MAX_PREVIEW_CAPACITY) length = MAX_PREVIEW_CAPACITY;
    if (length < 1) length = 1;

    memset(p->pieces, 0, sizeof(p->pieces));
    
    p->head = 0;
    p->capacity = length;
    p->count = 0; 
}

PieceType previews_next(Previews* p, PieceType input) {
    // 这里的逻辑是一个环形队列
    // 1. 取出 head 指向的元素 (旧的队首)
    PieceType result = p->pieces[p->head];
    
    // 2. 将新元素覆盖到当前 head 位置
    // 为什么是覆盖？
    // 因为在固定长度的滑动窗口中，队首出去后，那个位置正好变成了逻辑上的队尾的下一个位置
    // 举例：[A, B, C], head=0. 
    // 取出 A. 此时逻辑变成 [B, C, ?]. 
    // 实际上我们在 0 的位置写入 D. 数组变成 [D, B, C]. head 移到 1 (指向 B).
    // 逻辑序列: B, C, D. 正确。
    p->pieces[p->head] = input;
    
    // 3. 移动指针
    p->head = (p->head + 1) % p->capacity;
    
    return result;
}

PieceType previews_peek(const Previews* p, int index) {
    if (index >= p->capacity) return (PieceType)0;
    
    int actual_index = (p->head + index) % p->capacity;
    return p->pieces[actual_index];
}
