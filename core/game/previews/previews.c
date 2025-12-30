// core/game/previews/previews.c
#include "previews.h"

void previews_init(Previews* p, int length) {
    // 边界检查
    if (length > MAX_PREVIEW_CAPACITY) length = MAX_PREVIEW_CAPACITY;
    if (length < 1) length = 1;

    // 清零内存
    // 虽然 game_init 通常已经 memset 0 了，但为了安全再次清零
    memset(p->pieces, 0, sizeof(p->pieces));
    
    p->head = 0;
    p->capacity = (int8_t)length;
    // count 字段已移除，因为队列始终被视为填满状态
}