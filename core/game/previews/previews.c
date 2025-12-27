// core/game/previews/previews.c

#include "previews.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h> // for printf

Previews*  init_previews(int length) {
    if (length <= 0) length = 5; // 默认值保护

    size_t total_size = sizeof(Previews) + length * sizeof(PieceType);
    Previews* previews = malloc(total_size);
    if (!previews) exit(1);

    // 显式清零
    memset(previews, 0, total_size);

    previews->current = 0;
    previews->length = length;
    return previews;
}

void free_previews(Previews* previews) {
    if (previews) free(previews);
}

Previews* copy_previews(Previews* previews) {
    if (!previews) return NULL;
    size_t total_size = sizeof(Previews) + previews->length * sizeof(PieceType);
    Previews* new_previews = malloc(total_size);
    if (!new_previews) exit(1);

    memcpy(new_previews, previews, total_size);
    return new_previews;
}

PieceType next_preview(Previews* previews, PieceType input) {
    if (!previews) return (PieceType)0;

    // 安全检查索引
    if (previews->current < 0 || previews->current >= previews->length) {
        previews->current = 0; // 强制修复
    }

    PieceType rtn = previews->previews[previews->current];
    previews->previews[previews->current] = input;
    previews->current = (previews->current + 1) % previews->length;
    return rtn;
}