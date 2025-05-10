// core/game/previews/previews.c

#include "previews.h"
#include <stdlib.h>

Previews*  init_previews(int length) {
    size_t total_size = sizeof(Previews) + length * sizeof(PieceType);
    Previews* previews = malloc(total_size);

    previews->current = 0;
    previews->length = length;
    return previews;
}

void free_previews(Previews* previews) {
    free(previews);
}

Previews* copy_previews(Previews* previews) {
    size_t total_size = sizeof(Previews) + previews->length * sizeof(PieceType);
    Previews* new_previews = malloc(total_size);

    memcpy(new_previews, previews, total_size);
    return new_previews;
}

PieceType next_preview(Previews* previews, PieceType input) {
    PieceType rtn = previews->previews[previews->current];
    previews->previews[previews->current] = input;
    previews->current = (previews->current + 1) % previews->length;
    return rtn;
}