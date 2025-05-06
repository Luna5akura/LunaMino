// core/game/previews/previews.h

#ifndef PREVIEWS_H
#define PREVIEWS_H

#include "../../piece/piece.h"

typedef struct {
    int current;
    int length;
    PieceType previews[];
} Previews;

Previews*  init_previews(int length);
PieceType next_preview(Previews* previews, PieceType input);

#endif