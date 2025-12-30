// core/piece/piece.c
#include "piece.h"

void piece_init(Piece* p, PieceType type) {
    if (type < 0 || type >= PIECE_COUNT) type = PIECE_I;
    p->type = type;
    p->rotation = 0;
    piece_get_spawn_pos(type, &p->pos[0], &p->pos[1]);
}


void piece_get_spawn_pos(PieceType type, int8_t* x, int8_t* y) {
    *x = 3; // 居中 (10列棋盘，x=3为左侧起第4列)
    *y = 20; // 顶部 (共23行，0-22，出生在20附近)
}