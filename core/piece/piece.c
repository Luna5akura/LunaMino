// core/piece/piece.c

#include "piece.h"


static const uint16_t PIECE_MASKS[7][4] = {
    {0x0F00, 0x2222, 0x00F0, 0x4444}, // I
    {0x6600, 0x6600, 0x6600, 0x6600}, // O
    {0x4E00, 0x4640, 0x0E40, 0x4C40}, // T
    {0x6C00, 0x4620, 0x06C0, 0x8C40}, // S
    {0xC600, 0x2640, 0x0C60, 0x4C80}, // Z
    {0x8E00, 0x6440, 0x0E20, 0x44C0}, // J
    {0x2E00, 0x4460, 0x0E80, 0xC440}  // L
    
};

void piece_init(Piece* p, PieceType type) {
    if (type < 0 || type >= PIECE_COUNT) type = PIECE_I;
    p->type = type;
    p->rotation = 0;
    piece_get_spawn_pos(type, &p->x, &p->y);
}

uint16_t piece_get_mask(const Piece* p) {
    return PIECE_MASKS[p->type][p->rotation];
}

void piece_move(Piece* p, MoveAction action) {
    switch (action) {
        case MOVE_LEFT:  p->x--; break;
        case MOVE_RIGHT: p->x++; break;
        case MOVE_DOWN:  p->y--; break;
        case MOVE_SOFT_DROP:  p->y--; break;
        case MOVE_HARD_DROP:  break;
    }
}

void piece_rotate(Piece* p, RotationAction action) {
    switch (action) {
        case ROTATE_CW:
            p->rotation = (p->rotation + 1) & 3; // % 4
            break;
        case ROTATE_CCW:
            p->rotation = (p->rotation + 3) & 3; // % 4
            break;
        case ROTATE_180:
            p->rotation = (p->rotation + 2) & 3; // % 4
            break;
    }
}

void piece_get_spawn_pos(PieceType type, int* x, int* y) {
    *x = 3; // 居中 (10列棋盘，x=3为左侧起第4列)
    *y = 20; // 顶部 (共23行，0-22，出生在20附近)
}