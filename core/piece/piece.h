// core/piece/piece.h

#ifndef PIECE_H
#define PIECE_H

#include <stdint.h>
#include <stdbool.h>

typedef enum {
    PIECE_I = 0,
    PIECE_O,
    PIECE_T,
    PIECE_S,
    PIECE_Z,
    PIECE_J,
    PIECE_L,
    PIECE_COUNT
} PieceType;

// 0: 0 deg, 1: 90 deg (CW), 2: 180 deg, 3: 270 deg (CCW)
typedef int Rotation; 

typedef struct {
    PieceType type;    
    Rotation rotation;
    int x;
    int y;         // Left-top corner of the 4x4 shape in the board, global position starts from left-bottom corner of the board
} Piece;

typedef enum {
    MOVE_LEFT,
    MOVE_RIGHT,
    MOVE_DOWN,
    MOVE_SOFT_DROP,
    MOVE_HARD_DROP
} MoveAction;

typedef enum {
    ROTATE_CW,
    ROTATE_CCW,
    ROTATE_180
} RotationAction;


void piece_init(Piece* p, PieceType type);
uint16_t piece_get_mask(const Piece* p);
void piece_move(Piece* p, MoveAction action);
void piece_rotate(Piece* p, RotationAction action);
void piece_get_spawn_pos(PieceType type, int* x, int* y);

#endif


