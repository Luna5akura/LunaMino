// core/piece/piece.h

#ifndef PIECE_H
#define PIECE_H

typedef enum {
    I_PIECE, O_PIECE, T_PIECE, S_PIECE, Z_PIECE, J_PIECE, L_PIECE
} PieceType;

typedef enum {
    UP, RIGHT, DOWN, LEFT
} Rotation;

typedef enum {
    MOVE_LEFT, MOVE_RIGHT, MOVE_DOWN, MOVE_SOFT_DROP, MOVE_HARD_DROP
} MoveAction;

typedef enum {
    ROTATE_CW, ROTATE_CCW, ROTATE_180 
} RotationAction;

typedef struct {
    PieceType type;    
    int x, y;         // Left-top corner of the 4x4 shape in the board, global position starts from left-bottom corner of the board
    Rotation rotation;     // 0 is up, 1 is right, 2 is down, 3 is left
    int shape[4][4];  // 0 is empty, 1 is filled
} Piece;

Piece* init_piece(PieceType type);
void move_piece(Piece* piece, MoveAction action);
void displace_piece(Piece* piece, const int direction[2]);
void rotate_piece(Piece* piece, RotationAction action);

#endif