// core/piece/piece.c

#include "piece.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h> // for printf

const int PIECE_SHAPES[7][4][4][4] = {
    // I_PIECE
    {
        // UP
        {
            {0, 0, 0, 0},
            {1, 1, 1, 1},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        },
        // RIGHT
        {
            {0, 0, 1, 0},
            {0, 0, 1, 0},
            {0, 0, 1, 0},
            {0, 0, 1, 0}
        },
        // DOWN
        {
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {1, 1, 1, 1},
            {0, 0, 0, 0}
        },
        // LEFT
        {
            {0, 1, 0, 0},
            {0, 1, 0, 0},
            {0, 1, 0, 0},
            {0, 1, 0, 0}
        }
    },
    // O_PIECE
    {
        {
            {0, 1, 1, 0},
            {0, 1, 1, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        },
        {
            {0, 1, 1, 0},
            {0, 1, 1, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        },
        {
            {0, 1, 1, 0},
            {0, 1, 1, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        },
        {
            {0, 1, 1, 0},
            {0, 1, 1, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // T_PIECE
    {
        // UP
        {
            {0, 1, 0, 0},
            {1, 1, 1, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        },
        // RIGHT
        {
            {0, 1, 0, 0},
            {0, 1, 1, 0},
            {0, 1, 0, 0},
            {0, 0, 0, 0}
        },
        // DOWN
        {
            {0, 0, 0, 0},
            {1, 1, 1, 0},
            {0, 1, 0, 0},
            {0, 0, 0, 0}
        },
        // LEFT
        {
            {0, 1, 0, 0},
            {1, 1, 0, 0},
            {0, 1, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // S_PIECE
    {
        // UP
        {
            {0, 1, 1, 0},
            {1, 1, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        },
        // RIGHT
        {
            {0, 1, 0, 0},
            {0, 1, 1, 0},
            {0, 0, 1, 0},
            {0, 0, 0, 0}
        },
        // DOWN
        {
            {0, 0, 0, 0},
            {0, 1, 1, 0},
            {1, 1, 0, 0},
            {0, 0, 0, 0}
        },
        // LEFT
        {
            {1, 0, 0, 0},
            {1, 1, 0, 0},
            {0, 1, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // Z_PIECE
    {
        // UP
        {
            {1, 1, 0, 0},
            {0, 1, 1, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        },
        // RIGHT
        {
            {0, 0, 1, 0},
            {0, 1, 1, 0},
            {0, 1, 0, 0},
            {0, 0, 0, 0}
        },
        // DOWN
        {
            {0, 0, 0, 0},
            {1, 1, 0, 0},
            {0, 1, 1, 0},
            {0, 0, 0, 0}
        },
        // LEFT
        {
            {0, 1, 0, 0},
            {1, 1, 0, 0},
            {1, 0, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // J_PIECE
    {
        // UP
        {
            {1, 0, 0, 0},
            {1, 1, 1, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        },
        // RIGHT
        {
            {0, 1, 1, 0},
            {0, 1, 0, 0},
            {0, 1, 0, 0},
            {0, 0, 0, 0}
        },
        // DOWN
        {
            {0, 0, 0, 0},
            {1, 1, 1, 0},
            {0, 0, 1, 0},
            {0, 0, 0, 0}
        },
        // LEFT
        {
            {0, 1, 0, 0},
            {0, 1, 0, 0},
            {1, 1, 0, 0},
            {0, 0, 0, 0}
        }
    },
    // L_PIECE
    {
        // UP
        {
            {0, 0, 1, 0},
            {1, 1, 1, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0}
        },
        // RIGHT
        {
            {0, 1, 0, 0},
            {0, 1, 0, 0},
            {0, 1, 1, 0},
            {0, 0, 0, 0}
        },
        // DOWN
        {
            {0, 0, 0, 0},
            {1, 1, 1, 0},
            {1, 0, 0, 0},
            {0, 0, 0, 0}
        },
        // LEFT
        {
            {1, 1, 0, 0},
            {0, 1, 0, 0},
            {0, 1, 0, 0},
            {0, 0, 0, 0}
        }
    }
};

Piece* init_piece(PieceType type) {
    // 【安全检查】防止数组越界导致 Segfault
    if (type < 0 || type > 6) {
        printf("[C-Error] init_piece received invalid type: %d\n", type);
        // 返回一个默认值 (例如 T块) 或者 NULL
        // 这里为了防止后续代码崩溃，强制修正为 0 (I_PIECE)
        type = (PieceType)0; 
    }

    Piece* piece = (Piece*)malloc(sizeof(Piece));
    if (piece == NULL) {
        exit(1);
    }
    piece->type = type;
    piece->x = 0;
    piece->y = 0;
    piece->rotation = (Rotation)0;

    memcpy(
        piece->shape, 
        PIECE_SHAPES[type][0], 
        sizeof(PIECE_SHAPES[type][0])
    );
    
    return piece;
}

void free_piece(Piece* piece) {
    if (piece) free(piece);
}

Piece* copy_piece(Piece* piece) {
    if (piece == NULL) return NULL;
    Piece* new_piece = (Piece*)malloc(sizeof(Piece));
    if (new_piece == NULL) {
        exit(1);
    }
    memcpy(new_piece, piece, sizeof(Piece));
    return new_piece;
}

void move_piece(Piece* piece, MoveAction action) {
    // no hard drop, hard drop is handledd by function hard_drop
    switch (action) {
        case MOVE_LEFT:
            piece->x--;
            break;
        case MOVE_RIGHT:
            piece->x++;
            break;
        case MOVE_DOWN:
            piece->y--;
            break;
        case MOVE_SOFT_DROP:
            piece->y--;
            break;
        case MOVE_HARD_DROP:
            break;
    }
}

void displace_piece(Piece* piece, const int direction[2]) {
    // direction: positive x is right, positive y is up
    piece->x += direction[0];
    piece->y += direction[1];
}

void rotate_piece(Piece* piece, RotationAction action) {
    int new_rotation;
    switch (action) {
        case ROTATE_CW:
            new_rotation = (Rotation)((int)piece->rotation + 1) % 4;
            break;
        case ROTATE_CCW:
            new_rotation = (Rotation)((int)piece->rotation + 3) % 4;
            break;
        case ROTATE_180:
            new_rotation = (Rotation)((int)piece->rotation + 2) % 4;
            break;
        default:
            new_rotation = (Rotation)0;
            break;
    }
    
    // 【安全检查】虽然理论上不会，但防止 rotation 越界
    if (new_rotation < 0) new_rotation = 0;
    
    memcpy(
        piece->shape, 
        PIECE_SHAPES[(int)piece->type][new_rotation], 
        sizeof(PIECE_SHAPES[(int)piece->type][new_rotation])
    );
    piece->rotation = new_rotation;
}