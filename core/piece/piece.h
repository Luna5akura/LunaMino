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
} PieceType; // Underlying type is int by default, but we can use it with uint8_t in struct if needed

// 0: 0 deg, 1: 90 deg (CW), 2: 180 deg, 3: 270 deg (CCW)
typedef uint8_t Rotation;

typedef struct {
    uint8_t type;
    uint8_t rotation;
    int8_t pos[2]; // Left-top corner of the 4x4 shape in the board, global position starts from left-bottom corner of the board
} Piece;

typedef enum {
    // Bit 0: 0 -> -1 (Minus), 1 -> +1 (Plus)
    // Bit 1: 0 -> X axis,      1 -> Y axis
    
    MOVE_LEFT      = 0, // Binary 000: X axis, -1 (x--)
    MOVE_RIGHT     = 1, // Binary 001: X axis, +1 (x++)
    MOVE_DOWN      = 2, // Binary 010: Y axis, -1 (y--)
    
    // HARD_DROP 需要特殊处理，设为一个不干扰计算的值，或者在逻辑前拦截
    MOVE_HARD_DROP = 4, // Binary 100: 单独处理
    
    // SOFT_DROP 物理上等于 DOWN，但如果逻辑上需要区分，
    // 可以设为 6 (110)，这样 Bit 0 和 Bit 1 与 DOWN 保持一致
    MOVE_SOFT_DROP = 6  // Binary 110: Y axis, -1 (Same physics as DOWN)
} MoveAction;

typedef enum {
    ROTATE_CW  = 1, // +1: 0->1->2->3->0
    ROTATE_180 = 2, // +2: 0->2, 1->3
    ROTATE_CCW = 3  // +3: 等价于 -1 (逆时针)
} RotationAction;

void piece_init(Piece* p, PieceType type);

static inline uint16_t piece_get_mask(const Piece* p) {
    static const uint16_t PIECE_MASKS[7][4] = {
        {0x0F00, 0x2222, 0x00F0, 0x4444}, // I
        {0x6600, 0x6600, 0x6600, 0x6600}, // O
        {0x4E00, 0x4640, 0x0E40, 0x4C40}, // T
        {0x6C00, 0x4620, 0x06C0, 0x8C40}, // S
        {0xC600, 0x2640, 0x0C60, 0x4C80}, // Z
        {0x8E00, 0x6440, 0x0E20, 0x44C0}, // J
        {0x2E00, 0x4460, 0x0E80, 0xC440}  // L
    };
    return PIECE_MASKS[p->type][p->rotation];
}

static inline void piece_move(Piece* p, MoveAction action) {
    // 1. 快速过滤 HARD_DROP (不做位移)
    if (action == MOVE_HARD_DROP) return;

    // 2. 提取轴向 (0: x, 1: y)
    // 取第 1 位
    int8_t axis = (action >> 1) & 1;

    // 3. 提取增量 (-1 或 +1)
    // 取第 0 位，如果是 0 则结果为 -1，是 1 则结果为 +1
    // 公式: (bit * 2) - 1
    int8_t delta = ((action & 1) << 1) - 1;

    // 4. 应用坐标
    p->pos[axis] += delta;
}

static inline void piece_rotate(Piece* p, RotationAction action) {
    p->rotation = (p->rotation + action) & 3;
}

void piece_get_spawn_pos(PieceType type, int8_t* x, int8_t* y);

#endif