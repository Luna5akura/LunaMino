// core/game/game.c

#include "game.h"
#include <stdlib.h>
#include <string.h>
#include <time.h> // for magic_srandom seed

#define GAME_VISIBLE_HEIGHT 20
#define DEFAULT_PREVIEW_COUNT 5
#define DEFAULT_IS_HOLD_ENABLED true
#define DEFAULT_SEED 0

// ============================================================================
// SRS (Super Rotation System) Tables - Optimized to int8_t
// ============================================================================
// 维度: [RotationState(4)][Direction(2)][TestIndex(5)][Axis(2)]
// Direction: 0 = CW, 1 = CCW
// Axis: 0 = x, 1 = y

static const int8_t NORMAL_PIECE_NORMAL_SRS[4][2][5][2] = {
    { // 0 (UP)
        { {0, 0}, {-1, 0}, {-1, 1}, {0, -2}, {-1, -2} }, // -> 1 (CW)
        { {0, 0}, {1, 0}, {1, 1}, {0, -2}, {1, -2} }     // -> 3 (CCW)
    },
    { // 1 (RIGHT)
        { {0, 0}, {1, 0}, {1, -1}, {0, 2}, {1, 2} },     // -> 2 (CW)
        { {0, 0}, {1, 0}, {1, -1}, {0, 2}, {1, 2} }      // -> 0 (CCW)
    },
    { // 2 (DOWN)
        { {0, 0}, {1, 0}, {1, 1}, {0, -2}, {1, -2} },    // -> 3 (CW)
        { {0, 0}, {-1, 0}, {-1, 1}, {0, -2}, {-1, -2} }  // -> 1 (CCW)
    },
    { // 3 (LEFT)
        { {0, 0}, {-1, 0}, {-1, -1}, {0, 2}, {-1, 2} },  // -> 0 (CW)
        { {0, 0}, {-1, 0}, {-1, -1}, {0, 2}, {-1, 2} }   // -> 2 (CCW)
    }
};

static const int8_t I_PIECE_NORMAL_SRS[4][2][5][2] = {
    { // 0
        { {0, 0}, {-2, 0}, {1, 0}, {-2, -1}, {1, 2} }, // -> 1
        { {0, 0}, {-1, 0}, {2, 0}, {-1, 2}, {2, -1} }  // -> 3
    },
    { // 1
        { {0, 0}, {-1, 0}, {2, 0}, {-1, 2}, {2, -1} }, // -> 2
        { {0, 0}, {2, 0}, {-1, 0}, {2, 1}, {-1, -2} }  // -> 0
    },
    { // 2
        { {0, 0}, {2, 0}, {-1, 0}, {2, 1}, {-1, -2} }, // -> 3
        { {0, 0}, {1, 0}, {-2, 0}, {1, -2}, {-2, 1} }  // -> 1
    },
    { // 3
        { {0, 0}, {1, 0}, {-2, 0}, {1, -2}, {-2, 1} }, // -> 0
        { {0, 0}, {-2, 0}, {1, 0}, {-2, -1}, {1, 2} }  // -> 2
    }
};

// 180度旋转表
// 维度: [RotationState(4)][TestIndex(6/2)][Axis(2)]
static const int8_t NORMAL_PIECE_180_SRS[4][6][2] = {
    { {0, 0}, {0, 1}, {1, 1}, {-1, 1}, {1, 0}, {-1, 0} }, // 0 -> 2
    { {0, 0}, {1, 0}, {1, 2}, {1, 1}, {0, 2}, {0, 1} },   // 1 -> 3
    { {0, 0}, {0, -1}, {-1, -1}, {1, -1}, {-1, 0}, {1, 0} }, // 2 -> 0
    { {0, 0}, {-1, 0}, {-1, 2}, {-1, 1}, {0, 2}, {0, 1} },   // 3 -> 1
};

static const int8_t I_PIECE_180_SRS[4][2][2] = {
    { {0, 0}, {0, -1} }, // 0 -> 2
    { {0, 0}, {1, 0} },  // 1 -> 3
    { {0, 0}, {0, 1} },  // 2 -> 0
    { {0, 0}, {-1, 0} }, // 3 -> 1
};

// T-Spin 判定辅助坐标 (x, y) 相对左下角
static const int8_t T_CORNERS[4][2] = {
    {0, 0}, {0, 2}, {2, 0}, {2, 2}
};
// T-Spin Mini 判定需要的 "背部" 坐标
static const int8_t T_RECTS[4][2][2] = {
    { {0, 0}, {0, 2} }, // Rot 0 (Point Up) -> Check back (0,0), (0,2)? No, check bottom corners usually
    { {0, 2}, {2, 2} }, 
    { {2, 0}, {2, 2} }, 
    { {0, 0}, {2, 0} }
};

// ============================================================================
// Lifecycle
// ============================================================================

Game* game_init(const GameConfig* config) {
    Game* game = (Game*)malloc(sizeof(Game));
    if (!game) return NULL;
    memset(game, 0, sizeof(Game));

    // Config setup
    game->config.preview_count = config ? config->preview_count : DEFAULT_PREVIEW_COUNT;
    if (game->config.preview_count > MAX_PREVIEW_CAPACITY) 
        game->config.preview_count = MAX_PREVIEW_CAPACITY;
    game->config.is_hold_enabled = config ? config->is_hold_enabled : DEFAULT_IS_HOLD_ENABLED;
    game->config.seed = config ? config->seed : DEFAULT_SEED;

    // --- 核心修改 1: 初始化 RNG 状态 ---
    game->state.rng_state = game->config.seed;
    if (game->state.rng_state == 0) game->state.rng_state = 1;

    // --- 核心修改 2: 传递 RNG 状态指针 ---
    // Init Bag
    bag_init(&game->state.bag, &game->state.rng_state);
    
    // Init Previews
    previews_init(&game->state.previews, game->config.preview_count);
    for (int i = 0; i < game->state.previews.capacity; i++) {
        // bag_next 需要传入 rng_state 的地址
        PieceType next = bag_next(&game->state.bag, &game->state.rng_state);
        (void)previews_next(&game->state.previews, next);
    }

    // Board init
    board_init(&game->board);

    // Spawn first piece
    PieceType type = previews_next(&game->state.previews, 
                                   bag_next(&game->state.bag, &game->state.rng_state));
    piece_init(&game->current_piece, type);

    game->state.can_hold_piece = true;
    game->state.ren = -1;
    
    return game;
}


void game_free(Game* game) {
    if (game) free(game);
}

Game* game_copy(const Game* src) {
    if (!src) return NULL;
    Game* dest = (Game*)malloc(sizeof(Game));
    if (dest) {
        memcpy(dest, src, sizeof(Game));
    }
    return dest;
}

// ============================================================================
// Core Logic: Collision & Movement
// ============================================================================

// 优化的按行碰撞检测
bool board_piece_overlaps(const Board* board, const Piece* p) {
    uint16_t mask = piece_get_mask(p);
    
    // 循环展开或保持循环，编译器通常能很好优化固定次数循环
    for (int r = 0; r < 4; r++) {
        // 提取 Piece 的当前行 (从高位开始)
        // mask 布局: Row 0 (Bits 15-12), Row 1 (Bits 11-8)...
        uint16_t row_bits = (mask >> ((3 - r) * 4)) & 0xF;
        
        if (!row_bits) continue; // 空行跳过优化

        int y = p->pos[1] - r;

        // 1. 底部碰撞
        if (y < 0) return true;
        // 2. 顶部之上 (缓冲区)，不视为碰撞，除非在生成位置就卡住（另行处理）
        if (y >= BOARD_HEIGHT) continue;

        // 3. 左右墙壁与方块碰撞
        // 我们根据 p->x 将 row_bits 映射到 board 的位空间
        
        if (p->pos[0] >= 0) {
            // 向右移位
            // 检查是否撞右墙：如果移位后超过 10 位 (0x3FF)
            if (((row_bits << p->pos[0]) & ~BOARD_ROW_MASK) != 0) return true;
            
            // 检查是否撞方块
            if (board->rows[y] & (row_bits << p->pos[0])) return true;
        } else {
            // 向左移位 (x 是负数)
            int shift = -p->pos[0];
            
            // 检查是否撞左墙：如果被移出去的位包含 1
            // ((1 << shift) - 1) 生成低位全是 1 的掩码
            if ((row_bits & ((1 << shift) - 1)) != 0) return true;
            
            // 检查是否撞方块
            if (board->rows[y] & (row_bits >> shift)) return true;
        }
    }
    return false;
}

static bool game_try_displace(Game* game, int8_t dx, int8_t dy) {
    game->current_piece.pos[0] += dx;
    game->current_piece.pos[1] += dy;
    
    if (board_piece_overlaps(&game->board, &game->current_piece)) {
        // 还原
        game->current_piece.pos[0] -= dx;
        game->current_piece.pos[1] -= dy;
        return false;
    }
    return true;
}

bool game_try_move(Game* game, MoveAction action) {
    if (game->is_game_over) return false;
    
    if (action == MOVE_HARD_DROP) {
        // 优化：计算 Shadow Height 直接下落，代替逐格判断
        int drop = 0;
        // 预先向下探测，直到重叠
        // 这里可以直接操作 y，因为 overlaps 很快
        while (!board_piece_overlaps(&game->board, &game->current_piece)) {
            game->current_piece.pos[1]--;
            drop++;
        }
        // 此时 current_piece 已经陷入地里，回退一格
        game->current_piece.pos[1]++; 
        drop--; // 实际下落格数
        
        // 锁定上一次旋转状态为 0 (Hard Drop 重置 Spin 状态吗？通常不，但移动会)
        // 规范：Hard Drop 自身不算作使得 Spin 失效的移动，但落地后判定需要位置
        game->state.is_last_rotate = 0; 
        return true;
    }

    Piece backup = game->current_piece;
    piece_move(&game->current_piece, action);

    if (board_piece_overlaps(&game->board, &game->current_piece)) {
        game->current_piece = backup; // 碰撞，还原
        return false;
    }

    // 移动成功，重置 Spin 标记 (除非是 Soft Drop? 标准中 Soft Drop 也重置 Spin 奖励)
    game->state.is_last_rotate = 0;
    return true;
}

bool game_try_rotate(Game* game, RotationAction action) {
    if (game->is_game_over) return false;

    Piece* p = &game->current_piece;
    Rotation old_rot = p->rotation;
    
    // 执行旋转 (piece.h 里的优化版)
    piece_rotate(p, action);

    // 选择 Kick Table
    const int8_t (*kicks)[2]; // Pointer to array of 2 int8_t
    int kick_count;

    if (action == ROTATE_180) {
        kick_count = (p->type == PIECE_I) ? 2 : 6;
        kicks = (p->type == PIECE_I) ? 
                I_PIECE_180_SRS[old_rot] : 
                NORMAL_PIECE_180_SRS[old_rot];
    } else {
        kick_count = 5;
        // action: CW=1, CCW=3. Maps to 0 and 1 for array index
        int dir_idx = (action == ROTATE_CW) ? 0 : 1;
        
        kicks = (p->type == PIECE_I) ? 
                I_PIECE_NORMAL_SRS[old_rot][dir_idx] : 
                NORMAL_PIECE_NORMAL_SRS[old_rot][dir_idx];
    }

    // 尝试 Kick
    for (int i = 0; i < kick_count; i++) {
        // kicks[i][0] is x, kicks[i][1] is y
        if (game_try_displace(game, kicks[i][0], kicks[i][1])) {
            // 成功旋转
            game->state.is_last_rotate = i + 1; // 1-based index (0 means no rotate)
            return true;
        }
    }

    // 全部失败，还原旋转
    p->rotation = old_rot;
    return false;
}

bool game_hold_piece(Game* game) {
    if (game->is_game_over || !game->config.is_hold_enabled || !game->state.can_hold_piece) 
        return false;

    PieceType current_type = game->current_piece.type;
    
    if (!game->state.has_hold_piece) {
        // 第一次 Hold：当前放入 Hold，从 Next 拿新的
        game->state.hold_piece.type = current_type;
        game->state.hold_piece.rotation = 0;
        game->state.has_hold_piece = true;
        
        // 从 Bag 取新方块，传入 RNG 状态
        PieceType next = previews_next(&game->state.previews, 
                                       bag_next(&game->state.bag, &game->state.rng_state));
        piece_init(&game->current_piece, next);
    } else {
        // 交换
        PieceType hold_type = game->state.hold_piece.type;
        game->state.hold_piece.type = current_type;
        game->state.hold_piece.rotation = 0;
        
        piece_init(&game->current_piece, hold_type);
    }

    // Hold 之后重置状态
    game->state.can_hold_piece = false;
    game->state.is_last_rotate = 0;

    // 检查生成位置是否死亡 (Block out)
    if (board_piece_overlaps(&game->board, &game->current_piece)) {
        game->is_game_over = true;
        return false;
    }
    return true;
}

// ============================================================================
// Logic: Locking & Clearing
// ============================================================================

// 判断方块是否完全在可见区域上方 (Top Out 判定的一种)
static bool is_completely_top_out(const Piece* p) {
    uint16_t mask = piece_get_mask(p);
    for (int r = 0; r < 4; r++) {
        // Check if row has blocks
        if ((mask >> ((3 - r) * 4)) & 0xF) {
            if ((p->pos[1] - r) < GAME_VISIBLE_HEIGHT) return false; // 有一部分进入了可见区
        }
    }
    return true; // 所有部分都在可见区之上
}

// 快速计算当前方块落地后会消除几行 (不修改 Board)
static int count_cleared_lines(const Board* board, const Piece* p) {
    // 这是一个只读预测，为了性能，我们不拷贝整个 Board
    // 只检查 p 影响的那 4 行
    int lines = 0;
    uint16_t mask = piece_get_mask(p);

    for (int r = 0; r < 4; r++) {
        uint16_t row_bits = (mask >> ((3 - r) * 4)) & 0xF;
        if (!row_bits) continue;

        int y = p->pos[1] - r;
        if (y < 0 || y >= BOARD_HEIGHT) continue;

        // 构造这一行如果放入后的样子
        uint16_t row_state = board->rows[y];
        
        if (p->pos[0] >= 0) row_state |= (row_bits << p->pos[0]);
        else           row_state |= (row_bits >> (-p->pos[0]));

        // 检查是否满 (0x3FF)
        if ((row_state & BOARD_ROW_MASK) == BOARD_ROW_MASK) {
            lines++;
        }
    }
    return lines;
}

bool game_next_step(Game* game) {
    if (game->is_game_over) return true;

    // 1. 锁定方块 (利用 game.h 中的内联优化版)
    place_piece(&game->board, &game->current_piece);

    // 2. 检查 Top Out (锁定完全在屏幕外)
    if (is_completely_top_out(&game->current_piece)) {
        game->is_game_over = true;
        return true;
    }

    // 3. 消行处理 (使用 board.c 的优化版)
    int cleared = board_clear_lines(&game->board);

    // 4. 更新 Combo 和 B2B
    game->state.is_last_clear_line = (cleared > 0);
    if (cleared > 0) {
        game->state.ren++;
    } else {
        game->state.ren = -1;
    }

    // 5. 生成新方块
    PieceType next_type = previews_next(&game->state.previews, 
                                        bag_next(&game->state.bag, &game->state.rng_state));
    piece_init(&game->current_piece, next_type);

    // 6. 重置状态
    game->state.can_hold_piece = true;
    game->state.is_last_rotate = 0;

    // 7. 检查新方块是否重叠 (Block Out)
    if (board_piece_overlaps(&game->board, &game->current_piece)) {
        game->is_game_over = true;
        return true;
    }

    return false;
}

// ============================================================================
// Queries: T-Spin & Attacks
// ============================================================================

int game_get_shadow_height(const Game* game) {
    if (game->is_game_over) return 0;
    
    // 模拟 Hard Drop 距离
    Piece temp = game->current_piece;
    while (!board_piece_overlaps(&game->board, &temp)) {
        temp.pos[1]--;
    }
    // temp.y + 1 是落地位置，current.y - (temp.y + 1) 是距离
    // 也就是 current.y - temp.y - 1
    return game->current_piece.pos[1] - temp.pos[1] - 1;
}

// 检查是否符合 "3-corner" 规则
static bool is_t_triple_corner(const Game* game) {
    const Piece* p = &game->current_piece;
    int count = 0;
    
    for (int i = 0; i < 4; i++) {
        // T_CORNERS 存储相对 (0,0) 的偏移
        int x = p->pos[0] + T_CORNERS[i][0];
        int y = p->pos[1] - T_CORNERS[i][1];
        
        // 检查该点是否被占据 (出界也算占据)
        if (board_get_cell(&game->board, x, y)) {
            count++;
        }
    }
    return count >= 3;
}

// 检查当前方块是否无法移动 (用于判定 T-Spin 是否有效)
static bool is_immobile(const Game* game) {
    Piece temp = game->current_piece;
    // 检查 左, 右, 下 (不需要检查上)
    const int8_t dirs[3][2] = {{-1, 0}, {1, 0}, {0, -1}};
    
    for (int i = 0; i < 3; i++) {
        temp.pos[0] = game->current_piece.pos[0] + dirs[i][0];
        temp.pos[1] = game->current_piece.pos[1] + dirs[i][1];
        if (!board_piece_overlaps(&game->board, &temp)) return false;
    }
    return true;
}

// 完整的 T-Spin Mini 判定逻辑 (遵循标准 SRS Guideline)
static bool is_t_mini(const Game* game, int cleared) {
    // 规则 1: 如果使用了第 5 个踢墙数据 (SRS Test Index 4，对应 is_last_rotate == 5)，
    // 则视为完整 T-Spin，不是 Mini (例如 TST 的最后一步)。
    if (game->state.is_last_rotate == 5) return false;

    // 规则 2: 如果消除了 3 行 (TST)，一定不是 Mini。
    if (cleared == 3) return false;

    // 规则 3: 检查 T 方块背后的两个角 (相对旋转方向)。
    // 如果背后的两个角中有一个是空的，通常视为 Mini。
    // T_RECTS 存储的是 T 字形相对 (0,0) 的两个"肩膀"位置，也就是背后的位置。
    // (注意：这里复用了 game.c 前文定义的 T_RECTS，它实际上存的是需要检查的位置)
    
    int empty_shoulder_count = 0;
    const Piece* p = &game->current_piece;
    
    // T_RECTS[rotation][0] 和 [1] 是 T 方块背后的两个格子的偏移量
    for (int i = 0; i < 2; i++) {
        int tx = p->pos[0] + T_RECTS[p->rotation][i][0];
        int ty = p->pos[1] - T_RECTS[p->rotation][i][1];
        
        // 检查是否为空 (注意 board_get_cell 返回 1 表示占据/出界，0 表示空)
        if (board_get_cell(&game->board, tx, ty) == 0) {
            empty_shoulder_count++;
        }
    }

    // 如果背后有一个角是空的，这通常是一个 T-Spin Mini (比如 T-Spin Single Mini)
    // 但如果两个都是空的，那就不是 3-corner 了 (逻辑上已经被 is_t_triple_corner 排除)
    return (empty_shoulder_count > 0);
}

AttackType game_get_attack_type(const Game* game) {
    // 1. 计算消除行数 (不修改 Board)
    int cleared = count_cleared_lines(&game->board, &game->current_piece);
    
    // 如果没有消除，直接返回
    if (cleared == 0) return ATK_NONE;

    // 2. 判定是否为 Spin (旋转进入且无法移动)
    // 必须发生过旋转 (is_last_rotate > 0) 且当前位置被卡住 (Immobile)
    bool is_spin = (game->state.is_last_rotate > 0) && is_immobile(game);

    // 3. T-Spin 特殊判定
    if (game->current_piece.type == PIECE_T && is_spin) {
        // 必须满足 3-corner 规则 (3个角被占据)
        if (is_t_triple_corner(game)) {
            // 判定是否为 Mini T-Spin
            if (is_t_mini(game, cleared)) {
                switch (cleared) {
                    case 1: return ATK_TSMS; // T-Spin Mini Single
                    case 2: return ATK_TSMD; // T-Spin Mini Double (常见于 B2B)
                    default: return ATK_TSS; // 兜底，理论上不会走到这里
                }
            } else {
                // 完整 T-Spin
                switch (cleared) {
                    case 1: return ATK_TSS; // T-Spin Single
                    case 2: return ATK_TSD; // T-Spin Double
                    case 3: return ATK_TST; // T-Spin Triple
                    default: return ATK_ERROR;
                }
            }
        }
    } 
    // 4. All-Spin 判定 (其他方块的 Spin)
    else if (is_spin) {
        switch (game->current_piece.type) {
            case PIECE_I:
                switch (cleared) {
                    case 1: return ATK_ISS;
                    case 2: return ATK_ISD;
                    case 3: return ATK_IST;
                    default: break; // I方块理论上消不了4行还能 Spin (4行通常是 Tetris)
                }
                break;
            case PIECE_O:
                // O-Spin 极其罕见，通常认为是 O-Spin Single/Double
                switch (cleared) {
                    case 1: return ATK_OSS;
                    case 2: return ATK_OSD;
                    default: break;
                }
                break;
            case PIECE_S:
                switch (cleared) {
                    case 1: return ATK_SSS;
                    case 2: return ATK_SSD;
                    case 3: return ATK_SST;
                    default: break;
                }
                break;
            case PIECE_Z:
                switch (cleared) {
                    case 1: return ATK_ZSS;
                    case 2: return ATK_ZSD;
                    case 3: return ATK_ZST;
                    default: break;
                }
                break;
            case PIECE_J:
                switch (cleared) {
                    case 1: return ATK_JSS;
                    case 2: return ATK_JSD;
                    case 3: return ATK_JST;
                    default: break;
                }
                break;
            case PIECE_L:
                switch (cleared) {
                    case 1: return ATK_LSS;
                    case 2: return ATK_LSD;
                    case 3: return ATK_LST;
                    default: break;
                }
                break;
            default: break;
        }
    }

    // 5. 普通消除判定 (非 Spin 或 Spin 条件未满足)
    switch (cleared) {
        case 1: return ATK_SINGLE;
        case 2: return ATK_DOUBLE;
        case 3: return ATK_TRIPLE;
        case 4: return ATK_TETRIS;
        default: return ATK_NONE; // 理论上不可能
    }
}

bool game_is_perfect_clear(const Game* game) {
    // 模拟消除
    Board temp = game->board;
    place_piece(&temp, &game->current_piece);
    board_clear_lines(&temp);
    
    // 检查棋盘是否全空
    // 优化：int 步进检查
    for (int y = 0; y < BOARD_HEIGHT; y++) {
        if (!board_is_row_empty(&temp, y)) return false;
    }
    return true;
}