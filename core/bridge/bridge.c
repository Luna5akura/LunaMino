// core/bridge/bridge.c

#include "bridge.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <raylib.h>
#include "../tetris/tetris_ui/tetris_ui.h"

// 维度定义 (需与 Python config 保持一致)
#define GRID_WIDTH_X 12
#define GRID_HEIGHT_Y 24
#define OFFSET_X 2
#define STRIDE_X GRID_HEIGHT_Y                    // 24
#define STRIDE_ROT (GRID_WIDTH_X * GRID_HEIGHT_Y) // 288
#define STRIDE_HOLD (STRIDE_ROT * 4)              // 1152

// ============================================================================
// 内部辅助：ID 计算与消除预测
// ============================================================================

static inline int16_t calculate_action_id(int8_t x, int8_t landing_y, int8_t rot, bool hold) {
    // 严格限制范围，防止 ID 越界导致 Python 端 CrossEntropyLoss 崩溃
    int16_t x_idx = x + OFFSET_X; 
    if (x_idx < 0) x_idx = 0;
    if (x_idx >= GRID_WIDTH_X) x_idx = GRID_WIDTH_X - 1;
    
    int16_t y_idx = landing_y;
    if (y_idx < 0) y_idx = 0;
    if (y_idx >= GRID_HEIGHT_Y) y_idx = GRID_HEIGHT_Y - 1;

    int16_t rot_idx = rot & 3;
    int16_t hold_idx = hold ? 1 : 0;

    return (int16_t)(
        (hold_idx * STRIDE_HOLD) +
        (rot_idx * STRIDE_ROT) +
        (x_idx * STRIDE_X) +
        y_idx
    );
}

static inline int predict_cleared_lines(const Board* board, const Piece* p) {
    int lines = 0;
    uint16_t mask = piece_get_mask(p);
    for (int r = 0; r < 4; r++) {
        uint16_t row_bits = (mask >> ((3 - r) * 4)) & 0xF;
        if (!row_bits) continue;
        int y = p->pos[1] - r;
        if (y < 0 || y >= BOARD_HEIGHT) continue;
        
        uint16_t row_state = board->rows[y];
        if (p->pos[0] >= 0) {
            row_state |= (row_bits << p->pos[0]);
        } else {
            row_state |= (row_bits >> (-p->pos[0]));
        }
        
        if ((row_state & BOARD_ROW_MASK) == BOARD_ROW_MASK) {
            lines++;
        }
    }
    return lines;
}

// 辅助：判断方块是否不可移动 (用于 T-Spin)
static bool is_immobile(const Game* game, const Piece* p) {
    const int8_t dirs[3][2] = {{-1, 0}, {1, 0}, {0, -1}};
    Board* board = (Board*)&game->board;
    Piece temp = *p;
    for (int i = 0; i < 3; i++) {
        temp.pos[0] = p->pos[0] + dirs[i][0];
        temp.pos[1] = p->pos[1] + dirs[i][1];
        if (!board_piece_overlaps(board, &temp)) return false;
    }
    return true;
}

// ============================================================================
// BFS Search (Flattened Visited Array & No Piece Copy)
// ============================================================================

typedef struct {
    int8_t x;
    int8_t y;
    int8_t rotation;
} BFSNode;

#define BFS_QUEUE_CAPACITY 4096
#define VISIT_W 16 // x范围 [-3, 12] -> [0, 15]
#define VISIT_H 30 // y范围 [0, 29]
#define VISIT_SIZE (4 * VISIT_W * VISIT_H)
#define VISIT_X_OFFSET 3

static bool visited[VISIT_SIZE]; 

typedef struct {
    BFSNode data[BFS_QUEUE_CAPACITY];
    int head;
    int tail;
} BFSQueue;

static inline int visit_index(int8_t rotation, int8_t x, int8_t y) {
    int vx = x + VISIT_X_OFFSET;
    if (vx < 0 || vx >= VISIT_W || y < 0 || y >= VISIT_H) return -1;
    return rotation * (VISIT_W * VISIT_H) + vx * VISIT_H + y;
}

static inline bool is_visited(int8_t rotation, int8_t x, int8_t y) {
    int idx = visit_index(rotation, x, y);
    return idx < 0 ? true : visited[idx];
}

static inline void mark_visited(int8_t rotation, int8_t x, int8_t y) {
    int idx = visit_index(rotation, x, y);
    if (idx >= 0) visited[idx] = true;
}

static inline void bfs_push(BFSQueue* q, int8_t x, int8_t y, int8_t rotation) {
    if (q->tail >= BFS_QUEUE_CAPACITY) return;
    q->data[q->tail].x = x;
    q->data[q->tail].y = y;
    q->data[q->tail].rotation = rotation;
    q->tail++;
}

static int compare_moves(const void* a, const void* b) {
    const MacroAction* ma = (const MacroAction*)a;
    const MacroAction* mb = (const MacroAction*)b;
    // 排序优先级: Hold > Rotation > X > Y
    if (ma->use_hold != mb->use_hold) return ma->use_hold - mb->use_hold;
    if (ma->rotation != mb->rotation) return ma->rotation - mb->rotation;
    if (ma->x != mb->x) return ma->x - mb->x;
    return ma->landing_height - mb->landing_height;
}



static void ai_simulate_next_board(const Board* base_board, const Piece* p, uint8_t* out_buf) {
    // 1. 拷贝棋盘行数据 (假设最大高度 40，为了安全使用 BOARD_HEIGHT)
    // 注意：我们需要包含 game.h 或 board.h 才能拿到 BOARD_HEIGHT。
    // 如果没有定义，bridge.c 前面定义的 GRID_HEIGHT_Y 是 24，通常够用。
    #ifndef BOARD_HEIGHT
    #define BOARD_HEIGHT 24 
    #endif
    
    uint16_t rows[BOARD_HEIGHT];
    memcpy(rows, base_board->rows, sizeof(uint16_t) * BOARD_HEIGHT);

    // 2. 锁定方块 (模拟 place_piece)
    uint16_t mask = piece_get_mask(p);
    for (int r = 0; r < 4; r++) {
        uint16_t row_bits = (mask >> ((3 - r) * 4)) & 0xF;
        if (!row_bits) continue;
        int y = p->pos[1] - r;
        if (y < 0 || y >= BOARD_HEIGHT) continue;

        if (p->pos[0] >= 0) {
            rows[y] |= (row_bits << p->pos[0]);
        } else {
            rows[y] |= (row_bits >> (-p->pos[0]));
        }
    }

    // 3. 模拟消行 (模拟 board_clear_lines)
    // 我们不需要真的修改 rows 数组的大小，只需要把非满行紧凑地移到底部
    int write_y = 0;
    uint16_t cleared_rows[BOARD_HEIGHT]; // 临时存放清理后的结果
    memset(cleared_rows, 0, sizeof(cleared_rows));

    for (int read_y = 0; read_y < BOARD_HEIGHT; read_y++) {
        // 假设 BOARD_ROW_MASK 是 0x3FF (10位)
        #ifndef BOARD_ROW_MASK
        #define BOARD_ROW_MASK 0x3FF
        #endif

        if ((rows[read_y] & BOARD_ROW_MASK) != BOARD_ROW_MASK) {
            // 这行没满，保留
            cleared_rows[write_y++] = rows[read_y];
        }
        // 如果满了，write_y 不增加，相当于删除了这行
    }
    // 上方自动补 0 (memset 已做)

    // 4. 序列化 (Flatten 20x10) - 格式必须与 ai_get_state 保持完全一致
    // ai_get_state 是从 y=0 到 y=19
    int idx = 0;
    for (int y = 0; y < 20; y++) {
        uint16_t row = cleared_rows[y];
        for (int x = 0; x < 10; x++) {
            out_buf[idx++] = (row & (1 << x)) ? 1 : 0;
        }
    }
}

// ============================================================================
// BFS Search (Modified to support Preview)
// ============================================================================

// [MODIFIED] 增加 preview_buffer 参数
static void run_bfs_for_current_piece(const Game* base_game, bool use_hold, LegalMoves* out_moves, uint8_t* preview_buffer) {
    Game sim_game = *base_game;
    Piece* p = &sim_game.current_piece;
    
    if (board_piece_overlaps(&sim_game.board, p)) return;

    memset(visited, 0, sizeof(visited));
    
    BFSQueue q = { .head = 0, .tail = 0 };
    mark_visited(p->rotation, p->pos[0], p->pos[1]);
    bfs_push(&q, p->pos[0], p->pos[1], p->rotation);

    while (q.head != q.tail) {
        BFSNode node = q.data[q.head++];

        p->pos[0] = node.x;
        p->pos[1] = node.y;
        p->rotation = node.rotation;

        // Check Landing
        p->pos[1]--; 
        if (board_piece_overlaps(&sim_game.board, p)) {
            // Found a valid landing spot
            if (out_moves->count < MAX_LEGAL_MOVES) {
                int idx = out_moves->count; // 当前索引
                
                MacroAction* m = &out_moves->moves[idx];
                m->x = node.x;
                m->y = node.y;
                m->rotation = node.rotation;
                m->use_hold = use_hold;
                m->landing_height = node.y;
                m->padding = 0;
                m->id = calculate_action_id(node.x, node.y, node.rotation, use_hold);

                // [NEW] Generate Preview
                // 只有当提供了缓冲区时才计算，避免影响正常游戏逻辑的性能
                if (preview_buffer) {
                    // 恢复 p 到正确的落地位置 (因为上面 p->pos[1]-- 导致它是重叠状态)
                    p->pos[1]++; 
                    // 计算偏移量: 每个动作 200 字节
                    ai_simulate_next_board(&sim_game.board, p, preview_buffer + (idx * 200));
                    // 再次 -1 以恢复到循环需要的状态 (虽然这里其实不需要了，但为了逻辑严谨)
                    p->pos[1]--; 
                }

                out_moves->count++;
            }
        }
        p->pos[1]++; // Restore

        // Expand Neighbors... (保持原样)
        int8_t saved_x = p->pos[0];
        int8_t saved_y = p->pos[1];
        uint8_t saved_rot = p->rotation;

        #define TRY_ACTION(action_call) \
            if (action_call) { \
                if (!is_visited(p->rotation, p->pos[0], p->pos[1])) { \
                    mark_visited(p->rotation, p->pos[0], p->pos[1]); \
                    bfs_push(&q, p->pos[0], p->pos[1], p->rotation); \
                } \
                p->pos[0] = saved_x; \
                p->pos[1] = saved_y; \
                p->rotation = saved_rot; \
            }

        TRY_ACTION(game_try_move(&sim_game, MOVE_LEFT));
        TRY_ACTION(game_try_move(&sim_game, MOVE_RIGHT));
        TRY_ACTION(game_try_move(&sim_game, MOVE_DOWN));
        TRY_ACTION(game_try_rotate(&sim_game, ROTATE_CW));
        TRY_ACTION(game_try_rotate(&sim_game, ROTATE_CCW));
        #undef TRY_ACTION
    }
}

// [MODIFIED] 原有的函数，传递 NULL
void ai_get_legal_moves(const Tetris* tetris, LegalMoves* out_moves) {
    out_moves->count = 0;
    run_bfs_for_current_piece(&tetris->game, false, out_moves, NULL);

    if (tetris->game.config.is_hold_enabled && tetris->game.state.can_hold_piece) {
        Game hold_game = tetris->game;
        // ... Hold Logic (保持原样) ...
        PieceType current_type = hold_game.current_piece.type;
        PieceType next_type;
        if (!hold_game.state.has_hold_piece) {
            hold_game.state.hold_piece.type = current_type;
            hold_game.state.has_hold_piece = true;
            next_type = previews_peek(&hold_game.state.previews, 0);
        } else {
            next_type = hold_game.state.hold_piece.type;
            hold_game.state.hold_piece.type = current_type;
        }
        piece_init(&hold_game.current_piece, next_type);
        // ... End Hold Logic ...
        
        run_bfs_for_current_piece(&hold_game, true, out_moves, NULL);
    }

    // 原始函数保留排序，保证兼容性
    qsort(out_moves->moves, out_moves->count, sizeof(MacroAction), compare_moves);
}

// ============================================================================
// [NEW] API for Training (No Sort, With Previews)
// ============================================================================

// states_buffer 必须在 Python 端预分配：大小 = MAX_LEGAL_MOVES (256) * 200 bytes
void ai_get_legal_moves_and_previews(const Tetris* tetris, LegalMoves* out_moves, uint8_t* states_buffer) {
    out_moves->count = 0;

    // 1. Current Piece
    run_bfs_for_current_piece(&tetris->game, false, out_moves, states_buffer);

    // 2. Hold Piece
    if (tetris->game.config.is_hold_enabled && tetris->game.state.can_hold_piece) {
        Game hold_game = tetris->game;
        // ... 复用上面的 Hold Logic 代码 ...
        PieceType current_type = hold_game.current_piece.type;
        PieceType next_type;
        if (!hold_game.state.has_hold_piece) {
            hold_game.state.hold_piece.type = current_type;
            hold_game.state.has_hold_piece = true;
            next_type = previews_peek(&hold_game.state.previews, 0);
        } else {
            next_type = hold_game.state.hold_piece.type;
            hold_game.state.hold_piece.type = current_type;
        }
        piece_init(&hold_game.current_piece, next_type);
        
        run_bfs_for_current_piece(&hold_game, true, out_moves, states_buffer);
    }

    // ！！！关键！！！
    // 这里绝对不能调用 qsort。
    // 因为 states_buffer 是并行填充的，如果 moves 数组排序了，buffer 里的数据就对应不上了。
    // AI 训练不关心顺序，只关心 (Action, NextState) 对。
}
// ============================================================================
// Lifecycle & State
// ============================================================================

Tetris* create_tetris(int seed) {
    if (seed == 0) seed = (int)time(NULL);
    GameConfig config = { .preview_count = 5, .is_hold_enabled = true, .seed = (unsigned int)seed };
    return tetris_init(&config);
}

void destroy_tetris(Tetris* tetris) {
    if (tetris) tetris_free(tetris);
}

Tetris* clone_tetris(const Tetris* tetris) {
    return tetris_copy(tetris);
}

void ai_reset_game(Tetris* tetris, int seed) {
    if (!tetris) return;
    if (seed == 0) seed = (int)time(NULL);
    GameConfig config = { .preview_count = 5, .is_hold_enabled = true, .seed = (unsigned int)seed };
    Tetris* temp = tetris_init(&config);
    if (temp) {
        *tetris = *temp; // Deep copy structure container
        free(temp); 
    }
}

void ai_get_state(const Tetris* tetris, uint8_t* board_buffer, float* ctx_buffer) {
    const Board* board = &tetris->game.board;
    int idx = 0;
    // Flatten 20x10 to 200 bytes
    for (int y = 0; y < 20; y++) {
        uint16_t row = board->rows[y];
        for (int x = 0; x < 10; x++) {
            board_buffer[idx++] = (row & (1 << x)) ? 1 : 0;
        }
    }

    // Context
    ctx_buffer[0] = (float)tetris->game.current_piece.type;
    ctx_buffer[1] = tetris->game.state.has_hold_piece ? (float)tetris->game.state.hold_piece.type : -1.0f;
    for (int i = 0; i < 5; i++) {
        ctx_buffer[2 + i] = (float)previews_peek(&tetris->game.state.previews, i);
    }
    ctx_buffer[7] = (float)tetris->state.b2b_count;
    ctx_buffer[8] = (float)tetris->game.state.ren;
    ctx_buffer[9] = tetris->game.state.can_hold_piece ? 1.0f : 0.0f;
    ctx_buffer[10] = (float)tetris->state.pending_attack;
}

// ============================================================================
// AI Action Execution
// ============================================================================

StepResult ai_step(Tetris* tetris, int x, int y, int rotation, int use_hold) {
    StepResult result = {0};
    
    if (tetris->state.is_game_over) {
        result.is_game_over = true;
        return result;
    }

    // 1. Hold logic
    if (use_hold) {
        if (tetris->game.state.can_hold_piece) {
            game_hold_piece(&tetris->game);
        }
    }

    Piece* p = &tetris->game.current_piece;
    
    // 2. Set position (Teleport)
    p->pos[0] = (int8_t)x;
    p->pos[1] = (int8_t)y; // 这里 y 应该是 BFS 算出来的 landing_height (valid position)
    p->rotation = (uint8_t)(rotation & 3);

    // 3. Check validity
    // 如果直接放上去就重叠了，说明位置非法或者 AI 预测错误。
    // 为了容错，如果当前位置重叠，尝试向上移动一格（极少数情况）
    if (board_piece_overlaps(&tetris->game.board, p)) {
        // 重大错误：尝试原地锁定会导致崩溃，直接判定结束或不做任何事
        // 但通常 BFS 保证了这里是安全的。如果重叠，可能是状态不同步。
        // 这里我们不做额外处理，让 game_lock_piece 可能会截断或者逻辑继续
        // 实际上如果 overlaps，game_lock_piece 的行为未定义（取决于实现），
        // 最好在这里直接返回 Game Over 防止内存破坏。
        tetris->game.is_game_over = true;
        tetris->state.is_game_over = true;
        result.is_game_over = true;
        return result;
    }

    // 4. Predict Lines (Read-only)
    int predicted_lines = predict_cleared_lines(&tetris->game.board, p);

    // 5. T-Spin Detection Helper
    tetris->game.state.is_last_rotate = 0; 
    if (p->type == PIECE_T && is_immobile(&tetris->game, p)) {
        // 瞬移无法检测 kick，所以如果卡住了，我们假设它是通过旋转进去的（简化逻辑）
        tetris->game.state.is_last_rotate = 1; 
    }

    // 6. Capture Attack State
    int prev_atk_total = tetris->state.atk_count;
    int prev_pending = tetris->state.pending_attack;

    tetris_update_clear_rows(tetris); // 消除行、计算分数、生成新方块

    // 8. Calculate Damage
    int raw_atk_generated = tetris->state.atk_count - prev_atk_total;
    int damage_sent = 0;

    if (raw_atk_generated > 0) {
        if (prev_pending > 0) {
            if (raw_atk_generated >= prev_pending) {
                damage_sent = raw_atk_generated - prev_pending;
                // tetris_update_clear_rows 内部可能已经处理了 pending 扣除，
                // 但我们需要返回这步操作“原本”发出了多少攻击。
                // 这里的 damage_sent 是溢出攻击（网战中的实际攻击）
            } else {
                damage_sent = 0;
            }
        } else {
            damage_sent = raw_atk_generated;
        }
    }

    // 9. Fill Result
    result.lines_cleared = predicted_lines;
    result.damage_sent = damage_sent;
    result.attack_type = tetris->state.attack_type;
    result.is_game_over = tetris->state.is_game_over;
    result.b2b_count = tetris->state.b2b_count;
    result.combo_count = tetris->game.state.ren;

    return result;
}

void ai_receive_garbage(Tetris* tetris, int lines) {
    if (tetris && lines > 0) {
        tetris_receive_attack(tetris, lines);
    }
}

// ============================================================================
// Visualization
// ============================================================================
static UIConfig* ai_ui_config = NULL;

void ai_enable_visualization(Tetris* tetris) {
    if (ai_ui_config == NULL) {
        ai_ui_config = init_ui_config();
        if (ai_ui_config) {
            init_window(ai_ui_config);
            SetTargetFPS(60);
        }
    }
}

void ai_render(Tetris* tetris) {
    if (ai_ui_config && !WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(BLACK);
        draw_content(tetris, ai_ui_config);
        EndDrawing();
    }
}

void ai_close_visualization() {
    if (ai_ui_config) {
        CloseWindow();
        free_ui_config(ai_ui_config);
        ai_ui_config = NULL;
    }
}