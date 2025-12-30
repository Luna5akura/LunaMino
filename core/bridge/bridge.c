// bridge.c
#include "bridge.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
// 引入 Raylib 和 UI 定义
#include <raylib.h>
#include "../tetris/tetris_ui/tetris_ui.h"
// ============================================================================
// 内部辅助：预计算消除行数
// ============================================================================
// 这是一个只读预测，不修改棋盘
static int predict_cleared_lines(const Board* board, const Piece* p) {
    int lines = 0;
    uint16_t mask = piece_get_mask(p);
    // 遍历方块的 4 行
    for (int r = 0; r < 4; r++) {
        // 提取方块当前行的数据 (4 bits)
        uint16_t row_bits = (mask >> ((3 - r) * 4)) & 0xF;
        if (!row_bits) continue; // 空行跳过
        int y = p->pos[1] - r;
        // 边界安全检查
        if (y < 0 || y >= BOARD_HEIGHT) continue;
        // 构造这一行如果放入后的样子
        uint16_t row_state = board->rows[y];
       
        if (p->pos[0] >= 0) {
            row_state |= (row_bits << p->pos[0]);
        } else {
            row_state |= (row_bits >> (-p->pos[0]));
        }
        // 检查是否满 (0x3FF = 10 bits set)
        if ((row_state & BOARD_ROW_MASK) == BOARD_ROW_MASK) {
            lines++;
        }
    }
    return lines;
}
// 辅助函数：判断方块是否不可移动 (Immobile)
// 用于在瞬移操作下辅助 T-Spin 判定
static bool is_immobile(const Game* game, const Piece* p) {
    // 检查 左, 右, 上
    const int8_t dirs[3][2] = {{-1, 0}, {1, 0}, {0, 1}};
    // 这里我们需要临时的非 const 指针来进行重叠测试，但不修改数据
    Board* board = (Board*)&game->board;
    Piece temp = *p;
    for (int i = 0; i < 3; i++) {
        temp.pos[0] = p->pos[0] + dirs[i][0];
        temp.pos[1] = p->pos[1] + dirs[i][1];
        if (!board_piece_overlaps(board, &temp)) return false; // 只要能往一个方向动，就不算 immobile
    }
    return true;
}
// ============================================================================
// BFS Search & Legal Moves
// ============================================================================
typedef struct {
    int8_t x;
    int8_t y;
    int8_t rotation;
} BFSNode;
// 队列容量：4朝向 * 10列 * 24行，4096 足够
#define BFS_QUEUE_CAPACITY 4096
// 访问表尺寸映射
#define VISIT_W 16 // x范围 [-3, 12] -> [0, 15]
#define VISIT_H 30 // y范围 [0, 29]
#define VISIT_X_OFFSET 3
typedef struct {
    BFSNode data[BFS_QUEUE_CAPACITY];
    int head;
    int tail;
} BFSQueue;
// 静态访问表，避免栈溢出 (非线程安全)
static bool visited[4][VISIT_W][VISIT_H];
static inline void bfs_push(BFSQueue* q, int8_t x, int8_t y, int8_t rotation) {
    if (q->tail >= BFS_QUEUE_CAPACITY) return;
    q->data[q->tail].x = x;
    q->data[q->tail].y = y;
    q->data[q->tail].rotation = rotation;
    q->tail++;
}
static inline BFSNode bfs_pop(BFSQueue* q) {
    return q->data[q->head++];
}
static inline bool bfs_empty(const BFSQueue* q) {
    return q->head == q->tail;
}
static inline bool is_visited(int8_t rotation, int8_t x, int8_t y) {
    int vx = x + VISIT_X_OFFSET;
    if (vx < 0 || vx >= VISIT_W || y < 0 || y >= VISIT_H) return true;
    return visited[rotation][vx][y];
}
static inline void mark_visited(int8_t rotation, int8_t x, int8_t y) {
    int vx = x + VISIT_X_OFFSET;
    if (vx >= 0 && vx < VISIT_W && y >= 0 && y < VISIT_H) {
        visited[rotation][vx][y] = true;
    }
}
static void run_bfs_for_current_piece(const Game* base_game, bool use_hold, LegalMoves* out_moves) {
    // 1. 复制游戏状态用于模拟
    Game sim_game = *base_game;
    Piece* p = &sim_game.current_piece;
    // 2. Block Out 检查
    if (board_piece_overlaps(&sim_game.board, p)) return;
    // 3. 初始化 BFS
    memset(visited, 0, sizeof(visited));
    BFSQueue q = { .head = 0, .tail = 0 };
    mark_visited(p->rotation, p->pos[0], p->pos[1]);
    bfs_push(&q, p->pos[0], p->pos[1], p->rotation);
    // 4. 搜索
    while (!bfs_empty(&q)) {
        BFSNode node = bfs_pop(&q);
        // 恢复方块状态
        p->pos[0] = node.x;
        p->pos[1] = node.y;
        p->rotation = node.rotation;
        // --- 检查落地 (Lockable) ---
        p->pos[1]--;
        if (board_piece_overlaps(&sim_game.board, p)) {
            // 下方有阻挡，说明当前位置有效
            if (out_moves->count < MAX_LEGAL_MOVES) {
                MacroAction* m = &out_moves->moves[out_moves->count++];
                m->x = node.x;
                m->y = node.y;
                m->rotation = node.rotation;
                m->use_hold = use_hold;
                m->landing_height = node.y;
            }
        }
        p->pos[1]++; // 还原
        // --- 扩展邻居 ---
        // 保存当前状态，尝试动作后恢复，避免 Game 结构体拷贝
        Piece saved_state = *p;
        #define TRY_ACTION(action_call) \
            if (action_call) { \
                if (!is_visited(p->rotation, p->pos[0], p->pos[1])) { \
                    mark_visited(p->rotation, p->pos[0], p->pos[1]); \
                    bfs_push(&q, p->pos[0], p->pos[1], p->rotation); \
                } \
                *p = saved_state; \
            }
        TRY_ACTION(game_try_move(&sim_game, MOVE_LEFT));
        TRY_ACTION(game_try_move(&sim_game, MOVE_RIGHT));
        TRY_ACTION(game_try_move(&sim_game, MOVE_DOWN));
        TRY_ACTION(game_try_rotate(&sim_game, ROTATE_CW));
        TRY_ACTION(game_try_rotate(&sim_game, ROTATE_CCW));
        #undef TRY_ACTION
    }
}
void ai_get_legal_moves(const Tetris* tetris, LegalMoves* out_moves) {
    out_moves->count = 0;
    // 1. 搜索当前方块
    run_bfs_for_current_piece(&tetris->game, false, out_moves);
    // 2. 搜索 Hold 后方块
    if (tetris->game.config.is_hold_enabled && tetris->game.state.can_hold_piece) {
        Game hold_game = tetris->game;
       
        PieceType current_type = hold_game.current_piece.type;
        PieceType next_type;
        if (!hold_game.state.has_hold_piece) {
            // 当前进 Hold，Next 变 Current
            hold_game.state.hold_piece.type = current_type;
            hold_game.state.has_hold_piece = true;
            next_type = previews_peek(&hold_game.state.previews, 0);
        } else {
            // 交换
            next_type = hold_game.state.hold_piece.type;
            hold_game.state.hold_piece.type = current_type;
        }
        piece_init(&hold_game.current_piece, next_type);
        run_bfs_for_current_piece(&hold_game, true, out_moves);
    }
}
// ============================================================================
// Lifecycle Management
// ============================================================================
Tetris* create_tetris(int seed) {
    if (seed == 0) seed = (int)time(NULL);
    GameConfig config = {
        .preview_count = 5,
        .is_hold_enabled = true,
        .seed = (unsigned int)seed
    };
    return tetris_init(&config);
}
void destroy_tetris(Tetris* tetris) {
    if (!tetris) return;
    tetris_free(tetris);
}
Tetris* clone_tetris(const Tetris* tetris) {
    return tetris_copy(tetris);
}
void ai_reset_game(Tetris* tetris, int seed) {
    if (!tetris) return;
    if (seed == 0) seed = (int)time(NULL);
   
    GameConfig config = {
        .preview_count = 5,
        .is_hold_enabled = true,
        .seed = (unsigned int)seed
    };
   
    Tetris* temp = tetris_init(&config);
    if (temp) {
        *tetris = *temp;
        free(temp); // 只释放容器，内容已拷贝
    }
}
// ============================================================================
// State Observation
// ============================================================================
void ai_get_state(const Tetris* tetris, int* board_buffer, int* queue_buffer, int* hold_buffer, int* meta_buffer) {
    const Board* board = &tetris->game.board;
   
    // Board: 10x20
    int idx = 0;
    for (int y = 0; y < 20; y++) {
        for (int x = 0; x < 10; x++) {
            board_buffer[idx++] = board_get_cell(board, x, y);
        }
    }
    // Queue: 5
    const Previews* previews = &tetris->game.state.previews;
    for (int i = 0; i < 5; i++) {
        queue_buffer[i] = (int)previews_peek(previews, i);
    }
    // Hold: 1
    if (tetris->game.state.has_hold_piece) {
        hold_buffer[0] = (int)tetris->game.state.hold_piece.type;
    } else {
        hold_buffer[0] = -1;
    }
    // Meta: [B2B, Combo, CanHold, PieceType, PendingGarbage]
    meta_buffer[0] = tetris->state.b2b_count;
    meta_buffer[1] = tetris->game.state.ren;
    meta_buffer[2] = tetris->game.state.can_hold_piece ? 1 : 0;
    meta_buffer[3] = (int)tetris->game.current_piece.type;
    meta_buffer[4] = tetris->state.pending_attack;
}
// ============================================================================
// AI Action Step (Optimized)
// ============================================================================
StepResult ai_step(Tetris* tetris, int x, int y, int rotation, int use_hold) {
    StepResult result = {0};
    if (tetris->state.is_game_over) {
        result.is_game_over = true;
        return result;
    }
    // 1. Hold
    if (use_hold) {
        if (tetris->game.state.can_hold_piece) {
            game_hold_piece(&tetris->game);
        }
    }
    Piece* p = &tetris->game.current_piece;
    // 2. 瞬移
    p->pos[0] = (int8_t)x;
    p->pos[1] = (int8_t)y;
    p->rotation = (uint8_t)(rotation & 3);
    // 3. 碰撞检查
    if (board_piece_overlaps(&tetris->game.board, p)) {
        tetris->game.is_game_over = true;
        tetris->state.is_game_over = true;
        result.is_game_over = true;
        return result;
    }
    // 4. 预计算消除行数 (Pre-calculation)
    // 必须在 tetris_update_clear_rows 之前进行，因为 update 后 board 会变
    int predicted_lines = predict_cleared_lines(&tetris->game.board, p);
    // 5. T-Spin 补偿
    tetris->game.state.is_last_rotate = 0;
    if (p->type == PIECE_T && is_immobile(&tetris->game, p)) {
        tetris->game.state.is_last_rotate = 1;
    }
    // 6. 捕获更新前的攻击状态
    int prev_atk_total = tetris->state.atk_count;
    int prev_pending = tetris->state.pending_attack;
    // 7. 执行核心逻辑 (Lock, Clear, Attack Calc, Garbage Cancel, Spawn)
    tetris_update_clear_rows(tetris);
    // 8. 后处理：计算实际发送垃圾量
    int raw_atk_generated = tetris->state.atk_count - prev_atk_total;
    int damage_sent = 0;
    // 逻辑：如果本次有攻击力
    if (raw_atk_generated > 0) {
        if (prev_pending > 0) {
            // 有等待的垃圾，先抵消
            if (raw_atk_generated >= prev_pending) {
                // 攻击力溢出，发送溢出部分
                damage_sent = raw_atk_generated - prev_pending;
                tetris->state.pending_attack = 0;
            } else {
                // 攻击力全部被吸收
                damage_sent = 0;
                tetris->state.pending_attack = prev_pending - raw_atk_generated;
            }
        } else {
            // 无等待垃圾，全部发送
            damage_sent = raw_atk_generated;
        }
    }
    // 9. 填充结果
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
        draw_content(tetris, ai_ui_config);
    }
}
void ai_close_visualization() {
    if (ai_ui_config) {
        CloseWindow();
        free_ui_config(ai_ui_config);
        ai_ui_config = NULL;
    }
}