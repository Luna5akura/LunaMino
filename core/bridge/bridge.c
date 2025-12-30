// core/tetris/bridge.c
#include "bridge.h"
#include <raylib.h>
#include "../tetris/tetris_ui/tetris_ui.h" // for vis
#include <string.h>
#include <stdlib.h>
#include <time.h>
// BFS for legal moves
typedef struct {
    int x; int y; int rotation;
} BFSNode;
typedef struct {
    BFSNode data[2048];
    int head, tail;
} BFSQueue;
static void bfs_push(BFSQueue* q, int x, int y, int rotation) {
    if (q->tail >= 2048) return;
    if (x < 0 || x >= 10 || y < 0 || y >= 40) return;
    q->data[q->tail].x = x;
    q->data[q->tail].y = y;
    q->data[q->tail].rotation = rotation;
    q->tail++;
}
static BFSNode bfs_pop(BFSQueue* q) {
    return q->data[q->head++];
}
static Bool bfs_empty(const BFSQueue* q) {
    return q->head == q->tail;
}
static void run_bfs_for_current_piece(Game* game, int use_hold, LegalMoves* out_moves) {
    Piece* p = &game->current_piece;
    int count = out_moves->count;
    Bool visited[4 * 10 * 40];
    memset(visited, 0, sizeof(visited));
    BFSQueue q = { .head = 0, .tail = 0 };
    if (p->x >= 0 && p->x < 10 && p->y >= 0 && p->y < 40) {
        int index = p->rotation * 10 * 40 + p->x * 40 + p->y;
        visited[index] = true;
        bfs_push(&q, p->x, p->y, p->rotation);
    }
    int ox = p->x, oy = p->y, orot = p->rotation;
    while (!bfs_empty(&q)) {
        BFSNode node = bfs_pop(&q);
        p->x = node.x;
        p->y = node.y;
        p->rotation = node.rotation;
        // check if at bottom
        int save_y = p->y;
        Bool can_down = game_try_move(game, MOVE_DOWN);
        if (can_down) {
            p->y = save_y; // undo
        } else {
            // at bottom, add the move
            if (count < MAX_LEGAL_MOVES) {
                out_moves->moves[count].x = node.x;
                out_moves->moves[count].y = node.y;
                out_moves->moves[count].rotation = node.rotation;
                out_moves->moves[count].use_hold = use_hold;
                out_moves->moves[count].landing_height = game_get_shadow_height(game);
                count++;
            }
        }
        // actions
        for (int action = 0; action < 5; action++) {
            p->x = node.x;
            p->y = node.y;
            p->rotation = node.rotation;
            Bool success = false;
            if (action == 0) success = game_try_move(game, MOVE_LEFT);
            if (action == 1) success = game_try_move(game, MOVE_RIGHT);
            if (action == 2) success = game_try_move(game, MOVE_DOWN);
            if (action == 3) success = game_try_rotate(game, ROTATE_CW);
            if (action == 4) success = game_try_rotate(game, ROTATE_CCW);
            if (success) {
                if (p->x < 0 || p->x >= 10 || p->y < 0 || p->y >= 40) continue;
                int index = p->rotation * 10 * 40 + p->x * 40 + p->y;
                if (index < 0 || index >= 4 * 10 * 40) continue; // Safety
                if (!visited[index]) {
                    visited[index] = true;
                    bfs_push(&q, p->x, p->y, p->rotation);
                }
            }
        }
    }
    p->x = ox; p->y = oy; p->rotation = orot;
    out_moves->count = count;
}
Tetris* create_tetris(int seed) {
    if (seed == 0) seed = (int)time(NULL); // 自动不同
    GameConfig config = { .preview_count = 5, .is_hold_enabled = true, .seed = (unsigned int)seed };
    return tetris_init(&config);
}
void destroy_tetris(Tetris* tetris) {
    tetris_free(tetris);
}
Tetris* clone_tetris(const Tetris* tetris) {
    return tetris_copy(tetris);
}
// AI functions
void ai_reset_game(Tetris* tetris, int seed) {
    GameConfig config = { .preview_count = 5, .is_hold_enabled = true, .seed = seed };
    magic_srandom(seed);
    Tetris* new_tetris = tetris_init(&config);
    *tetris = *new_tetris;
    free(new_tetris);
}
void ai_get_state(const Tetris* tetris, int* board_buffer, int* queue_buffer, int* hold_buffer, int* meta_buffer) {
    const Board* board = &tetris->game.board;
    int idx = 0;
    for (int y = 0; y < 20; y++) {
        for (int x = 0; x < 10; x++) {
            board_buffer[idx++] = board_get_cell(board, x, y);
        }
    }
    const Previews* previews = &tetris->game.state.previews;
    for (int i = 0; i < 5; i++) {
        queue_buffer[i] = (int)previews_peek(previews, i);
    }
    hold_buffer[0] = tetris->game.state.has_hold_piece ? (int)tetris->game.state.hold_piece.type : 0;
    meta_buffer[0] = tetris->state.b2b_count;
    meta_buffer[1] = tetris->game.state.ren;
    meta_buffer[2] = tetris->game.state.can_hold_piece;
    meta_buffer[3] = (int)tetris->game.current_piece.type;
}
void ai_get_legal_moves(const Tetris* tetris, LegalMoves* out_moves) {
    out_moves->count = 0;
    Game temp_game = tetris->game; // copy to avoid modify
    run_bfs_for_current_piece(&temp_game, 0, out_moves);
    if (temp_game.config.is_hold_enabled && temp_game.state.can_hold_piece) {
        PieceType new_type;
        if (!temp_game.state.has_hold_piece) {
            new_type = previews_peek(&temp_game.state.previews, 0);
        } else {
            new_type = temp_game.state.hold_piece.type;
        }
        PieceType old_type = temp_game.current_piece.type;
        temp_game.current_piece.type = new_type;
        temp_game.current_piece.rotation = 0;
        piece_get_spawn_pos(new_type, &temp_game.current_piece.x, &temp_game.current_piece.y);
        run_bfs_for_current_piece(&temp_game, 1, out_moves);
        temp_game.current_piece.type = old_type;
        temp_game.current_piece.rotation = 0;
        piece_get_spawn_pos(old_type, &temp_game.current_piece.x, &temp_game.current_piece.y);
    }
}
StepResult ai_step(Tetris* tetris, int x, int y, int rotation, int use_hold) {
    StepResult result = {0};
    Game* game = &tetris->game;
    if (tetris->state.is_game_over) {
        result.is_game_over = true;
        return result;
    }
    if (use_hold) game_hold_piece(game);
    Piece* p = &game->current_piece;
    p->rotation = rotation % 4;
    p->x = x;
    p->y = y;
    if (board_piece_overlaps(&game->board, p)) {
        result.is_game_over = true;
        return result;
    }
    place_piece(&game->board, p);
    tetris->state.attack_type = game_get_attack_type(game);
    tetris->state.is_pc = game_is_perfect_clear(game);
    int atk = tetris_get_atk(tetris);
    result.attack_type = tetris->state.attack_type;
    result.lines_cleared = clear_rows(&game->board);
    game->state.ren = (result.lines_cleared > 0) ? game->state.ren + 1 : -1;
    result.combo_count = game->state.ren;
    if (result.attack_type != ATK_NONE && result.attack_type != ATK_SINGLE && result.attack_type != ATK_DOUBLE && result.attack_type != ATK_TRIPLE) {
        tetris->state.b2b_count++;
    } else if (result.attack_type != ATK_NONE) {
        tetris->state.b2b_count = -1;
    }
    result.b2b_count = tetris->state.b2b_count;
    if (result.lines_cleared > 0) {
        if (tetris->state.pending_attack > 0) {
            if (atk > tetris->state.pending_attack) {
                result.damage_sent = atk - tetris->state.pending_attack;
                tetris->state.pending_attack = 0;
            } else {
                tetris->state.pending_attack -= atk;
            }
        } else {
            result.damage_sent = atk;
        }
    } else {
        if (tetris->state.pending_attack > 0) {
            int p = tetris->state.pending_attack;
            if (p > 8) p = 8;
            tetris->state.pending_attack -= p;
            tetris_receive_garbage_line(tetris, p);
        }
    }
    PieceType type = previews_next(&game->state.previews, bag_next(&game->state.bag));
    piece_init(&game->current_piece, type);
    piece_get_spawn_pos(type, &game->current_piece.x, &game->current_piece.y);
    game->state.can_hold_piece = true;
    game->state.is_last_rotate = 0;
    if (board_piece_overlaps(&game->board, &game->current_piece)) {
        game->is_game_over = true;
    }
    result.is_game_over = game->is_game_over;
    return result;
}
void ai_receive_garbage(Tetris* tetris, int lines) {
    tetris_receive_garbage_line(tetris, lines); // 修改为直接添加垃圾行
}
static UIConfig* ai_ui_config = NULL;
void ai_enable_visualization(Tetris* tetris) {
    if (ai_ui_config == NULL) {
        ai_ui_config = init_ui_config(); // assume function
        init_window(tetris, ai_ui_config);
        SetTargetFPS(60);
    }
}
void ai_render(Tetris* tetris) {
    if (ai_ui_config) draw_content(tetris, ai_ui_config);
}
void ai_close_visualization() {
    if (ai_ui_config) {
        CloseWindow();
        free_ui_config(ai_ui_config);
        ai_ui_config = NULL;
    }
}