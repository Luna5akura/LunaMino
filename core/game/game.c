// core/game/game.c
#include "game.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h> // 如果需要调试，保留，否则可移除
#include <time.h> // for potential time seed

#define GAME_VISIBLE_HEIGHT 20
#define DEFAULT_PREVIEW_COUNT 5
#define DEFAULT_IS_HOLD_ENABLED true
#define DEFAULT_SEED 0

// SRS 踢墙表 (优化：const 数组)
static const int NORMAL_PIECE_NORMAL_SRS[4][2][5][2] = {
    { // UP
        { {0, 0}, {-1, 0}, {-1, 1}, {0, -2}, {-1, -2} }, // CW
        { {0, 0}, {1, 0}, {1, 1}, {0, -2}, {1, -2} }, // CCW
    },
    { // RIGHT
        { {0, 0}, {1, 0}, {1, -1}, {0, 2}, {1, 2} }, // CW
        { {0, 0}, {1, 0}, {1, -1}, {0, 2}, {1, 2} }, // CCW
    },
    { // DOWN
        { {0, 0}, {1, 0}, {1, 1}, {0, -2}, {1, -2} }, // CW
        { {0, 0}, {-1, 0}, {-1, 1}, {0, -2}, {-1, -2} }, // CCW
    },
    { // LEFT
        { {0, 0}, {-1, 0}, {-1, -1}, {0, 2}, {-1, 2} }, // CW
        { {0, 0}, {-1, 0}, {-1, -1}, {0, 2}, {-1, 2} }, // CCW
    }
};
static const int I_PIECE_NORMAL_SRS[4][2][5][2] = {
    { // UP
        { {0, 0}, {-2, 0}, {1, 0}, {-2, -1}, {1, 2} }, // CW
        { {0, 0}, {-1, 0}, {2, 0}, {-1, 2}, {2, -1} }, // CCW
    },
    { // RIGHT
        { {0, 0}, {-1, 0}, {2, 0}, {-1, 2}, {2, -1} }, // CW
        { {0, 0}, {2, 0}, {-1, 0}, {2, 1}, {-1, -2} }, // CCW
    },
    { // DOWN
        { {0, 0}, {2, 0}, {-1, 0}, {2, 1}, {-1, -2} }, // CW
        { {0, 0}, {1, 0}, {-2, 0}, {1, -2}, {-2, 1} }, // CCW
    },
    { // LEFT
        { {0, 0}, {1, 0}, {-2, 0}, {1, -2}, {-2, 1} }, // CW
        { {0, 0}, {-2, 0}, {1, 0}, {-2, -1}, {1, 2} }, // CCW
    }
};
static const int NORMAL_PIECE_180_SRS[4][6][2] = {
    { {0, 0}, {0, 1}, {1, 1}, {-1, 1}, {1, 0}, {-1, 0} }, // UP
    { {0, 0}, {1, 0}, {1, 2}, {1, 1}, {0, 2}, {0, 1} }, // RIGHT
    { {0, 0}, {0, -1}, {-1, -1}, {1, -1}, {-1, 0}, {1, 0} }, // DOWN
    { {0, 0}, {-1, 0}, {-1, 2}, {-1, 1}, {0, 2}, {0, 1} }, // LEFT
};
static const int I_PIECE_180_SRS[4][2][2] = {
    { {0, 0}, {0, -1} }, // UP
    { {0, 0}, {1, 0} }, // RIGHT
    { {0, 0}, {0, 1} }, // DOWN
    { {0, 0}, {-1, 0} }, // LEFT
};

// T helpers
static const int T_CORNERS[4][2] = {
    {0, 0}, {0, 2}, {2, 0}, {2, 2}
};
static const int T_RECTS[4][2][2] = {
    { {0, 0}, {0, 2} },
    { {0, 2}, {2, 2} },
    { {2, 0}, {2, 2} },
    { {0, 0}, {2, 0} }
};


Game* game_init(const GameConfig* config) {
    Game* game = (Game*)malloc(sizeof(Game));
    if (game == NULL) return NULL;
    memset(game, 0, sizeof(Game));

    // Config defaults
    game->config.preview_count = config ? config->preview_count : DEFAULT_PREVIEW_COUNT;
    if (game->config.preview_count > MAX_PREVIEW_CAPACITY) game->config.preview_count = MAX_PREVIEW_CAPACITY;
    game->config.is_hold_enabled = config ? config->is_hold_enabled : DEFAULT_IS_HOLD_ENABLED;
    game->config.seed = config ? config->seed : DEFAULT_SEED;

    // Random seed
    magic_srandom(game->config.seed);

    // Init state
    bag_init(&game->state.bag);
    previews_init(&game->state.previews, game->config.preview_count);

    // Fill previews
    for (int i = 0; i < game->state.previews.capacity; i++) {
        (void)previews_next(&game->state.previews, bag_next(&game->state.bag));
    }

    // Init board (no malloc, direct init)
    memset(&game->board, 0, sizeof(Board));
    game->board.width = BOARD_WIDTH;
    game->board.height = BOARD_HEIGHT;

    // Current piece
    PieceType type = previews_next(&game->state.previews, bag_next(&game->state.bag));
    piece_init(&game->current_piece, type);
    piece_get_spawn_pos(type, &game->current_piece.x, &game->current_piece.y);

    // Hold and state
    game->state.has_hold_piece = false;
    game->state.can_hold_piece = true;
    game->state.is_last_rotate = 0;
    game->state.is_last_clear_line = false;
    game->state.ren = -1;

    game->is_game_over = false;

    return game;
}

void game_free(Game* game) {
    free(game);
}

Game* game_copy(const Game* src) {
    if (!src) return NULL;
    Game* dest = (Game*)malloc(sizeof(Game));
    if (dest == NULL) return NULL;
    memcpy(dest, src, sizeof(Game));
    return dest;
}

Bool board_piece_overlaps(const Board* board, const Piece* p) {
    uint16_t mask = piece_get_mask(p);
    for (int dy = 0; dy < 4; dy++) {
        for (int dx = 0; dx < 4; dx++) {
            if (mask & (1u << (15 - (dy * 4 + dx)))) {
                int bx = p->x + dx;
                int by = p->y - dy;
                if (bx < 0 || bx >= board->width || by < 0) return true;
                if (by >= board->height) continue; // 上方缓冲区允许
                if (board_get_cell(board, bx, by)) return true;
            }
        }
    }
    return false;
}

static void place_piece(Board* board, const Piece* p) {
    uint16_t mask = piece_get_mask(p);
    for (int dy = 0; dy < 4; dy++) {
        for (int dx = 0; dx < 4; dx++) {
            if (mask & (1u << (15 - (dy * 4 + dx)))) {
                int bx = p->x + dx;
                int by = p->y - dy;
                if (bx >= 0 && bx < board->width && by >= 0 && by < board->height) {
                    board_set_cell(board, bx, by, 1);
                }
            }
        }
    }
}

int clear_rows(Board* board) {
    int cleared = 0;
    int dst = 0;
    for (int src = 0; src < board->height; src++) {
        if (board_is_row_full(board, src)) {
            cleared++;
        } else {
            if (dst != src) {
                board->rows[dst] = board->rows[src];
            }
            dst++;
        }
    }
    while (dst < board->height) {
        board->rows[dst++] = 0;
    }
    return cleared;
}

static Bool is_top_out(const Piece* p) {
    uint16_t mask = piece_get_mask(p);
    Bool all_above = true;
    for (int dy = 0; dy < 4; dy++) {
        for (int dx = 0; dx < 4; dx++) {
            if (mask & (1u << (15 - (dy * 4 + dx)))) {
                int by = p->y - dy;
                if (by < GAME_VISIBLE_HEIGHT) all_above = false;
            }
        }
    }
    return all_above;
}

static int detect_clear_rows(const Board* board, const Piece* p) {
    Board temp = *board;
    place_piece(&temp, p);
    int cleared = 0;
    for (int y = 0; y < temp.height; y++) {
        if (board_is_row_full(&temp, y)) cleared++;
    }
    return cleared;
}

static Bool game_try_displace(Game* game, int dx, int dy) {
    Piece* p = &game->current_piece;
    int ox = p->x;
    int oy = p->y;
    p->x += dx;
    p->y += dy;
    if (!board_piece_overlaps(&game->board, p)) return true;
    p->x = ox;
    p->y = oy;
    return false;
}

Bool game_try_move(Game* game, MoveAction action) {
    if (game->is_game_over) return false;
    Piece* p = &game->current_piece;
    int ox = p->x, oy = p->y;
    piece_move(p, action);
    if (action == MOVE_HARD_DROP) {
        while (!board_piece_overlaps(&game->board, p)) {
            p->y--;
        }
        p->y++; // 回退一步
        game->state.is_last_rotate = 0;
        return true;
    }
    if (!board_piece_overlaps(&game->board, p)) {
        game->state.is_last_rotate = 0;
        return true;
    }
    p->x = ox;
    p->y = oy;
    return false;
}

Bool game_try_rotate(Game* game, RotationAction action) {
    if (game->is_game_over) return false;
    Piece* p = &game->current_piece;
    Rotation orot = p->rotation;
    piece_rotate(p, action);
    const int (*kicks)[2];
    int kick_count;
    if (action == ROTATE_180) {
        kick_count = (p->type == PIECE_I) ? 2 : 6;
        kicks = (p->type == PIECE_I) ? I_PIECE_180_SRS[(int)orot] : NORMAL_PIECE_180_SRS[(int)orot];
    } else {
        kick_count = 5;
        int dir = (action == ROTATE_CW) ? 0 : 1;
        kicks = (p->type == PIECE_I) ? I_PIECE_NORMAL_SRS[(int)orot][dir] : NORMAL_PIECE_NORMAL_SRS[(int)orot][dir];
    }
    for (int i = 0; i < kick_count; i++) {
        if (game_try_displace(game, kicks[i][0], kicks[i][1])) {
            game->state.is_last_rotate = i + 1;
            return true;
        }
    }
    p->rotation = orot;
    return false;
}

Bool game_hold_piece(Game* game) {
    if (game->is_game_over || !game->config.is_hold_enabled || !game->state.can_hold_piece) return false;
    PieceType ctype = game->current_piece.type;
    Rotation crot = game->current_piece.rotation;
    if (!game->state.has_hold_piece) {
        game->state.hold_piece.type = ctype;
        game->state.hold_piece.rotation = 0; // 重置旋转
        game->state.has_hold_piece = true;
    } else {
        PieceType htype = game->state.hold_piece.type;
        game->state.hold_piece.type = ctype;
        game->state.hold_piece.rotation = 0;
        game->current_piece.type = htype;
        game->current_piece.rotation = 0;
    }
    piece_get_spawn_pos(game->current_piece.type, &game->current_piece.x, &game->current_piece.y);
    game->state.can_hold_piece = false;
    game->state.is_last_rotate = 0;
    if (board_piece_overlaps(&game->board, &game->current_piece)) {
        game->is_game_over = true;
        return false;
    }
    return true;
}

Bool game_next_step(Game* game) {
    if (game->is_game_over) return true;
    place_piece(&game->board, &game->current_piece);
    if (is_top_out(&game->current_piece)) {
        game->is_game_over = true;
        return true;
    }
    int cleared = clear_rows(&game->board);
    game->state.is_last_clear_line = (cleared > 0);
    if (cleared > 0) {
        game->state.ren++;
    } else {
        game->state.ren = -1;
    }
    PieceType type = previews_next(&game->state.previews, bag_next(&game->state.bag));
    piece_init(&game->current_piece, type);
    piece_get_spawn_pos(type, &game->current_piece.x, &game->current_piece.y);
    game->state.can_hold_piece = true;
    game->state.is_last_rotate = 0;
    if (board_piece_overlaps(&game->board, &game->current_piece)) {
        game->is_game_over = true;
        return true;
    }
    return false;
}

int game_get_shadow_height(const Game* game) {
    if (game->is_game_over) return 0;
    Piece temp = game->current_piece;
    int height = 0;
    while (!board_piece_overlaps(&game->board, &temp)) {
        temp.y--;
        height++;
    }
    return height - 1; // 减去最后一步
}

static Bool is_t_triple_corner(const Game* game) {
    const Piece* p = &game->current_piece;
    int count = 0;
    for (int i = 0; i < 4; i++) {
        int x = p->x + T_CORNERS[i][0];
        int y = p->y - T_CORNERS[i][1];
        if (board_get_cell(&game->board, x, y)) count++; // returns 1 if occupied or out
    }
    return count >= 3;
}

static Bool is_t_rect_has_hole(const Game* game) {
    const Piece* p = &game->current_piece;
    int count = 0;
    for (int i = 0; i < 2; i++) {
        int x = p->x + T_RECTS[p->rotation][i][0];
        int y = p->y - T_RECTS[p->rotation][i][1];
        if (board_get_cell(&game->board, x, y)) count++;
    }
    return count == 1;
}

static Bool is_current_piece_movable(const Game* game) {
    Piece temp = game->current_piece;
    const int directions[4][2] = {{ -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 }}; // left, right, down, up
    for (int i = 0; i < 4; i++) {
        temp.x = game->current_piece.x + directions[i][0];
        temp.y = game->current_piece.y + directions[i][1];
        if (!board_piece_overlaps(&game->board, &temp)) return true;
    }
    return false;
}

static AttackType get_none_spin_attack_type(int cleared) {
    switch (cleared) {
        case 1: return ATK_SINGLE;
        case 2: return ATK_DOUBLE;
        case 3: return ATK_TRIPLE;
        case 4: return ATK_TETRIS;
        default: return ATK_NONE;
    }
}

static AttackType get_t_attack_type(const Game* game, int cleared) {
    if (game->state.is_last_rotate == 0 || (!is_t_triple_corner(game) && is_current_piece_movable(game))) {
        return get_none_spin_attack_type(cleared);
    }
    if (is_t_rect_has_hole(game) && game->state.is_last_rotate != 5) {
        switch (cleared) {
            case 0: return ATK_TSMS; // mini single?
            case 1: return ATK_TSS;
            case 2: return ATK_TSMD;
            default: return ATK_ERROR;
        }
    }
    switch (cleared) {
        case 1: return ATK_TSS;
        case 2: return ATK_TSD;
        case 3: return ATK_TST;
        default: return ATK_ERROR;
    }
}

// 类似为其他类型实现 get_i_attack_type 等，简化假设相同逻辑
// 对于 I, O, S, Z, J, L: if last_rotate !=0 && !movable, then spin version, else normal.

static AttackType get_spin_attack_type(PieceType type, int cleared, Bool is_spin) {
    if (!is_spin) return get_none_spin_attack_type(cleared);
    switch (type) {
        case PIECE_I:
            if (cleared == 1) return ATK_ISS;
            if (cleared == 2) return ATK_ISD;
            if (cleared == 3) return ATK_IST;
            break;
        case PIECE_O:
            if (cleared == 1) return ATK_OSS;
            if (cleared == 2) return ATK_OSD;
            break;
        case PIECE_S:
            if (cleared == 1) return ATK_SSS;
            if (cleared == 2) return ATK_SSD;
            if (cleared == 3) return ATK_SST;
            break;
        case PIECE_Z:
            if (cleared == 1) return ATK_ZSS;
            if (cleared == 2) return ATK_ZSD;
            if (cleared == 3) return ATK_ZST;
            break;
        case PIECE_J:
            if (cleared == 1) return ATK_JSS;
            if (cleared == 2) return ATK_JSD;
            if (cleared == 3) return ATK_JST;
            break;
        case PIECE_L:
            if (cleared == 1) return ATK_LSS;
            if (cleared == 2) return ATK_LSD;
            if (cleared == 3) return ATK_LST;
            break;
        default: break;
    }
    return ATK_ERROR;
}

AttackType game_get_attack_type(const Game* game) {
    int cleared = detect_clear_rows(&game->board, &game->current_piece);
    if (cleared == 0) return ATK_NONE;
    Bool is_spin = (game->state.is_last_rotate != 0 && !is_current_piece_movable(game));
    if (game->current_piece.type == PIECE_T) {
        return get_t_attack_type(game, cleared);
    } else {
        return get_spin_attack_type(game->current_piece.type, cleared, is_spin);
    }
}

Bool game_is_perfect_clear(const Game* game) {
    Board temp = game->board;
    place_piece(&temp, &game->current_piece);
    clear_rows(&temp);
    for (int y = 0; y < temp.height; y++) {
        if (!board_is_row_empty(&temp, y)) return false;
    }
    return true;
}

static Bool is_grounded(const Game* game) {
    Piece temp = game->current_piece;
    temp.y--;
    return board_piece_overlaps(&game->board, &temp);
}