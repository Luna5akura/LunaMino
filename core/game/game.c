// core/game/game.c

#include "game.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define GAME_SPAWN_X 3
#define GAME_SPAWN_Y 21
#define GAME_VISIBLE_HEIGHT 20

#define GAME_PREVIEW_COUNT 5
#define GAME_IS_HOLD_ENABLED TRUE
#define GAME_SEED 0

const int NORMAL_PIECE_NORMAL_SRS[4][2][5][2] = {
    { // UP
        { {0, 0}, {-1, 0}, {-1 ,1}, {0, -2}, {-1, -2} }, // CW
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

const int I_PIECE_NORMAL_SRS[4][2][5][2] = {
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

const int NORMAL_PIECE_180_SRS[4][6][2] = {
    { {0, 0}, {0, 1}, {1, 1}, {-1, 1}, {1, 0}, {-1, 0} }, // UP
    { {0, 0}, {1, 0}, {1, 2}, {1, 1}, {0, 2}, {0, 1} }, // RIGHT
    { {0, 0}, {0, -1}, {-1, -1}, {1, -1}, {-1, 0}, {1, 0} }, // DOWN
    { {0, 0}, {-1, 0}, {-1, 2}, {-1, 1}, {0, 2}, {0, 1} }, // LEFT
};

const int I_PIECE_180_SRS[4][2][2] = {
    { {0, 0}, {0, -1} }, // UP
    { {0, 0}, {1, 0} }, // RIGHT
    { {0, 0}, {0, 1} }, // DOWN
    { {0, 0}, {-1, 0} }, // LEFT
};

const int T_CORNERS[4][2] = {
    {0, 0}, {0, 2}, {2, 0}, {2, 2}
};


const int T_RECTS[4][2][2] = {
    { {0, 0}, {0, 2} },
    { {0, 2}, {2, 2} },
    { {2, 0}, {2, 2} },
    { {0, 0}, {2, 0} }
};

GameConfig* init_game_config() {
    GameConfig* config = malloc(sizeof(GameConfig));
    
    config->preview_count = GAME_PREVIEW_COUNT;
    config->is_hold_enabled = GAME_IS_HOLD_ENABLED;
    config->seed = GAME_SEED;
    srandom(config->seed);

    return config;
}

void free_game_config(GameConfig* config) {
    free(config);
}

GameConfig* copy_game_config(GameConfig* config) {
    GameConfig* new_config = malloc(sizeof(GameConfig));
    memcpy(new_config, config, sizeof(GameConfig));
    return new_config;
}

GameState* init_game_state(GameConfig* config) {
    GameState* state = malloc(sizeof(GameState));
    
    state->bag = init_bag();

    Previews* previews = init_previews(config->preview_count);
    previews->previews[0] = state->bag->sequence[0];
    for (int i = 1; i < config->preview_count; i++) {
        previews->previews[i] = bag_next_piece(state->bag);
    }
    state->previews = previews;
    state->hold_piece = NULL;
    state->can_hold_piece = TRUE;
    state->is_last_rotate = 0;
    state->is_last_clear_line = FALSE;
    state->ren = -1;

    return state;
}

void free_game_state(GameState* state) {
    free_bag(state->bag);
    free_previews(state->previews);
    if (state->hold_piece) free_piece(state->hold_piece);
    free(state);
}

GameState* copy_game_state(GameState* state) {
    GameState* new_state = malloc(sizeof(GameState));
    memcpy(new_state, state, sizeof(GameState));
    new_state->bag = copy_bag(state->bag);
    new_state->previews = copy_previews(state->previews);
    if (state->hold_piece) new_state->hold_piece = copy_piece(state->hold_piece);
    return new_state;
}

Game* init_game() {
    Game* game = malloc(sizeof(Game));

    GameConfig* config = init_game_config();
    game->config = config;

    GameState* state = init_game_state(config);
    game->state = state;

    Board* board = init_board();
    game->board = board;

    Piece* current_piece = init_piece(bag_next_piece(state->bag));
    current_piece->x = 3;
    current_piece->y = 21;
    current_piece->rotation = (Rotation)0;
    game->current_piece = current_piece;

    return game;
}

void free_game(Game* game) {
    free_game_config(game->config);
    free_game_state(game->state);
    free_board(game->board);
    if (game->current_piece) free_piece(game->current_piece);
    free(game);
}

Game* copy_game(Game* game) {
    Game* new_game = malloc(sizeof(Game));
    memcpy(new_game, game, sizeof(Game));
    new_game->config = copy_game_config(game->config);
    new_game->state = copy_game_state(game->state);
    new_game->board = copy_board(game->board);
    if (game->current_piece) new_game->current_piece = copy_piece(game->current_piece);
    return new_game;
}

Bool is_overlapping(Board* board, Piece* piece) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (piece->shape[i][j] == 0) continue;

            int x = piece->x + j;
            int y = piece->y - i;

            if (x < 0 || x >= board->width || y < 0) return TRUE; 
            if (board->state[x][y] != 0) return TRUE;
        }
    }
    return FALSE;
}

Bool is_top_out(Board* board, Piece* piece) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (piece->shape[i][j] == 0) continue;
            
            int y = piece->y - i;
            if (y < GAME_VISIBLE_HEIGHT) return FALSE;
        }
    }
    return TRUE;
} 

Bool hard_drop(Piece* piece, Board* board) {
    // return: is_successful
    piece->y--;
    if (is_overlapping(board, piece)) {
        piece->y++;
        return FALSE;
    }
    while (!is_overlapping(board, piece)) {
        piece->y--;
    }
    piece->y++;
    return TRUE;
}

Bool try_move_piece(Game* game, MoveAction action) {
    // return: is_successful
    Piece* current_piece = game->current_piece;
    Board* board = game->board;


    // hard drop
    if (action == MOVE_HARD_DROP) {
        Bool rtn = hard_drop(current_piece, board);
        if (rtn) game->state->is_last_rotate = 0;
        printf("hard drop: %d\n", rtn);
        return rtn;
    }

    // not hard drop
    int original_x = current_piece->x;
    int original_y = current_piece->y;
    move_piece(current_piece, action);

    if (!is_overlapping(board, current_piece)) {
        game->state->is_last_rotate = 0;
        return TRUE;
    }
    
    current_piece->x = original_x;
    current_piece->y = original_y;
    return FALSE;
}  


Bool try_displace_piece(Game* game, const int direction[2]) {
    // return: is_successful
    Piece* current_piece = game->current_piece;
    Board* board = game->board;
    int original_x = current_piece->x;
    int original_y = current_piece->y;
    displace_piece(current_piece, direction);

    if (!is_overlapping(board, current_piece)) return TRUE;
    
    current_piece->x = original_x;
    current_piece->y = original_y;
    return FALSE;
}

Bool try_rotate_piece_normal(Game* game, RotationAction action) {
    // return: is_successful
    Piece* current_piece = game->current_piece;
    Rotation original_rotation = current_piece->rotation;
    int original_shape[4][4];
    memcpy(original_shape, current_piece->shape, sizeof(current_piece->shape));
    rotate_piece(current_piece, action);

    for (int i = 0; i < 5; i++) {
        Bool is_successful = current_piece->type == I_PIECE
            ? try_displace_piece(game, I_PIECE_NORMAL_SRS[(int)original_rotation][(int)action][i])
            : try_displace_piece(game, NORMAL_PIECE_NORMAL_SRS[(int)original_rotation][(int)action][i]);

        if (is_successful) {
            game->state->is_last_rotate = i + 1;
            return TRUE;
        }
    }
    current_piece->rotation = original_rotation;
    memcpy(current_piece->shape, original_shape, sizeof(current_piece->shape));
    return FALSE;
}

Bool try_rotate_piece_180(Game* game, RotationAction action) {
    // return: is_successful
    Piece* current_piece = game->current_piece;
    Rotation original_rotation = current_piece->rotation;
    int original_shape[4][4];
    memcpy(original_shape, current_piece->shape, sizeof(current_piece->shape));
    rotate_piece(current_piece, action);

    int kick_cnt = current_piece->type == I_PIECE ? 2 : 6;

    for (int i = 0; i < kick_cnt; i++) {
        Bool is_successful = current_piece->type == I_PIECE
            ? try_displace_piece(game, I_PIECE_180_SRS[(int)original_rotation][i])
            : try_displace_piece(game, NORMAL_PIECE_180_SRS[(int)original_rotation][i]);

        if (is_successful) {
            game->state->is_last_rotate = i + 1;
            return TRUE;
        }
    }
    current_piece->rotation = original_rotation;
    memcpy(current_piece->shape, original_shape, sizeof(current_piece->shape));
    return FALSE;
}

Bool try_rotate_piece(Game* game, RotationAction action) {
    // return: is_successful
    Bool rtn;
    if (action == ROTATE_180) rtn = try_rotate_piece_180(game, action);
    else rtn = try_rotate_piece_normal(game, action);
    return rtn;
}

Bool lock_piece(Game* game) {
    // return: is_game_over
    Board* board = game->board;
    Piece* current_piece = game->current_piece;

    Bool rtn = TRUE;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (current_piece->shape[i][j] == 0) continue;

            int x = current_piece->x + j;
            int y = current_piece->y - i;

            if (y < 20) rtn = FALSE;
            
            board->state[x][y] = (int)current_piece->type + 1;
        }
    }
    return rtn;
}

int detect_clear_rows(Game* game) {
    // before lock_piece
    Board* board = game->board;
    Piece* current_piece = game->current_piece;

    int num_rows_cleared = 0;
    int temp_board[board->width][board->height];
    memcpy(temp_board, board->state, sizeof(board->state));

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (current_piece->shape[i][j] == 0) continue;

            int x = current_piece->x + j;
            int y = current_piece->y - i;
            
            temp_board[x][y] = (int)current_piece->type + 1;
        }
    }    

    for (int y = board->height - 1; y >= 0; y--) {
        Bool is_row_full = TRUE;
        for (int x = 0; x < board->width; x++) {
            if (temp_board[x][y] == 0) {
                is_row_full = FALSE;
                break;
            }
        }
        if (!is_row_full) continue;

        num_rows_cleared++;
    }

    return num_rows_cleared;
}

void update_ren(Game* game) {
    // before lock_piece
    int num_rows_cleared = detect_clear_rows(game);
    if (num_rows_cleared == 0) {
        game->state->ren = -1;
    }
    else {
        game->state->ren += 1;
    }
}

int clear_rows(Board* board) {
    int num_rows_cleared = 0;
    for (int y = board->height; y >= 0; y--) {
        Bool is_row_full = TRUE;
        for (int x = 0; x < board->width; x++) {
            if (board->state[x][y] == 0) {
                is_row_full = FALSE;
                break;
            }
        }
        if (!is_row_full) continue;

        num_rows_cleared++;
        for (int yy = y; yy <= board->height - 1; yy++) {
            for (int x = 0; x < board->width; x++) {
                board->state[x][yy] = board->state[x][yy + 1];
            }
        }
        for (int x = 0; x < board->width; x++) {
            board->state[x][board->height - 1] = 0;
        }
    }

    return num_rows_cleared;
}

Bool spawn_piece(Game* game) {
    // return: is_game_over
    Piece* current_piece = game->current_piece;

    PieceType type;
    if (game->config->preview_count == 0) {
        type = bag_next_piece(game->state->bag);
    }
    else {
        type = next_preview(game->state->previews, bag_next_piece(game->state->bag));
    }
    

    Piece* new_piece = init_piece(type);
    new_piece->x = GAME_SPAWN_X;
    new_piece->y = GAME_SPAWN_Y;
    game->current_piece = new_piece;
    
    free(current_piece);

    game->state->can_hold_piece = TRUE;

    if (is_overlapping(game->board, new_piece)) return TRUE;

    return FALSE;
}

Bool next_piece(Game* game) {
    Bool rtn; // is_game_over
    rtn = lock_piece(game);
    rtn = rtn || spawn_piece(game);
    return rtn;
}

Bool hold_piece(Game* game) {
    // return: is_game_over
    PieceType current_piece_type = game->current_piece->type;

    if (game->state->hold_piece == NULL) {
        Bool is_game_over = spawn_piece(game);
        game->state->hold_piece = init_piece(current_piece_type);

        game->state->can_hold_piece = FALSE;
        return is_game_over;
    }
    else {
        Piece* hold_piece = game->state->hold_piece;
        free_piece(game->current_piece);
        game->state->hold_piece = init_piece(current_piece_type);

        Piece* new_piece = init_piece(hold_piece->type);
        new_piece->x = 3;
        new_piece->y = 21;
        new_piece->rotation = (Rotation)0;
        memcpy(new_piece->shape, hold_piece->shape, sizeof(hold_piece->shape));

        free_piece(hold_piece);
        game->current_piece = new_piece;

        if (is_overlapping(game->board, new_piece)) return TRUE;

        game->state->can_hold_piece = FALSE;
        return FALSE;
    }
}

Bool try_hold_piece(Game* game) {
    // return: is_game_over
    if (!game->config->is_hold_enabled) return FALSE;
    if (!game->state->can_hold_piece) return FALSE;
    return hold_piece(game);
}

Bool is_t_triple_corner(Game* game) {
    Piece* current_piece = game->current_piece;
    int count = 0;
    for (int i = 0; i < 4; i++) {
        int x = current_piece->x + T_CORNERS[i][0];
        int y = current_piece->y - T_CORNERS[i][1];
        if (
            game->board->state[x][y] != 0
            || x < 0
            || x >= game->board->width
            || y < 0
        ) count++;
    }
    return count >= 3;
}

Bool is_t_rect_has_hole(Game* game) {
    Piece* current_piece = game->current_piece;
    int count = 0;
    for (int i = 0; i < 2; i++) {
        int x = current_piece->x + T_RECTS[(int)current_piece->rotation][i][1];
        int y = current_piece->y - T_RECTS[(int)current_piece->rotation][i][0];
        if (game->board->state[x][y] != 0) count++;
    }
    return count == 1;
}

Bool is_current_piece_movable(Game* game) {
    Piece* current_piece = game->current_piece;
    
    int original_x = current_piece->x;
    int original_y = current_piece->y;
    int directions[4][2] = {
        {0, 1}, {1, 0}, {0, -1}, {-1, 0}
    };
    for (int i = 0; i < 4; i++) {
        current_piece->x = original_x + directions[i][0];
        current_piece->y = original_y + directions[i][1];
        printf("try move: %d, %d\n", current_piece->x, current_piece->y);
        printf("is_overlapping: %d\n", is_overlapping(game->board, current_piece));
        if (!is_overlapping(game->board, current_piece)) {
            current_piece->x = original_x;
            current_piece->y = original_y;
            return TRUE;
        }
    }
    current_piece->x = original_x;
    current_piece->y = original_y;
    return FALSE;
}
        

AttackType get_none_spin_attack_type(int num_rows_cleared) {
    switch (num_rows_cleared) {
        case 1: return ATK_SINGLE;
        case 2: return ATK_DOUBLE;
        case 3: return ATK_TRIPLE;
        case 4: return ATK_TETRIS;
        default: return ATK_ERROR;
    }
}

AttackType get_t_attack_type(Game* game, int num_rows_cleared) {
    if (
        game->state->is_last_rotate == 0
        || (
            !is_t_triple_corner(game)
            // mini+
            && is_current_piece_movable(game)
        )
    ) return get_none_spin_attack_type(num_rows_cleared);
    printf("game->state->is_last_rotate: %d\n");
    if (is_t_rect_has_hole(game) && game->state->is_last_rotate != 5) {
        // mini
        switch (num_rows_cleared) {
            case 1: return ATK_TSMS;
            case 2: return ATK_TSMD;
            default: return ATK_ERROR;
        }
    }
    else {
        // not mini
        switch (num_rows_cleared) {
            case 1: return ATK_TSS;
            case 2: return ATK_TSD;
            case 3: return ATK_TST;
            default: return ATK_ERROR;
        }
    }
}

AttackType get_i_attack_type(Game* game, int num_rows_cleared) {
    if (
        game->state->is_last_rotate == 0
        || is_current_piece_movable(game)
    ) return get_none_spin_attack_type(num_rows_cleared);

    switch (num_rows_cleared) {
        case 1: return ATK_ISS;
        case 2: return ATK_ISD;
        case 3: return ATK_IST;
        default: return ATK_ERROR;
    }
}

AttackType get_o_attack_type(Game* game, int num_rows_cleared) {
    if (
        game->state->is_last_rotate == 0
        || is_current_piece_movable(game)
    ) return get_none_spin_attack_type(num_rows_cleared);
    
    switch (num_rows_cleared) {
        case 1: return ATK_OSS;
        case 2: return ATK_OSD;
        default: return ATK_ERROR;
    }
}

AttackType get_s_attack_type(Game* game, int num_rows_cleared) {
    if (
        game->state->is_last_rotate == 0
        || is_current_piece_movable(game)
    ) return get_none_spin_attack_type(num_rows_cleared);
    
    switch (num_rows_cleared) {
        case 1: return ATK_SSS;
        case 2: return ATK_SSD;
        case 3: return ATK_SST;
        default: return ATK_ERROR;
    }
}

AttackType get_z_attack_type(Game* game, int num_rows_cleared) {
    if (
        game->state->is_last_rotate == 0
        || is_current_piece_movable(game)
    ) return get_none_spin_attack_type(num_rows_cleared);
    
    switch (num_rows_cleared) {
        case 1: return ATK_ZSS;
        case 2: return ATK_ZSD;
        case 3: return ATK_ZST;
        default: return ATK_ERROR;
    }
}

AttackType get_j_attack_type(Game* game, int num_rows_cleared) {
    if (
        game->state->is_last_rotate == 0
        || is_current_piece_movable(game)
    ) return get_none_spin_attack_type(num_rows_cleared);
    
    switch (num_rows_cleared) {
        case 1: return ATK_JSS;
        case 2: return ATK_JSD;
        case 3: return ATK_JST;
        default: return ATK_ERROR;
    }
}

AttackType get_l_attack_type(Game* game, int num_rows_cleared) {
    if (
        game->state->is_last_rotate == 0
        || is_current_piece_movable(game)
    ) return get_none_spin_attack_type(num_rows_cleared);
    
    switch (num_rows_cleared) {
        case 1: return ATK_LSS;
        case 2: return ATK_LSD;
        case 3: return ATK_LST;
        default: return ATK_ERROR;
    }
}


AttackType get_attack_type(Game* game) {
    // before lock_piece
    int num_rows_cleared = detect_clear_rows(game);
    if (num_rows_cleared == 0) return ATK_NONE;

    switch (game->current_piece->type) {
        case T_PIECE: return get_t_attack_type(game, num_rows_cleared);
        case I_PIECE: return get_i_attack_type(game, num_rows_cleared);
        case O_PIECE: return get_o_attack_type(game, num_rows_cleared);
        case S_PIECE: return get_s_attack_type(game, num_rows_cleared);
        case Z_PIECE: return get_z_attack_type(game, num_rows_cleared);
        case J_PIECE: return get_j_attack_type(game, num_rows_cleared);
        case L_PIECE: return get_l_attack_type(game, num_rows_cleared);
        default: return ATK_ERROR;
    }
}

Bool is_perfect_clear(Game* game) {
    Board* board = game->board;
    Piece* current_piece = game->current_piece;

    int temp_board[board->width][board->height];
    memcpy(temp_board, board->state, sizeof(board->state));

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (current_piece->shape[i][j] == 0) continue;

            int x = current_piece->x + j;
            int y = current_piece->y - i;
            
            temp_board[x][y] = (int)current_piece->type + 1;
        }
    }    

    for (int y = board->height - 1; y >= 0; y--) {
        Bool is_row_full = TRUE;
        for (int x = 0; x < board->width; x++) {
            if (temp_board[x][y] == 0) {
                is_row_full = FALSE;
                break;
            }
        }
        if (!is_row_full) continue;

        for (int yy = y; yy <= board->height - 1; yy++) {
            for (int x = 0; x < board->width; x++) {
                temp_board[x][yy] = temp_board[x][yy + 1];
            }
        }
        for (int x = 0; x < board->width; x++) {
            temp_board[x][board->height - 1] = 0;
        }
    }

    for (int y = 0; y < board->height; y++) {
        for (int x = 0; x < board->width; x++) {
            if (temp_board[x][y] != 0) return FALSE;
        }
    }
    return TRUE;
}


// Not core functions

Bool is_grounded(Game* game) {
    Piece* current_piece = game->current_piece;
    
    current_piece->y--;
    Bool rtn = is_overlapping(game->board, current_piece);
    current_piece->y++;

    return rtn;
}

int get_shadow_height(Game* game) {
    Piece* current_piece = game->current_piece;

    if (is_overlapping(game->board, current_piece)) return 0;

    int shadow_height = -1;
    int original_y = current_piece->y;
    while (!is_overlapping(game->board, current_piece)) {
        current_piece->y--;
        shadow_height++;
    }
    current_piece->y = original_y;
    
    return shadow_height;
}