// core/game/game.c

#include "game.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define GAME_FPS 60
#define GAME_GRAVITY 1.0f / 60.0f

#define GAME_PREVIEW_COUNT 5
#define GAME_IS_HOLD_ENABLED TRUE

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

GameUIConfig* init_game_ui_config() {
    GameUIConfig* config = malloc(sizeof(GameUIConfig));
    
    config->fps = GAME_FPS;
    config->gravity = GAME_GRAVITY;
    
    return config;
}

GameConfig* init_game_config() {
    GameConfig* config = malloc(sizeof(GameConfig));
    
    config->preview_count = GAME_PREVIEW_COUNT;
    config->is_hold_enabled = GAME_IS_HOLD_ENABLED;

    return config;
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

    return state;
}

Game* init_game(Bool is_ui_enabled) {
    Game* game = malloc(sizeof(Game));

    if (is_ui_enabled) {
        GameUIConfig* ui_config = init_game_ui_config();
        game->ui_config = ui_config;
    }
    else {
        game->ui_config = NULL;
    }

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

Bool is_overlapping(Board* board, Piece* piece) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (piece->shape[i][j] == 0) continue;

            int x = piece->x + j;
            int y = piece->y - i;

            if (x < 0 || x >= board->width || y < 0 || y >= board->height + 3) return TRUE; 
            if (board->state[x][y] != 0) return TRUE;
        }
    }
    return FALSE;
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
        return rtn;
    }

    // not hard drop
    int original_x = current_piece->x;
    int original_y = current_piece->y;
    move_piece(current_piece, action);

    if (!is_overlapping(board, current_piece)) return TRUE;
    
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
    // not I piece
    if (current_piece->type != I_PIECE) {
        for (int i = 0; i < 5; i++) {
            Bool is_successful = try_displace_piece(
                game,
                NORMAL_PIECE_NORMAL_SRS[(int)original_rotation][(int)action][i]
            );
            if (is_successful) return TRUE;
        }
        current_piece->rotation = original_rotation;
        memcpy(current_piece->shape, original_shape, sizeof(current_piece->shape));
    }
    // I piece
    else {
        for (int i = 0; i < 5; i++) {
            Bool is_successful = try_displace_piece(
                game, 
                I_PIECE_NORMAL_SRS[(int)original_rotation][(int)action][i]
            );
            if (is_successful) return TRUE;
        }
        current_piece->rotation = original_rotation;
        memcpy(current_piece->shape, original_shape, sizeof(current_piece->shape));
    }
    return FALSE;
}

Bool try_rotate_piece_180(Game* game, RotationAction action) {
    // return: is_successful
    Piece* current_piece = game->current_piece;
    Rotation original_rotation = current_piece->rotation;

    int original_shape[4][4];
    memcpy(original_shape, current_piece->shape, sizeof(current_piece->shape));

    rotate_piece(current_piece, action);
    // not I piece
    if (current_piece->type != I_PIECE) {
        for (int i = 0; i < 6; i++) {
            Bool is_successful = try_displace_piece(
                game, 
                NORMAL_PIECE_180_SRS[(int)original_rotation][i]
            );
            if (is_successful) return TRUE;
        }
        current_piece->rotation = original_rotation;
        memcpy(current_piece->shape, original_shape, sizeof(current_piece->shape));
    }
    // I piece
    else {
        for (int i = 0; i < 2; i++) {
            Bool is_successful = try_displace_piece(
                game,
                I_PIECE_180_SRS[(int)original_rotation][i]
            );
            if (is_successful) return TRUE;
        }
        current_piece->rotation = original_rotation;
        memcpy(current_piece->shape, original_shape, sizeof(current_piece->shape));
    }
    return FALSE;
}

Bool try_rotate_piece(Game* game, RotationAction action) {
    // return: is_successful
    Bool rtn;
    if (action == ROTATE_180) {
        rtn = try_rotate_piece_180(game, action);
    }
    else {
        rtn = try_rotate_piece_normal(game, action);
    }
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

int clear_rows(Board* board) {
    int num_rows_cleared = 0;
    for (int y = board->height - 1; y >= 0; y--) {
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

Bool spawn_piece(Game* game, Bool is_free_needed) {
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
    new_piece->x = 3;
    new_piece->y = 21;
    game->current_piece = new_piece;
    
    if (is_free_needed) free(current_piece);

    game->state->can_hold_piece = TRUE;

    if (is_overlapping(game->board, new_piece)) return TRUE;

    return FALSE;
}

Bool next_piece(Game* game) {
    Bool rtn; // is_game_over
    rtn = lock_piece(game);
    rtn = rtn || spawn_piece(game, TRUE);
    return rtn;
}

Bool hold_piece(Game* game) {
    // return: is_game_over
    Piece* current_piece = game->current_piece;

    if (game->state->hold_piece == NULL) {
        game->state->hold_piece = current_piece;
        Bool is_game_over = spawn_piece(game, FALSE);
        game->state->can_hold_piece = FALSE;
        return is_game_over;
    }
    else {
        Piece* hold_piece = game->state->hold_piece;
        game->state->hold_piece = current_piece;
        Piece* new_piece = init_piece(hold_piece->type);
        new_piece->x = 3;
        new_piece->y = 21;
        new_piece->rotation = (Rotation)0;
        memcpy(new_piece->shape, hold_piece->shape, sizeof(hold_piece->shape));

        free(hold_piece);
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