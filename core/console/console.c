// core/console/console.c

#include "../game/game.h"
#include <stdio.h>
#include <stdlib.h>

const char PRINT_TABLE[8] = "IOTSZJL";

void print_board(Game* game) {
    Board* board = game->board;
    Piece* piece = game->current_piece;

    int temp_board[board->width][board->height + 2];
    for (int x = 0; x < board->width; x++) {
        for (int y = 0; y < board->height + 2; y++) {
            temp_board[x][y] = board->state[x][y];
        }
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (piece->shape[i][j] != 0) { // shape[y][x]
                int x = piece->x + j;
                int y = piece->y - i;
                if (x >= 0 && x < board->width && y >= 0 && y < 20) {
                    temp_board[x][y] = (int)piece->type + 1;
                }
            }
        }
    }

    for (int y = 19; y >= 0; y--) {
        printf("|");
        for (int x = 0; x < board->width; x++) {
            if (temp_board[x][y] == 0) {
                printf(" .");
            } else {
                printf(" %c", PRINT_TABLE[temp_board[x][y] - 1]);
            }
        }
        printf("|\n");
    }
    printf("+");
    for (int x = 0; x < board->width; x++) {
        printf("--");
    }
    printf("+\n");
}

void handle_input(Game* game) {
    char input;
    scanf(" %c", &input);

    switch (input) {
        case 'a':
            try_move_piece(game, MOVE_LEFT);
            break;
        case 'd':
            try_move_piece(game, MOVE_RIGHT);
            break;
        case 's':
            try_move_piece(game, MOVE_DOWN);
            break;
        case 'w':
            try_move_piece(game, MOVE_HARD_DROP);
            next_piece(game);
            break;
        case 'e':
            try_rotate_piece(game, ROTATE_CW);
            break;
        case 'q':
            try_rotate_piece(game, ROTATE_CCW);
            break;
        case 'r':
            try_rotate_piece(game, ROTATE_180);
            break;
        case 'x':
            free(game->current_piece);
            free(game->board);
            free(game);
            exit(0);
            break;
        default:
            printf("Invalid input! Use: a (left), d (right), s (down), w (hard drop), q (rotate CW), e (rotate CCW), r (rotate 180), x (exit)\n");
    }
}

int main() {
    srandom(0);
    Game* game = init_game();

    printf("Tetris Console Test\n");
    printf("Controls: a (left), d (right), s (down), w (hard drop), q (rotate CW), e (rotate CCW), r (rotate 180), n (next piece), x (exit)\n");

    while (1) {
        print_board(game);
        printf("Enter action: ");
        handle_input(game);
        clear_rows(game->board); 
    }

    return 0;
}