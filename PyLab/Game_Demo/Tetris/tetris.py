import pygame
import random

# 初始化游戏
pygame.init()

# 设置游戏窗口的尺寸
WIDTH, HEIGHT = 400, 600
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tetris")

# 定义颜色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# 定义方块的形状
SHAPES = [
    [[1, 1, 1],
     [0, 1, 0]],

    [[0, 2, 2],
     [2, 2, 0]],

    [[3, 3, 0],
     [0, 3, 3]],

    [[4, 0, 0],
     [4, 4, 4]],

    [[0, 0, 5],
     [5, 5, 5]],

    [[6, 6, 6, 6]]
]

# 定义方块的颜色
SHAPE_COLORS = [
    WHITE,
    RED,
    GREEN,
    BLUE,
    (255, 255, 0),  # Yellow
    (255, 140, 0),  # Orange
    (138, 43, 226)  # Violet
]

# 设置方块的大小
BLOCK_SIZE = 30

# 定义游戏板的大小
ROWS = 20
COLS = 10

# 定义游戏板
game_board = [[0] * COLS for _ in range(ROWS)]


def draw_board():
    for row in range(ROWS):
        for col in range(COLS):
            pygame.draw.rect(WINDOW, SHAPE_COLORS[game_board[row][col]], (col * BLOCK_SIZE, row * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 0)
            pygame.draw.rect(WINDOW, BLACK, (col * BLOCK_SIZE, row * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 1)


def draw_shape(shape, x, y, color):
    for row in range(len(shape)):
        for col in range(len(shape[0])):
            if shape[row][col]:
                pygame.draw.rect(WINDOW, color, ((x + col) * BLOCK_SIZE, (y + row) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 0)
                pygame.draw.rect(WINDOW, BLACK, ((x + col) * BLOCK_SIZE, (y + row) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 1)


def check_collision(shape, x, y):
    for row in range(len(shape)):
        for col in range(len(shape[0])):
            if shape[row][col]:
                if y + row >= ROWS or x + col < 0 or x + col >= COLS or game_board[y + row][x + col]:
                    return True
    return False


def place_shape(shape, x, y):
    for row in range(len(shape)):
        for col in range(len(shape[0])):
            if shape[row][col]:
                game_board[y + row][x + col] = shape[row][col]


def remove_complete_rows():
    global game_board
    new_board = [row for row in game_board if 0 not in row]
    if len(new_board) < ROWS:
        game_board = [[0] * COLS for _ in range(ROWS - len(new_board))] + new_board


def main():
    clock = pygame.time.Clock()
    current_shape = random.choice(SHAPES)
    current_color = random.choice(SHAPE_COLORS[1:])
    current_x, current_y = COLS // 2 - len(current_shape[0]) // 2, 0
    fall_time = 0
    fall_speed = 0.5

    running = True
    while running:
        WINDOW.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and not check_collision(current_shape, current_x - 1, current_y):
                    current_x -= 1
                elif event.key == pygame.K_RIGHT and not check_collision(current_shape, current_x + 1, current_y):
                    current_x += 1
                elif event.key == pygame.K_DOWN and not check_collision(current_shape, current_x, current_y + 1):
                    current_y += 1

        if not check_collision(current_shape, current_x, current_y + 1):
            current_y += 1
        else:
            place_shape(current_shape, current_x, current_y)
            remove_complete_rows()
            current_shape = random.choice(SHAPES)
            current_color = random.choice(SHAPE_COLORS[1:])
            current_x, current_y = COLS // 2 - len(current_shape[0]) // 2, 0

        draw_board()
        draw_shape(current_shape, current_x, current_y, current_color)
        pygame.display.update()

        if fall_time >= 1 / fall_speed:
            if not check_collision(current_shape, current_x, current_y + 1):
                current_y += 1
            else:
                place_shape(current_shape, current_x, current_y)
                remove_complete_rows()
                current_shape = random.choice(SHAPES)
                current_color = random.choice(SHAPE_COLORS[1:])
                current_x, current_y = COLS // 2 - len(current_shape[0]) // 2, 0
            fall_time = 0

        fall_time += clock.get_rawtime() / 1000
        clock.tick()

    pygame.quit()


if __name__ == "__main__":
    main()
