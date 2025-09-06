import pygame
import numpy as np
import sys
import random

# ----------------------
# Enhanced Tic-Tac-Toe
# Pygame + NumPy + Unbeatable Minimax AI
# Features:
# - Polished GUI with hover highlight
# - Scoreboard and Restart button
# - Winning line animation
# - Minimax AI (unbeatable)
# Run: python tic_tac_toe_enhanced.py
# ----------------------

pygame.init()

# ----- Config -----
WIDTH = 640
BOARD_SIZE = 3
SQUARE = WIDTH // BOARD_SIZE
INFO_HEIGHT = 120
HEIGHT = WIDTH + INFO_HEIGHT
LINE_WIDTH = 10
CIRCLE_RADIUS = int(SQUARE * 0.28)
CIRCLE_WIDTH = 12
CROSS_WIDTH = 18
PADDING = int(SQUARE * 0.18)
FPS = 60

# Colors
BG = (30, 30, 40)
BOARD_BG = (18, 24, 33)
LINE_COLOR = (40, 180, 170)
HOVER_COLOR = (60, 200, 190)
X_COLOR = (230, 230, 230)
O_COLOR = (240, 200, 170)
TEXT = (230, 230, 230)
WIN_LINE = (255, 80, 80)
BUTTON_BG = (40, 120, 120)
BUTTON_TEXT = (255, 255, 255)

# Fonts
TITLE_FONT = pygame.font.SysFont("Segoe UI", 28, bold=True)
INFO_FONT = pygame.font.SysFont("Segoe UI", 18)
SCORE_FONT = pygame.font.SysFont("Segoe UI", 20, bold=True)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tic-Tac-Toe — Enhanced")
clock = pygame.time.Clock()

# Board: 0 empty, 1 player(X), 2 AI(O)
board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)

# Score
scores = {"Player": 0, "AI": 0, "Draws": 0}

# Game state
player_turn = True
game_over = False
winner_line = None
winner_symbol = None
animate_progress = 0.0

# Precompute win lines (tuples of coordinates)
WIN_LINES = [
    [(0, 0), (0, 1), (0, 2)],
    [(1, 0), (1, 1), (1, 2)],
    [(2, 0), (2, 1), (2, 2)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 1), (1, 1), (2, 1)],
    [(0, 2), (1, 2), (2, 2)],
    [(0, 0), (1, 1), (2, 2)],
    [(0, 2), (1, 1), (2, 0)],
]

# ----- Drawing Helpers -----

def draw_board():
    screen.fill(BG)
    # board background
    pygame.draw.rect(screen, BOARD_BG, (20, 20, WIDTH - 40, WIDTH - 40), border_radius=8)
    # grid lines
    for i in range(1, BOARD_SIZE):
        start_h = (20, 20 + i * SQUARE)
        end_h = (20 + WIDTH - 40, 20 + i * SQUARE)
        pygame.draw.line(screen, LINE_COLOR, start_h, end_h, LINE_WIDTH)
        start_v = (20 + i * SQUARE, 20)
        end_v = (20 + i * SQUARE, 20 + WIDTH - 40)
        pygame.draw.line(screen, LINE_COLOR, start_v, end_v, LINE_WIDTH)

    # draw pieces
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            x = 20 + c * SQUARE
            y = 20 + r * SQUARE
            center = (x + SQUARE // 2, y + SQUARE // 2)
            if board[r, c] == 1:
                # X (player)
                pygame.draw.line(screen, X_COLOR, (x + PADDING, y + PADDING), (x + SQUARE - PADDING, y + SQUARE - PADDING), CROSS_WIDTH)
                pygame.draw.line(screen, X_COLOR, (x + PADDING, y + SQUARE - PADDING), (x + SQUARE - PADDING, y + PADDING), CROSS_WIDTH)
            elif board[r, c] == 2:
                pygame.draw.circle(screen, O_COLOR, center, CIRCLE_RADIUS, CIRCLE_WIDTH)

    # info panel
    pygame.draw.rect(screen, (12, 12, 18), (0, WIDTH, WIDTH, INFO_HEIGHT))
    title = TITLE_FONT.render("Tic-Tac-Toe — Unbeatable AI", True, TEXT)
    screen.blit(title, (24, WIDTH + 12))

    # Scores
    score_text = f"You: {scores['Player']}    AI: {scores['AI']}    Draws: {scores['Draws']}"
    scr = SCORE_FONT.render(score_text, True, TEXT)
    screen.blit(scr, (24, WIDTH + 52))

    # Restart button
    btn_rect = pygame.Rect(WIDTH - 160, WIDTH + 24, 130, 44)
    pygame.draw.rect(screen, BUTTON_BG, btn_rect, border_radius=8)
    btn_txt = INFO_FONT.render("Restart (R)", True, BUTTON_TEXT)
    screen.blit(btn_txt, (WIDTH - 140, WIDTH + 36))

    return btn_rect


def draw_hover(mouse_pos):
    mx, my = mouse_pos
    # only inside board area
    bx0, by0 = 20, 20
    if bx0 <= mx <= bx0 + WIDTH - 40 and by0 <= my <= by0 + WIDTH - 40:
        c = (mx - bx0) // SQUARE
        r = (my - by0) // SQUARE
        if board[r, c] == 0 and not game_over and player_turn:
            x = bx0 + c * SQUARE
            y = by0 + r * SQUARE
            rect = (x + 6, y + 6, SQUARE - 12, SQUARE - 12)
            s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
            s.fill((*HOVER_COLOR, 40))
            screen.blit(s, (rect[0], rect[1]))


def draw_winning_line(line, progress=1.0):
    # line: list of three (r,c) coords; draw a thick line from first center to last center
    (r1, c1), _, (r3, c3) = line
    x1 = 20 + c1 * SQUARE + SQUARE // 2
    y1 = 20 + r1 * SQUARE + SQUARE // 2
    x3 = 20 + c3 * SQUARE + SQUARE // 2
    y3 = 20 + r3 * SQUARE + SQUARE // 2
    # interpolate
    xi = x1 + (x3 - x1) * progress
    yi = y1 + (y3 - y1) * progress
    pygame.draw.line(screen, WIN_LINE, (x1, y1), (xi, yi), LINE_WIDTH + 6)

# ----- Game logic -----

def available_moves():
    return [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if board[r, c] == 0]


def is_draw():
    return not np.any(board == 0)


def check_winner(symbol):
    for line in WIN_LINES:
        vals = [board[r, c] for r, c in line]
        if vals[0] == symbol and vals[1] == symbol and vals[2] == symbol:
            return line
    return None

# Minimax with alpha-beta

def minimax(depth, alpha, beta, maximizing, ai_sym):
    # terminal
    w_ai = check_winner(ai_sym)
    w_op = check_winner(1 if ai_sym == 2 else 2)
    if w_ai:
        return 10 - depth, None
    if w_op:
        return depth - 10, None
    if is_draw():
        return 0, None

    if maximizing:
        best = -float('inf')
        best_move = None
        for r, c in available_moves():
            board[r, c] = ai_sym
            val, _ = minimax(depth + 1, alpha, beta, False, ai_sym)
            board[r, c] = 0
            if val > best:
                best = val
                best_move = (r, c)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best, best_move
    else:
        opp = 1 if ai_sym == 2 else 2
        best = float('inf')
        best_move = None
        for r, c in available_moves():
            board[r, c] = opp
            val, _ = minimax(depth + 1, alpha, beta, True, ai_sym)
            board[r, c] = 0
            if val < best:
                best = val
                best_move = (r, c)
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best, best_move


def ai_move():
    # first move optimization
    if len(available_moves()) == 9:
        choice = random.choice([(0, 0), (0, 2), (2, 0), (2, 2)])
        board[choice] = 2
        return
    _, mv = minimax(0, -float('inf'), float('inf'), True, 2)
    if mv:
        board[mv] = 2


def restart(first='random'):
    global board, player_turn, game_over, winner_line, winner_symbol, animate_progress
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    winner_line = None
    winner_symbol = None
    animate_progress = 0.0
    game_over = False
    if first == 'player':
        player_turn = True
    elif first == 'ai':
        player_turn = False
    else:
        player_turn = random.choice([True, False])

# start
restart(first='player')

# ----- Main Loop -----
running = True
while running:
    dt = clock.tick(FPS) / 1000.0
    btn_rect = draw_board()
    mouse = pygame.mouse.get_pos()
    draw_hover(mouse)

    # animate winning line if game over
    if game_over and winner_line is not None and animate_progress < 1.0:
        animate_progress = min(1.0, animate_progress + dt * 2.0)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                restart('player')
            if event.key == pygame.K_q:
                running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # restart button
            if btn_rect.collidepoint(event.pos):
                restart('player')
                continue

            if not game_over and player_turn:
                mx, my = event.pos
                if 20 <= mx <= 20 + WIDTH - 40 and 20 <= my <= 20 + WIDTH - 40:
                    c = (mx - 20) // SQUARE
                    r = (my - 20) // SQUARE
                    if board[r, c] == 0:
                        board[r, c] = 1
                        # check player win
                        wl = check_winner(1)
                        if wl is not None:
                            game_over = True
                            winner_line = wl
                            winner_symbol = 1
                            scores['Player'] += 1
                        elif is_draw():
                            game_over = True
                            winner_line = None
                            winner_symbol = None
                            scores['Draws'] += 1
                        else:
                            player_turn = False

    # AI turn (non-blocking)
    if not game_over and not player_turn:
        ai_move()
        wl = check_winner(2)
        if wl is not None:
            game_over = True
            winner_line = wl
            winner_symbol = 2
            scores['AI'] += 1
        elif is_draw():
            game_over = True
            winner_line = None
            winner_symbol = None
            scores['Draws'] += 1
        else:
            player_turn = True

    # draw winning animation on top
    if winner_line is not None:
        draw_winning_line(winner_line, animate_progress)

    # final overlay text when game over
    if game_over:
        overlay = pygame.Surface((WIDTH, WIDTH), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))
        screen.blit(overlay, (20, 20))
        msg = "Draw!" if winner_symbol is None else ("You win!" if winner_symbol == 1 else "AI wins!")
        text = TITLE_FONT.render(msg, True, TEXT)
        screen.blit(text, (WIDTH // 2 - text.get_width() // 2, WIDTH // 2 - 20))

    pygame.display.flip()

pygame.quit()
sys.exit()
