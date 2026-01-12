"""Морской бой (расширенная версия)

Включено:
- стартовое окно (меню)
- окно расстановки кораблей с перетаскиванием
- игровой экран со спрайтами
- финальный экран (результаты)
- несколько уровней сложности (easy/medium)
- улучшенный ИИ с разной логикой стрельбы
- хранение статистики в SQLite
- задержки для лучшего UX
"""

from __future__ import annotations

import random
import sqlite3
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Set
from dataclasses import dataclass
from enum import Enum

import arcade

# ----------------------------
# Настройки
# ----------------------------

# Размер поля
GRID_SIZE = 10

# Геометрия клетки/отступы
CELL_SIZE = 36
GRID_PADDING = 24

# Верхняя панель
TOP_BAR = 110

# Размер окна
SCREEN_WIDTH = GRID_PADDING * 2 + CELL_SIZE * GRID_SIZE * 2 + 140
SCREEN_HEIGHT = TOP_BAR + GRID_PADDING * 2 + CELL_SIZE * GRID_SIZE + 40
SCREEN_TITLE = "Морской бой (Arcade)"

# Координаты полей
PLAYER_GRID_ORIGIN = (GRID_PADDING, GRID_PADDING)
ENEMY_GRID_ORIGIN = (GRID_PADDING + CELL_SIZE * GRID_SIZE + 140, GRID_PADDING)

# Цвета
C_BG = arcade.color.DARK_MIDNIGHT_BLUE
C_PANEL = arcade.color.DARK_SLATE_GRAY
C_GRID = arcade.color.LIGHT_GRAY
C_TEXT = arcade.color.WHITE
C_SHIP = arcade.color.SEA_GREEN
C_HIT = arcade.color.ORANGE_RED
C_MISS = arcade.color.DODGER_BLUE
C_UNKNOWN = arcade.color.DIM_GRAY
C_DRAG_SHIPS = arcade.color.LIGHT_GRAY

# Папка с данными
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "stats.sqlite3"

# Стандартный набор кораблей: 1x4, 2x3, 3x2, 4x1
FLEET = [4, 3, 3, 2, 2, 2, 1, 1, 1, 1]


# ----------------------------
# Вспомогательные функции для рисования
# ----------------------------

def draw_rect_filled(center_x: float, center_y: float, width: float, height: float, color):
    """Рисует заполненный прямоугольник."""
    left = center_x - width / 2
    right = center_x + width / 2
    bottom = center_y - height / 2
    top = center_y + height / 2
    arcade.draw_lrbt_rectangle_filled(left, right, bottom, top, color)


def draw_rect_outline(center_x: float, center_y: float, width: float, height: float, color, border_width: float = 1):
    """Рисует контур прямоугольника."""
    left = center_x - width / 2
    right = center_x + width / 2
    bottom = center_y - height / 2
    top = center_y + height / 2
    arcade.draw_lrbt_rectangle_outline(left, right, bottom, top, color, border_width)


def draw_polygon_filled(center_x: float, center_y: float, width: float, height: float, color):
    """Рисует заполненный прямоугольник через полигон."""
    half_width = width / 2
    half_height = height / 2
    points = [
        (center_x - half_width, center_y - half_height),
        (center_x + half_width, center_y - half_height),
        (center_x + half_width, center_y + half_height),
        (center_x - half_width, center_y + half_height),
    ]
    arcade.draw_polygon_filled(points, color)


# ----------------------------
# Вспомогательные классы
# ----------------------------

class ShotResult:
    """Результат выстрела."""

    def __init__(self, hit: bool, already_shot: bool = False, sunk_ship: Optional[Ship] = None):
        self.hit = hit
        self.already_shot = already_shot
        self.sunk_ship = sunk_ship


@dataclass
class Ship:
    """Корабль с его параметрами."""
    length: int
    x: int
    y: int
    horizontal: bool
    hits: Set[Tuple[int, int]] = None

    def __post_init__(self):
        self.hits = set()

    @property
    def cells(self) -> List[Tuple[int, int]]:
        """Возвращает все клетки корабля."""
        cells = []
        for i in range(self.length):
            cx = self.x + i if self.horizontal else self.x
            cy = self.y if self.horizontal else self.y + i
            cells.append((cx, cy))
        return cells

    @property
    def is_sunk(self) -> bool:
        """Проверяет, потоплен ли корабль."""
        return len(self.hits) == self.length

    def hit(self, x: int, y: int) -> bool:
        """Регистрирует попадание в корабль."""
        if (x, y) in self.cells:
            self.hits.add((x, y))
            return True
        return False


class StatsStorage:
    """Хранилище статистики в SQLite."""

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._ensure_schema()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self):
        with self._connect() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS games (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts INTEGER NOT NULL,
                    difficulty TEXT NOT NULL,
                    result TEXT NOT NULL,
                    shots_total INTEGER NOT NULL,
                    hits INTEGER NOT NULL,
                    misses INTEGER NOT NULL,
                    duration_sec REAL NOT NULL
                )
            """)
            con.commit()

    def add_game(self, difficulty: str, result: str, shots_total: int,
                 hits: int, misses: int, duration_sec: float):
        with self._connect() as con:
            con.execute("""
                INSERT INTO games(ts, difficulty, result, shots_total, hits, misses, duration_sec)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (int(time.time()), difficulty, result, shots_total, hits, misses, duration_sec))
            con.commit()

    def aggregate(self) -> dict:
        with self._connect() as con:
            row = con.execute("""
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE WHEN result = 'lose' THEN 1 ELSE 0 END) AS loses,
                    SUM(shots_total) AS shots,
                    SUM(hits) AS hits
                FROM games
            """).fetchone()

        total, wins, loses, shots, hits = row
        total = total or 0
        wins = wins or 0
        loses = loses or 0
        shots = shots or 0
        hits = hits or 0
        acc = (hits / shots) if shots else 0.0
        winrate = (wins / total) if total else 0.0

        return {
            "total": total,
            "wins": wins,
            "loses": loses,
            "winrate": winrate,
            "accuracy": acc,
        }


class Board:
    """Игровое поле."""

    def __init__(self):
        self.ships: List[Ship] = []
        self.hits: Set[Tuple[int, int]] = set()
        self.misses: Set[Tuple[int, int]] = set()
        self.ship_cells: Set[Tuple[int, int]] = set()

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

    def can_place_ship(self, ship: Ship, ignore_ships: List[Ship] = None) -> bool:
        """Проверяет, можно ли поставить корабль."""
        ignore_ships = ignore_ships or []
        ignore_cells = set()
        for s in ignore_ships:
            ignore_cells.update(s.cells)
            # Добавляем клетки вокруг игнорируемых кораблей
            for x, y in s.cells:
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nx, ny = x + dx, y + dy
                        if self.in_bounds(nx, ny):
                            ignore_cells.add((nx, ny))

        for x, y in ship.cells:
            if not self.in_bounds(x, y):
                return False
            if (x, y) in ignore_cells:
                return False

        return True

    def place_ship(self, ship: Ship) -> bool:
        """Размещает корабль на поле."""
        if not self.can_place_ship(ship):
            return False

        self.ships.append(ship)
        self.ship_cells.update(ship.cells)
        return True

    def remove_ship(self, ship: Ship):
        """Удаляет корабль с поля."""
        if ship in self.ships:
            self.ships.remove(ship)
            self.ship_cells.difference_update(ship.cells)

    def shoot(self, x: int, y: int) -> ShotResult:
        """Обрабатывает выстрел."""
        if (x, y) in self.hits or (x, y) in self.misses:
            return ShotResult(hit=False, already_shot=True)

        sunk_ship = None
        for ship in self.ships:
            if ship.hit(x, y):
                self.hits.add((x, y))
                if ship.is_sunk:
                    sunk_ship = ship
                return ShotResult(hit=True, sunk_ship=sunk_ship)

        self.misses.add((x, y))
        return ShotResult(hit=False)

    def all_ships_destroyed(self) -> bool:
        return all(ship.is_sunk for ship in self.ships)

    def get_ship_at(self, x: int, y: int) -> Optional[Ship]:
        """Находит корабль по координатам."""
        for ship in self.ships:
            if (x, y) in ship.cells:
                return ship
        return None


def generate_compact_fleet(board: Board, seed: Optional[int] = None) -> bool:
    """
    Генерирует компактный флот для ИИ.
    Корабли стараются размещаться ближе друг к другу.
    """
    rnd = random.Random(seed)

    # Сортируем корабли по убыванию длины
    fleet_sorted = sorted(FLEET, reverse=True)

    for length in fleet_sorted:
        placed = False
        # Пробуем разместить корабль в разных позициях
        for _ in range(10000):
            horizontal = rnd.choice([True, False])

            # Выбираем позицию ближе к центру с большей вероятностью
            if rnd.random() < 0.7:
                x = rnd.randint(2, GRID_SIZE - length if horizontal else GRID_SIZE - 3)
                y = rnd.randint(2, GRID_SIZE - length if not horizontal else GRID_SIZE - 3)
            else:
                x = rnd.randint(0, GRID_SIZE - length if horizontal else GRID_SIZE - 1)
                y = rnd.randint(0, GRID_SIZE - length if not horizontal else GRID_SIZE - 1)

            ship = Ship(length, x, y, horizontal)

            if board.can_place_ship(ship, board.ships):
                board.place_ship(ship)
                placed = True
                break

        if not placed:
            return False

    return True


class ComputerAI:
    """Улучшенный ИИ для компьютера."""

    def __init__(self, difficulty: str):
        self.difficulty = difficulty
        self.last_shot_time = 0
        self.hunting_mode = False
        self.target_stack = []
        self.last_hit = None
        self.direction = None
        self.tried_directions = set()

    def next_shot(self, known_hits: Set[Tuple[int, int]],
                  known_misses: Set[Tuple[int, int]]) -> Tuple[int, int]:
        """Определяет следующий выстрел."""
        shot_set = known_hits | known_misses

        if self.difficulty == "medium" and known_hits:
            return self._hunting_shot(shot_set)

        # Легкий уровень или нет попаданий
        return self._random_shot(shot_set)

    def _random_shot(self, shot_set: Set[Tuple[int, int]]) -> Tuple[int, int]:
        """Случайный выстрел."""
        candidates = []
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if (x, y) not in shot_set:
                    candidates.append((x, y))

        if not candidates:
            return random.randrange(GRID_SIZE), random.randrange(GRID_SIZE)

        return random.choice(candidates)

    def _hunting_shot(self, shot_set: Set[Tuple[int, int]]) -> Tuple[int, int]:
        """Режим охоты - добивание поврежденного корабля."""
        # Если есть цель в стеке, стреляем по ней
        while self.target_stack:
            target = self.target_stack.pop()
            if target not in shot_set:
                return target

        # Ищем последнее попадание
        recent_hits = list(shot_set)
        if not recent_hits:
            return self._random_shot(shot_set)

        last_hit = recent_hits[-1]

        # Добавляем соседей для добивания
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        random.shuffle(directions)

        for dx, dy in directions:
            x, y = last_hit[0] + dx, last_hit[1] + dy
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and (x, y) not in shot_set:
                self.target_stack.append((x, y))

        if self.target_stack:
            return self.target_stack.pop()

        return self._random_shot(shot_set)

    def reset_hunting(self):
        """Сбрасывает режим охоты."""
        self.hunting_mode = False
        self.target_stack = []
        self.last_hit = None
        self.direction = None
        self.tried_directions = set()


# ----------------------------
# Sprite System (Комментарии для добавления спрайтов)
# ----------------------------

"""
Для добавления спрайтов выполните следующие шаги:

1. Создайте папку 'images' в той же директории, что и main.py

2. Подготовьте изображения (PNG с прозрачным фоном):

   Для кораблей (рекомендуемый размер 32x32 пикселя на клетку):
   - ship_small.png    (1 клетка)
   - ship_medium.png   (2 клетки)  
   - ship_large.png    (3 клетки)
   - ship_huge.png     (4 клетки)

   Для эффектов:
   - fire.png          (огонь при попадании, примерно 32x32)
   - miss.png          (крестик/круг для промаха, примерно 24x24)

3. Измените код ниже, раскомментировав строки с загрузкой спрайтов
   и заменив draw_polygon_filled на отрисовку спрайтов
"""


class GameSprites:
    """Класс для управления спрайтами игры."""

    def __init__(self):
        self.ship_sprites = {}
        self.effect_sprites = {}

        # Вместо реальной загрузки спрайтов, создаем цветные квадраты
        # Для реальных спрайтов раскомментируйте код ниже:

        """
        # Загрузка спрайтов кораблей
        self.ship_sprites = {
            1: arcade.Sprite("images/ship_small.png", scale=0.8),
            2: arcade.Sprite("images/ship_medium.png", scale=0.8),
            3: arcade.Sprite("images/ship_large.png", scale=0.8),
            4: arcade.Sprite("images/ship_huge.png", scale=0.8),
        }
        
        # Загрузка спрайтов эффектов
        self.effect_sprites = {
            'fire': arcade.Sprite("images/fire.png", scale=0.7),
            'miss': arcade.Sprite("images/miss.png", scale=0.6),
        }
        """

    def draw_ship(self, x: float, y: float, length: int, horizontal: bool):
        """Отрисовывает корабль."""
        # Вместо спрайта рисуем цветной прямоугольник
        color = [arcade.color.SEA_GREEN, arcade.color.DARK_GREEN,
                 arcade.color.GREEN, arcade.color.FOREST_GREEN][min(length - 1, 3)]

        if horizontal:
            width = CELL_SIZE * length - 4
            height = CELL_SIZE - 4
            draw_rect_filled(x, y, width, height, color)
            # Контур
            draw_rect_outline(x, y, width, height, arcade.color.BLACK, 1)
        else:
            width = CELL_SIZE - 4
            height = CELL_SIZE * length - 4
            draw_rect_filled(x, y, width, height, color)
            draw_rect_outline(x, y, width, height, arcade.color.BLACK, 1)

    def draw_fire(self, x: float, y: float):
        """Отрисовывает огонь при попадании."""
        # Временный эффект огня
        arcade.draw_circle_filled(x, y, CELL_SIZE // 3, arcade.color.ORANGE_RED)
        arcade.draw_circle_filled(x, y, CELL_SIZE // 4, arcade.color.YELLOW)

    def draw_miss(self, x: float, y: float):
        """Отрисовывает промах."""
        # Временный эффект промаха
        radius = CELL_SIZE // 4
        arcade.draw_circle_outline(x, y, radius, arcade.color.DODGER_BLUE, 3)
        arcade.draw_line(x - radius, y - radius, x + radius, y + radius,
                         arcade.color.DODGER_BLUE, 2)
        arcade.draw_line(x + radius, y - radius, x - radius, y + radius,
                         arcade.color.DODGER_BLUE, 2)


# ----------------------------
# Views (Экраны игры)
# ----------------------------

class BaseView(arcade.View):
    """Базовый экран."""

    def __init__(self):
        super().__init__()
        self.storage = StatsStorage(DB_PATH)
        self.sprites = GameSprites()


class MenuView(BaseView):
    """Главное меню."""

    def __init__(self):
        super().__init__()
        self.texts = []

    def on_show_view(self):
        arcade.set_background_color(C_BG)
        self.texts = [
            arcade.Text("МОРСКОЙ БОЙ", SCREEN_WIDTH / 2, SCREEN_HEIGHT - 60,
                        arcade.color.GOLD, 32, bold=True,
                        anchor_x="center", anchor_y="center"),
            arcade.Text("Нажмите 1 — Новая игра (Лёгкий)", SCREEN_WIDTH / 2,
                        SCREEN_HEIGHT / 2 + 40, C_TEXT, 18,
                        anchor_x="center", anchor_y="center"),
            arcade.Text("Нажмите 2 — Новая игра (Средний)", SCREEN_WIDTH / 2,
                        SCREEN_HEIGHT / 2 + 10, C_TEXT, 18,
                        anchor_x="center", anchor_y="center"),
            arcade.Text("Нажмите S — Статистика", SCREEN_WIDTH / 2,
                        SCREEN_HEIGHT / 2 - 20, C_TEXT, 18,
                        anchor_x="center", anchor_y="center"),
            arcade.Text("Нажмите Esc — Выход", SCREEN_WIDTH / 2,
                        SCREEN_HEIGHT / 2 - 50, C_TEXT, 18,
                        anchor_x="center", anchor_y="center"),
        ]

    def on_draw(self):
        self.clear()
        for text in self.texts:
            text.draw()

    def on_key_press(self, symbol: int, modifiers: int):
        if symbol == arcade.key.KEY_1:
            self.window.show_view(ShipPlacementView(difficulty="easy"))
        elif symbol == arcade.key.KEY_2:
            self.window.show_view(ShipPlacementView(difficulty="medium"))
        elif symbol == arcade.key.S:
            self.window.show_view(StatsView())
        elif symbol == arcade.key.ESCAPE:
            arcade.exit()


class StatsView(BaseView):
    """Экран статистики."""

    def __init__(self):
        super().__init__()
        self.texts = []

    def on_show_view(self):
        arcade.set_background_color(C_BG)
        self._rebuild_text()

    def _rebuild_text(self):
        agg = self.storage.aggregate()
        self.texts = [
            arcade.Text("СТАТИСТИКА", SCREEN_WIDTH / 2, SCREEN_HEIGHT - 60,
                        arcade.color.GOLD, 28, bold=True,
                        anchor_x="center", anchor_y="center"),
            arcade.Text(f"Игр всего: {agg['total']}", SCREEN_WIDTH / 2,
                        SCREEN_HEIGHT / 2 + 60, C_TEXT, 18,
                        anchor_x="center", anchor_y="center"),
            arcade.Text(f"Победы: {agg['wins']}", SCREEN_WIDTH / 2,
                        SCREEN_HEIGHT / 2 + 32, C_TEXT, 18,
                        anchor_x="center", anchor_y="center"),
            arcade.Text(f"Поражения: {agg['loses']}", SCREEN_WIDTH / 2,
                        SCREEN_HEIGHT / 2 + 4, C_TEXT, 18,
                        anchor_x="center", anchor_y="center"),
            arcade.Text(f"Процент побед: {agg['winrate'] * 100:.1f}%",
                        SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 - 24, C_TEXT, 18,
                        anchor_x="center", anchor_y="center"),
            arcade.Text(f"Точность: {agg['accuracy'] * 100:.1f}%",
                        SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 - 52, C_TEXT, 18,
                        anchor_x="center", anchor_y="center"),
            arcade.Text("Esc — назад", SCREEN_WIDTH / 2,
                        SCREEN_HEIGHT / 2 - 100, C_TEXT, 14,
                        anchor_x="center", anchor_y="center"),
        ]

    def on_draw(self):
        self.clear()
        for text in self.texts:
            text.draw()

    def on_key_press(self, symbol: int, modifiers: int):
        if symbol == arcade.key.ESCAPE:
            self.window.show_view(MenuView())


class ShipPlacementView(BaseView):
    """Экран расстановки кораблей."""

    def __init__(self, difficulty: str):
        super().__init__()
        self.difficulty = difficulty
        self.board = Board()

        # Корабли для расстановки
        self.available_ships = []
        for length in FLEET:
            self.available_ships.append({
                'length': length,
                'horizontal': True,
                'dragging': False,
                'drag_x': 0,
                'drag_y': 0,
                'placed': False
            })

        # Переменные для перетаскивания
        self.dragging_ship = None
        self.drag_offset_x = 0
        self.drag_offset_y = 0

        self.texts = []

    def on_show_view(self):
        arcade.set_background_color(C_BG)
        self._rebuild_text()

    def _rebuild_text(self):
        right_margin = SCREEN_WIDTH - 30
        left_margin = 20
        # Вычисляем dock_left здесь один раз, чтобы использовать в подписях
        dock_left = PLAYER_GRID_ORIGIN[0] + (GRID_SIZE * CELL_SIZE) + 200

        self.texts = [
            arcade.Text("R: Поворот | Backspace: Сброс | Enter: Старт",
                        right_margin, SCREEN_HEIGHT - 30, arcade.color.ASH_GREY, 10,
                        anchor_x="right"),

            arcade.Text("Space: Быстрая расстановка",
                        right_margin, SCREEN_HEIGHT - 45, arcade.color.ASH_GREY, 10,
                        anchor_x="right"),

            arcade.Text("РАССТАНОВКА КОРАБЛЕЙ", SCREEN_WIDTH / 2, SCREEN_HEIGHT - 50,
                        arcade.color.GOLD, 20, bold=True,
                        anchor_x="center", anchor_y="center"),

            arcade.Text("РАССТАВЬТЕ ВСЕ КОРАБЛИ",
                        right_margin, SCREEN_HEIGHT - 65, arcade.color.DIM_GRAY, 14, bold=True,
                        anchor_x="right"),

            arcade.Text("ESC - В МЕНЮ", right_margin, SCREEN_HEIGHT - 90,
                        arcade.color.ASH_GREY, 10, anchor_x="right"),

            arcade.Text("ВАШЕ ПОЛЕ", PLAYER_GRID_ORIGIN[0],
                        PLAYER_GRID_ORIGIN[1] + (GRID_SIZE * CELL_SIZE) + 10,
                        arcade.color.WHITE, 12, bold=True, anchor_x="left"),

            arcade.Text("ДОСТУПНЫЕ КОРАБЛИ",
                        PLAYER_GRID_ORIGIN[0] + (GRID_SIZE * CELL_SIZE) + 270,
                        PLAYER_GRID_ORIGIN[1] + (GRID_SIZE * CELL_SIZE) + 25,
                        arcade.color.WHITE, 12, bold=True, anchor_x="left"),

        ]

    def on_draw(self):
        self.clear()

        # Общая верхняя панель
        arcade.draw_lrbt_rectangle_filled(0, SCREEN_WIDTH, SCREEN_HEIGHT - 100, SCREEN_HEIGHT, (20, 30, 45))

        # 1. СИНХРОНИЗИРОВАННЫЙ РАСЧЕТ ДОКА
        dock_left = PLAYER_GRID_ORIGIN[0] + (GRID_SIZE * CELL_SIZE) + 200
        dock_right = dock_left + 300
        dock_bottom = PLAYER_GRID_ORIGIN[1] - 20
        dock_top = PLAYER_GRID_ORIGIN[1] + (GRID_SIZE * CELL_SIZE) + 40

        arcade.draw_lrbt_rectangle_filled(dock_left, dock_right, dock_bottom, dock_top, (20, 30, 45))
        arcade.draw_lrbt_rectangle_outline(dock_left, dock_right, dock_bottom, dock_top, (60, 70, 90), 2)

        # 2. ПОЛЕ
        self._draw_board(PLAYER_GRID_ORIGIN, self.board, show_ships=True)
        self._draw_grid_lines(PLAYER_GRID_ORIGIN)

        # 3. КОРАБЛИ (СИНХРОНИЗИРОВАНО)
        start_x = dock_left + 75
        start_y = dock_top - 60

        for i, ship_info in enumerate(self.available_ships):
            if ship_info['placed']: continue
            col, row = i % 2, i // 2
            current_x = start_x + (col * 140)
            current_y = start_y - (row * 75)

            draw_rect_filled(current_x, current_y, 120, 45, (40, 50, 65))
            if not ship_info['dragging']:
                self.sprites.draw_ship(current_x, current_y + 5, ship_info['length'], ship_info['horizontal'])
                arcade.draw_text(f"{ship_info['length']}-палубный", current_x, current_y - 25,
                                 arcade.color.ASH_GREY, 9, anchor_x="center")

        if self.dragging_ship:
            self.sprites.draw_ship(self.dragging_ship['drag_x'], self.dragging_ship['drag_y'],
                                   self.dragging_ship['length'], self.dragging_ship['horizontal'])
        for t in self.texts: t.draw()

    def _draw_board(self, origin: Tuple[int, int], board: Board, show_ships: bool):
        """Отрисовывает поле."""
        ox, oy = origin

        # Фон клеток
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                cx = ox + x * CELL_SIZE + CELL_SIZE / 2
                cy = oy + y * CELL_SIZE + CELL_SIZE / 2

                color = (30, 40, 55)
                if show_ships and (x, y) in board.ship_cells:
                    color = C_SHIP

                # Используем функцию из примеров
                draw_rect_filled(cx, cy, CELL_SIZE - 2, CELL_SIZE - 2, color)

    def _draw_grid_lines(self, origin: Tuple[int, int]):
        """Рисует линии сетки."""
        ox, oy = origin
        w = CELL_SIZE * GRID_SIZE
        h = CELL_SIZE * GRID_SIZE

        for i in range(GRID_SIZE + 1):
            x = ox + i * CELL_SIZE
            arcade.draw_line(x, oy, x, oy + h, C_GRID, 1)

        for j in range(GRID_SIZE + 1):
            y = oy + j * CELL_SIZE
            arcade.draw_line(ox, y, ox + w, y, C_GRID, 1)

    def _screen_to_grid(self, x: float, y: float, origin: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Преобразует экранные координаты в координаты клетки."""
        ox, oy = origin
        gx = int((x - ox) // CELL_SIZE)
        gy = int((y - oy) // CELL_SIZE)

        if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
            return gx, gy
        return None

    def on_mouse_press(self, x: float, y: float, button: int, modifiers: int):
        if button != arcade.MOUSE_BUTTON_LEFT: return

        if not self.dragging_ship:
            # ИСПРАВЛЕННЫЙ РАСЧЕТ: Теперь точно такой же, как в on_draw (+200)
            dock_left = PLAYER_GRID_ORIGIN[0] + (GRID_SIZE * CELL_SIZE) + 200
            dock_top = PLAYER_GRID_ORIGIN[1] + (GRID_SIZE * CELL_SIZE) + 40
            start_x = dock_left + 75
            start_y = dock_top - 60

            for i, ship_info in enumerate(self.available_ships):
                if ship_info['placed']: continue
                col, row = i % 2, i // 2
                current_x = start_x + (col * 140)
                current_y = start_y - (row * 75) + 5

                length = ship_info['length']
                w, h = (length * CELL_SIZE, CELL_SIZE) if ship_info['horizontal'] else (CELL_SIZE, length * CELL_SIZE)

                if (current_x - w / 2 <= x <= current_x + w / 2 and
                        current_y - h / 2 <= y <= current_y + h / 2):
                    self.dragging_ship = ship_info
                    self.drag_offset_x = x - current_x
                    self.drag_offset_y = y - current_y
                    ship_info['dragging'] = True
                    return

        # Установка (с фиксом смещения из предыдущего шага)
        if self.dragging_ship:
            # Используем смещение, чтобы корабль вставал ровно под курсор
            target_x = x - self.drag_offset_x
            target_y = y - self.drag_offset_y
            grid_pos = self._screen_to_grid(target_x, target_y, PLAYER_GRID_ORIGIN)

            if grid_pos:
                gx, gy = grid_pos
                ship = Ship(self.dragging_ship['length'], gx, gy, self.dragging_ship['horizontal'])
                if self.board.can_place_ship(ship, self.board.ships):
                    self.board.place_ship(ship)
                    self.dragging_ship['placed'] = True
                    if all(s['placed'] for s in self.available_ships):
                        self.texts[2].text = "ГОТОВ К БОЮ! НАЖМИ ENTER"  # Индекс текста готовности
                        self.texts[2].color = arcade.color.NEON_GREEN

            self.dragging_ship['dragging'] = False
            self.dragging_ship = None

    def _random_placement(self):
        """Случайно расставляет корабли на поле."""
        self.board = Board()  # Сбрасываем поле полностью

        # Сбрасываем статус всех кораблей в UI
        for ship_info in self.available_ships:
            ship_info['placed'] = False

        # Пытаемся расставить каждый корабль
        for ship_info in self.available_ships:
            placed = False
            attempts = 0
            while not placed and attempts < 1000:  # Защита от вечного цикла
                attempts += 1
                length = ship_info['length']
                horizontal = random.choice([True, False])

                # Вычисляем максимальные координаты, чтобы корабль не вылез за границы
                max_x = GRID_SIZE - length if horizontal else GRID_SIZE - 1
                max_y = GRID_SIZE - 1 if horizontal else GRID_SIZE - length

                gx = random.randint(0, max_x)
                gy = random.randint(0, max_y)

                new_ship = Ship(length, gx, gy, horizontal)

                if self.board.can_place_ship(new_ship, self.board.ships):
                    self.board.place_ship(new_ship)

                    # Обновляем инфо для UI
                    ship_info['placed'] = True
                    ship_info['horizontal'] = horizontal
                    placed = True

        # Обновляем текст готовности
        self._check_ready_state()

    def _check_ready_state(self):
        """Проверяет, все ли расставлено, и меняет цвет текста."""
        if all(s['placed'] for s in self.available_ships):
            self.texts[2].text = "ГОТОВ К БОЮ! НАЖМИ ENTER"
            self.texts[2].color = arcade.color.NEON_GREEN
        else:
            self.texts[2].text = "РАССТАВЬТЕ ВСЕ КОРАБЛИ"
            self.texts[2].color = arcade.color.DIM_GRAY

    def on_mouse_motion(self, x: float, y: float, dx: float, dy: float):
        """Обработка движения мыши при перетаскивании."""
        if self.dragging_ship:
            self.dragging_ship['drag_x'] = x - self.drag_offset_x
            self.dragging_ship['drag_y'] = y - self.drag_offset_y

    def on_mouse_release(self, x: float, y: float, button: int, modifiers: int):
        """Обработка отпускания мыши."""
        if button == arcade.MOUSE_BUTTON_LEFT and self.dragging_ship:
            self.on_mouse_press(x, y, button, modifiers)

    def on_key_press(self, symbol: int, modifiers: int):
        """Обработка нажатия клавиш."""
        if symbol == arcade.key.R and self.dragging_ship:
            # Поворот корабля
            self.dragging_ship['horizontal'] = not self.dragging_ship['horizontal']

        elif symbol == arcade.key.SPACE:
            # Если что-то тащим — отменяем перетаскивание перед рандомом
            if self.dragging_ship:
                self.dragging_ship['dragging'] = False
                self.dragging_ship = None
            self._random_placement()

        elif symbol == arcade.key.BACKSPACE:
            # Удаление последнего корабля
            if self.board.ships:
                last_ship = self.board.ships[-1]
                self.board.remove_ship(last_ship)

                for ship_info in self.available_ships:
                    if ship_info['placed'] and ship_info['length'] == last_ship.length:
                        ship_info['placed'] = False
                        break

                # Возвращаем цвет текста, если удалили корабль после завершения расстановки
                self.texts[-2].color = arcade.color.LIGHT_GREEN
                self.texts[-2].text = "Enter - начать игру"


        elif symbol == arcade.key.ENTER:
            # Начало игры
            if all(ship['placed'] for ship in self.available_ships):
                self.window.show_view(GameView(self.difficulty, self.board))

        elif symbol == arcade.key.ESCAPE:
            self.window.show_view(MenuView())


class GameView(BaseView):
    """Основной игровой экран."""

    def __init__(self, difficulty: str, player_board: Board = None):
        super().__init__()
        self.difficulty = difficulty
        self.ai = ComputerAI(difficulty)

        # Игровые поля
        self.player_board = player_board or Board()
        self.enemy_board = Board()

        # Генерируем флот для ИИ
        if not generate_compact_fleet(self.enemy_board):
            # Если не удалось компактно разместить, используем случайное размещение
            self._random_place_fleet(self.enemy_board)

        # Игровые переменные
        self.start_time = time.time()
        self.player_turn = True
        self.message = "Ваш ход: кликните по правому полю."
        self.last_player_shot_time = 0
        self.ai_delay = 0
        self.game_over = False
        self.victory_delay = 0

        # Звуки
        self.Miss_SOUND = arcade.load_sound("sounds/Miss.mp3")
        self.Popal_SOUND = arcade.load_sound("sounds/PopalNew.mp3")

        # Статистика
        self.shots_total = 0
        self.hits = 0
        self.misses = 0

        # Тексты
        self.texts = []

    def _random_place_fleet(self, board: Board):
        """Случайное размещение флота."""
        for length in FLEET:
            placed = False
            for _ in range(10000):
                horizontal = random.choice([True, False])
                x = random.randrange(GRID_SIZE - length + 1) if horizontal else random.randrange(GRID_SIZE)
                y = random.randrange(GRID_SIZE) if horizontal else random.randrange(GRID_SIZE - length + 1)

                ship = Ship(length, x, y, horizontal)
                if board.can_place_ship(ship, board.ships):
                    board.place_ship(ship)
                    placed = True
                    break

            if not placed:
                raise RuntimeError("Не удалось расставить корабли")

    def on_show_view(self):
        arcade.set_background_color(C_BG)
        self._rebuild_text()

    def _rebuild_text(self):
        # Чтобы текст не лежал на поле, привязываем его к SCREEN_HEIGHT
        self.texts = [
            arcade.Text("МОРСКОЙ БОЙ", SCREEN_WIDTH / 2, SCREEN_HEIGHT - 30,
                        arcade.color.GOLD, 20, bold=True,
                        anchor_x="center", anchor_y="center"),

            # Подписи полей теперь учитывают высоту сетки
            arcade.Text("Ваше поле", PLAYER_GRID_ORIGIN[0],
                        PLAYER_GRID_ORIGIN[1] + (GRID_SIZE * CELL_SIZE) + 10,
                        C_TEXT, 12, bold=True),

            arcade.Text("Поле противника", ENEMY_GRID_ORIGIN[0],
                        ENEMY_GRID_ORIGIN[1] + (GRID_SIZE * CELL_SIZE) + 10,
                        C_TEXT, 12, bold=True),
        ]

    def on_draw(self):
        self.clear()

        # Верхняя панель (фон для текстов)
        arcade.draw_lrbt_rectangle_filled(0, SCREEN_WIDTH,
                                          SCREEN_HEIGHT - 100, SCREEN_HEIGHT,
                                          (20, 30, 45))

        # Тексты
        for text in self.texts:
            text.draw()

        # Статус игры (поднимаем выше, чтобы не наезжало на поле)
        turn_text = "Ваш ход" if self.player_turn else "Ход компьютера"
        status_text = f"Сложность: {self.difficulty} | {turn_text}"
        arcade.draw_text(status_text, SCREEN_WIDTH / 2, SCREEN_HEIGHT - 60,
                         C_TEXT, 13, anchor_x="center")
        arcade.draw_text(self.message, SCREEN_WIDTH / 2, SCREEN_HEIGHT - 80,
                         arcade.color.LIGHT_GRAY, 11, anchor_x="center")

        # Поля
        self._draw_board(PLAYER_GRID_ORIGIN, self.player_board, show_ships=True)
        self._draw_board(ENEMY_GRID_ORIGIN, self.enemy_board, show_ships=False)

        # Сетки
        self._draw_grid_lines(PLAYER_GRID_ORIGIN)
        self._draw_grid_lines(ENEMY_GRID_ORIGIN)

    def _draw_board(self, origin: Tuple[int, int], board: Board, show_ships: bool):
        """Отрисовывает игровое поле со спрайтами: слои по порядку."""
        ox, oy = origin

        # СЛОЙ 1: Фон клеток
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                cx = ox + x * CELL_SIZE + CELL_SIZE / 2
                cy = oy + y * CELL_SIZE + CELL_SIZE / 2
                # Темная подложка клетки
                draw_rect_filled(cx, cy, CELL_SIZE - 2, CELL_SIZE - 2, (30, 40, 55))

        # СЛОЙ 2: Корабли (свои или потопленные чужие)
        if show_ships:
            for ship in board.ships:
                length = ship.length
                if ship.horizontal:
                    center_x = ox + (ship.x + length / 2) * CELL_SIZE
                    center_y = oy + (ship.y + 0.5) * CELL_SIZE
                else:
                    center_x = ox + (ship.x + 0.5) * CELL_SIZE
                    center_y = oy + (ship.y + length / 2) * CELL_SIZE

                self.sprites.draw_ship(center_x, center_y, length, ship.horizontal)
        else:
            # Для поля врага рисуем только полностью потопленные корабли
            for ship in board.ships:
                if ship.is_sunk:
                    length = ship.length
                    if ship.horizontal:
                        center_x = ox + (ship.x + length / 2) * CELL_SIZE
                        center_y = oy + (ship.y + 0.5) * CELL_SIZE
                    else:
                        center_x = ox + (ship.x + 0.5) * CELL_SIZE
                        center_y = oy + (ship.y + length / 2) * CELL_SIZE
                    # Рисуем контур или полупрозрачный корабль
                    self.sprites.draw_ship(center_x, center_y, length, ship.horizontal)
                    # Дополнительно выделим красным, что он "всё"
                    draw_rect_outline(center_x, center_y,
                                      (length * CELL_SIZE if ship.horizontal else CELL_SIZE) - 2,
                                      (CELL_SIZE if ship.horizontal else length * CELL_SIZE) - 2,
                                      arcade.color.RED, 2)

        # СЛОЙ 3: Эффекты (Промахи и Попадания) — ВСЕГДА ПОВЕРХ ВСЕГО
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                cx = ox + x * CELL_SIZE + CELL_SIZE / 2
                cy = oy + y * CELL_SIZE + CELL_SIZE / 2

                # Рисуем промах (синяя точка/крестик)
                if (x, y) in board.misses:
                    self.sprites.draw_miss(cx, cy)

                # Рисуем огонь/взрыв (рисуется поверх корабля)
                if (x, y) in board.hits:
                    self.sprites.draw_fire(cx, cy)

    def _draw_grid_lines(self, origin: Tuple[int, int]):
        """Рисует линии сетки."""
        ox, oy = origin
        w = CELL_SIZE * GRID_SIZE
        h = CELL_SIZE * GRID_SIZE

        for i in range(GRID_SIZE + 1):
            x = ox + i * CELL_SIZE
            arcade.draw_line(x, oy, x, oy + h, C_GRID, 1)

        for j in range(GRID_SIZE + 1):
            y = oy + j * CELL_SIZE
            arcade.draw_line(ox, y, ox + w, y, C_GRID, 1)

    def _screen_to_grid(self, x: float, y: float, origin: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Преобразует экранные координаты в координаты клетки."""
        ox, oy = origin
        gx = int((x - ox) // CELL_SIZE)
        gy = int((y - oy) // CELL_SIZE)

        if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
            return gx, gy
        return None

    def on_mouse_press(self, x: float, y: float, button: int, modifiers: int):
        """Обработка выстрела игрока."""
        if self.game_over or not self.player_turn or button != arcade.MOUSE_BUTTON_LEFT:
            return

        grid_pos = self._screen_to_grid(x, y, ENEMY_GRID_ORIGIN)
        if not grid_pos:
            return

        gx, gy = grid_pos
        shot_time = time.time()

        # Обрабатываем выстрел
        result = self.enemy_board.shoot(gx, gy)

        if result.already_shot:
            self.message = "Сюда уже стреляли!"
            return

        self.shots_total += 1
        self.last_player_shot_time = shot_time

        if result.hit:
            self.hits += 1
            self.message = "Попадание!"
            self.ship_playback = arcade.play_sound(self.Popal_SOUND)

            if result.sunk_ship:
                self.message = f"Корабль потоплен! ({result.sunk_ship.length}-палубный)"

            # Проверяем победу
            if self.enemy_board.all_ships_destroyed():
                self.message = "Все корабли противника уничтожены!"
                self.game_over = True
                self.victory_delay = 2.0  # Задержка 2 секунды перед финальным экраном
            else:
                # При попадании игрок ходит снова
                self.player_turn = True
        else:
            self.misses += 1
            self.message = "Промах!"
            self.ship_playback = arcade.play_sound(self.Miss_SOUND)
            self.player_turn = False
            self.ai_delay = 1.5

    def on_update(self, delta_time: float):
        """Обновление игровой логики."""
        if self.game_over:
            self.victory_delay -= delta_time
            if self.victory_delay <= 0:
                self._finish_game(won=True)
            return

        if not self.player_turn and not self.game_over:
            if self.ai_delay > 0:
                self.ai_delay -= delta_time
            else:
                self._ai_turn()

    def _ai_turn(self):
        """Ход компьютера."""
        # Получаем координаты для выстрела
        x, y = self.ai.next_shot(self.player_board.hits, self.player_board.misses)

        # Выстрел
        result = self.player_board.shoot(x, y)

        if result.hit:
            self.message = "Компьютер попал!"
            self.ship_playback = arcade.play_sound(self.Popal_SOUND)

            if result.sunk_ship:
                self.message = f"Компьютер потопил ваш {result.sunk_ship.length}-палубный корабль!"
                self.ai.reset_hunting()  # Сбрасываем режим охоты после потопления

            # Проверяем поражение
            if self.player_board.all_ships_destroyed():
                self.message = "Все ваши корабли уничтожены!"
                self.game_over = True
                self.victory_delay = 2.0  # Задержка 2 секунды
            else:
                # При попадании ИИ ходит снова (с задержкой)
                self.ai_delay = 1.0  # Пауза 1 секунда между выстрелами ИИ
        else:
            self.message = "Компьютер промахнулся. Ваш ход."
            self.ship_playback = arcade.play_sound(self.Miss_SOUND)
            self.player_turn = True

    def _finish_game(self, won: bool):
        """Завершение игры."""
        duration = time.time() - self.start_time
        self.window.show_view(
            ResultView(
                difficulty=self.difficulty,
                won=won,
                shots=self.shots_total,
                hits=self.hits,
                misses=self.misses,
                duration=duration,
            )
        )

    def on_key_press(self, symbol: int, modifiers: int):
        if symbol == arcade.key.ESCAPE:
            self.window.show_view(MenuView())


class ResultView(BaseView):
    """Экран результатов."""

    def __init__(self, *, difficulty: str, won: bool, shots: int,
                 hits: int, misses: int, duration: float):
        super().__init__()
        self.difficulty = difficulty
        self.won = won
        self.shots = shots
        self.hits = hits
        self.misses = misses
        self.duration = duration

        # Сохраняем статистику
        self.storage.add_game(
            difficulty=difficulty,
            result="win" if won else "lose",
            shots_total=shots,
            hits=hits,
            misses=misses,
            duration_sec=duration,
        )

        self.texts = []

    def on_show_view(self):
        arcade.set_background_color(C_BG)
        self._rebuild_text()

    def _rebuild_text(self):
        title = "ПОБЕДА!" if self.won else "ПОРАЖЕНИЕ"
        title_color = arcade.color.GOLD if self.won else arcade.color.RED

        self.texts = [
            arcade.Text(title, SCREEN_WIDTH / 2, SCREEN_HEIGHT - 80,
                        title_color, 32, bold=True,
                        anchor_x="center", anchor_y="center"),
            arcade.Text(f"Сложность: {self.difficulty}", SCREEN_WIDTH / 2,
                        SCREEN_HEIGHT - 140, C_TEXT, 18,
                        anchor_x="center", anchor_y="center"),
            arcade.Text(f"Выстрелов: {self.shots}", SCREEN_WIDTH / 2,
                        SCREEN_HEIGHT - 170, C_TEXT, 18,
                        anchor_x="center", anchor_y="center"),
            arcade.Text(f"Попаданий: {self.hits}", SCREEN_WIDTH / 2,
                        SCREEN_HEIGHT - 200, C_TEXT, 18,
                        anchor_x="center", anchor_y="center"),
            arcade.Text(f"Точность: {self.hits / self.shots * 100:.1f}%" if self.shots > 0 else "Точность: 0%",
                        SCREEN_WIDTH / 2, SCREEN_HEIGHT - 230, C_TEXT, 18,
                        anchor_x="center", anchor_y="center"),
            arcade.Text(f"Время: {self.duration:.1f} сек", SCREEN_WIDTH / 2,
                        SCREEN_HEIGHT - 260, C_TEXT, 18,
                        anchor_x="center", anchor_y="center"),
            arcade.Text("Enter - сыграть ещё", SCREEN_WIDTH / 2,
                        SCREEN_HEIGHT - 320, arcade.color.LIGHT_GREEN, 20,
                        anchor_x="center", anchor_y="center"),
            arcade.Text("Esc - главное меню", SCREEN_WIDTH / 2,
                        SCREEN_HEIGHT - 350, C_TEXT, 16,
                        anchor_x="center", anchor_y="center"),
        ]

    def on_draw(self):
        self.clear()
        for text in self.texts:
            text.draw()

    def on_key_press(self, symbol: int, modifiers: int):
        if symbol == arcade.key.ENTER:
            self.window.show_view(ShipPlacementView(difficulty=self.difficulty))
        elif symbol == arcade.key.ESCAPE:
            self.window.show_view(MenuView())


# ----------------------------
# Главная функция
# ----------------------------

def main():
    """Точка входа в игру."""
    window = arcade.Window(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE, update_rate=1 / 60)
    window.show_view(MenuView())
    arcade.run()


if __name__ == "__main__":
    main()
