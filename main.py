"""Морской бой (частичная реализация)

Включено:
- стартовое окно (меню)
- игровой экран
- финальный экран (результаты)
- несколько уровней сложности (easy/medium)
- камера (в упрощённой версии статическая)
- хранение статистики в SQLite

"""

from __future__ import annotations

import random
import sqlite3
import time
from pathlib import Path
from typing import List, Optional, Tuple

import arcade

# ----------------------------
# Настройки
# ----------------------------

# Размер поля 
GRID_SIZE = 10

# Геометрия клетки/отступы (используется и при отрисовке, и при преобразовании координат мыши)
CELL_SIZE = 36
GRID_PADDING = 24

# Верхняя панель 
TOP_BAR = 110

# Размер окна рассчитан так, чтобы поместились два поля, разделитель и верхняя панель
SCREEN_WIDTH = GRID_PADDING * 2 + CELL_SIZE * GRID_SIZE * 2 + 140
SCREEN_HEIGHT = TOP_BAR + GRID_PADDING * 2 + CELL_SIZE * GRID_SIZE + 40
SCREEN_TITLE = "Морской бой (Arcade) — демо часть"

# Координаты нижнего-левого угла каждого поля на экране
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

# Папка с данными/БД. SQLite файл создаётся автоматически при первом запуске.
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "stats.sqlite3"


# ----------------------------
# Хранилище статистики (SQLite)
# ----------------------------

class StatsStorage:
    """обёртка над SQLite для хранения статистики.
    Храним каждую сыгранную партию отдельной строкой, а на экране статистики
    показываем агрегаты (победы/поражения/точность).
    """

    def __init__(self, db_path: Path):
        # Создаём директорию под БД, если её ещё нет (например, при первом запуске).
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        # На старте гарантируем наличие таблицы.
        self._ensure_schema()

    def _connect(self):
        # Каждый вызов возвращает новое соединение.
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self):
        with self._connect() as con:
            con.execute(
                """
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
                """
            )
            con.commit()

    def add_game(
        self,
        difficulty: str,
        result: str,
        shots_total: int,
        hits: int,
        misses: int,
        duration_sec: float,
    ):
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO games(ts, difficulty, result, shots_total, hits, misses, duration_sec)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (int(time.time()), difficulty, result, shots_total, hits, misses, duration_sec),
            )
            con.commit()

    def aggregate(self) -> dict:
        with self._connect() as con:
            row = con.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE WHEN result = 'lose' THEN 1 ELSE 0 END) AS loses,
                    SUM(shots_total) AS shots,
                    SUM(hits) AS hits
                FROM games
                """
            ).fetchone()
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


# ----------------------------
# Игровая модель
# ----------------------------

# Состояние клетки для отрисовки/логики.
UNKNOWN = 0  # по этой клетке ещё не стреляли
MISS = 1     # был выстрел и промах
HIT = 2      # был выстрел и попадание
SHIP = 3     # используется только на поле игрока (если бы мы хранили корабли в grid)


class ShotResult:
    """Результат выстрела.

    hit=True   -> попали в корабль
    already_shot=True -> в клетку уже стреляли (ход не тратим, просто сообщение)
    """

    def __init__(self, hit: bool, already_shot: bool = False):
        self.hit = hit
        self.already_shot = already_shot


def neighbors8(x: int, y: int):
    """8-соседи клетки (включая диагонали).

    Используется при расстановке кораблей, чтобы запретить касания кораблей
    (правило «между кораблями должна быть хотя бы одна клетка»).
    """

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            yield x + dx, y + dy


class Board:
    """Игровое поле.

    Для простоты состояние храним набором координат:
    - ship_cells: где стоят корабли
    - hits/misses: история выстрелов

    в текущей версии фактически всё основано на множествах.
    """

    def __init__(self):
        # Сетка grid[y][x]. Сейчас используется ограниченно, но полезна как понятная модель.
        self.grid = [[UNKNOWN for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

        # Корабли и история выстрелов: множества дают O(1) на проверки попаданий/повторов.
        self.ship_cells: set[tuple[int, int]] = set()
        self.hits: set[tuple[int, int]] = set()
        self.misses: set[tuple[int, int]] = set()

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

    def can_place_ship(self, cells: List[Tuple[int, int]], forbid_touching: bool = True) -> bool:
        """Проверяет, можно ли поставить корабль в указанные клетки.

        forbid_touching=True запрещает касание кораблей даже по диагонали.
        """
        for (x, y) in cells:
            if not self.in_bounds(x, y):
                return False
            # Нельзя поставить корабль на уже занятое место.
            if (x, y) in self.ship_cells:
                return False
            if forbid_touching:
                # Нельзя поставить рядом (включая диагонали) с другим кораблём.
                for nx, ny in neighbors8(x, y):
                    if (nx, ny) in self.ship_cells:
                        return False
        return True

    def place_ship(self, length: int, x: int, y: int, horizontal: bool, forbid_touching: bool = True) -> bool:
        """Пытается разместить корабль длины length начиная с (x,y)."""
        cells = []
        for i in range(length):
            cx = x + i if horizontal else x
            cy = y if horizontal else y + i
            cells.append((cx, cy))

        if not self.can_place_ship(cells, forbid_touching=forbid_touching):
            return False

        for c in cells:
            self.ship_cells.add(c)
        return True

    def shoot(self, x: int, y: int) -> ShotResult:
        """Обрабатывает выстрел по клетке.

        В этой версии мы не различаем «ранил/убил» и не отмечаем контуры кораблей.
        Храним только факт попадания/промаха.
        """
        # Повторный выстрел не меняет состояние.
        if (x, y) in self.hits or (x, y) in self.misses:
            return ShotResult(hit=False, already_shot=True)

        if (x, y) in self.ship_cells:
            self.hits.add((x, y))
            return ShotResult(hit=True)

        self.misses.add((x, y))
        return ShotResult(hit=False)

    def all_ships_destroyed(self) -> bool:
        return len(self.hits) == len(self.ship_cells)


# Стандартный набор кораблей: 1x4, 2x3, 3x2, 4x1
FLEET = [4, 3, 3, 2, 2, 2, 1, 1, 1, 1]


def auto_place_fleet(board: Board, *, seed: Optional[int] = None):
    """Случайно расставляет флот на поле.

    Используем перебор с большим числом попыток: для маленького поля и фиксированного
    флота это работает надёжно и просто.

    """
    rnd = random.Random(seed)
    for length in FLEET:
        # Много попыток на каждый корабль, чтобы избежать редких тупиков.
        for _ in range(10_000):
            horizontal = rnd.choice([True, False])
            x = rnd.randrange(GRID_SIZE)
            y = rnd.randrange(GRID_SIZE)
            if board.place_ship(length, x, y, horizontal, forbid_touching=True):
                break
        else:
            # Если за 10k попыток не получилось, значит мы «заклинили» из-за предыдущих расстановок
            raise RuntimeError("Не удалось расставить корабли")


# ----------------------------
# Логика компьютера 
# ----------------------------

class ComputerAI:
    """Примитивный AI.

    easy:
      - просто стреляет в случайные непроверенные клетки.

    medium:
      - если есть попадания, пытается «добивать» корабль: добавляет в очередь 4
        ортогональных соседа вокруг последних попаданий.

    Это не честный морской бой с вероятностными картами.
    """

    def __init__(self, difficulty: str):
        self.difficulty = difficulty
        # Очередь «кандидатов» на выстрел вокруг попаданий.
        self._candidates: List[Tuple[int, int]] = []

    def next_shot(self, known_hits: set[tuple[int, int]], known_misses: set[tuple[int, int]]) -> Tuple[int, int]:
        # Все клетки, по которым уже стреляли.
        shot_set = known_hits | known_misses

        # Средний: простое «добивание» — после попаданий пробуем 4 соседние клетки.
        if self.difficulty == "medium":
            # Если очередь пуста — пополняем её соседями вокруг нескольких последних попаданий.
            if not self._candidates:
                for hx, hy in list(known_hits)[-5:]:
                    for (nx, ny) in ((hx + 1, hy), (hx - 1, hy), (hx, hy + 1), (hx, hy - 1)):
                        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and (nx, ny) not in shot_set:
                            self._candidates.append((nx, ny))

            # Берём первого кандидата.
            while self._candidates:
                c = self._candidates.pop(0)
                if c not in shot_set:
                    return c

        # Лёгкий/резервный вариант: случайный выстрел.
        # Генерируем до тех пор, пока не попадём в ещё не прострелянную клетку.
        while True:
            x = random.randrange(GRID_SIZE)
            y = random.randrange(GRID_SIZE)
            if (x, y) not in shot_set:
                return x, y


class BaseView(arcade.View):
    """Базовый экран.
    """

    def __init__(self):
        super().__init__()
        self.storage = StatsStorage(DB_PATH)


class MenuView(BaseView):
    def __init__(self):
        super().__init__()
        self.selected_difficulty = "easy"
        self._title_text: arcade.Text | None = None
        self._menu_text: arcade.Text | None = None

    def on_show_view(self):
        arcade.set_background_color(C_BG)
        self._title_text = arcade.Text(
            "МОРСКОЙ БОЙ",
            SCREEN_WIDTH / 2,
            SCREEN_HEIGHT - 60,
            C_TEXT,
            32,
            anchor_x="center",
            anchor_y="center",
            bold=True,
        )
        self._menu_text = None
        self._menu_lines = [
            arcade.Text(
                "Нажмите 1 — Новая игра (Лёгкий)",
                SCREEN_WIDTH / 2,
                SCREEN_HEIGHT / 2 + 40,
                C_TEXT,
                18,
                anchor_x="center",
                anchor_y="center",
            ),
            arcade.Text(
                "Нажмите 2 — Новая игра (Средний)",
                SCREEN_WIDTH / 2,
                SCREEN_HEIGHT / 2 + 10,
                C_TEXT,
                18,
                anchor_x="center",
                anchor_y="center",
            ),
            arcade.Text(
                "Нажмите S — Статистика",
                SCREEN_WIDTH / 2,
                SCREEN_HEIGHT / 2 - 20,
                C_TEXT,
                18,
                anchor_x="center",
                anchor_y="center",
            ),
            arcade.Text(
                "Нажмите Esc — Выход",
                SCREEN_WIDTH / 2,
                SCREEN_HEIGHT / 2 - 50,
                C_TEXT,
                18,
                anchor_x="center",
                anchor_y="center",
            ),
        ]

    def on_draw(self):
        self.clear()
        if self._title_text:
            self._title_text.draw()
        if getattr(self, "_menu_lines", None):
            for t in self._menu_lines:
                t.draw()
        elif self._menu_text:
            self._menu_text.draw()

    def on_key_press(self, symbol: int, modifiers: int):
        if symbol == arcade.key.KEY_1:
            self.window.show_view(GameView(difficulty="easy"))
        elif symbol == arcade.key.KEY_2:
            self.window.show_view(GameView(difficulty="medium"))
        elif symbol == arcade.key.S:
            self.window.show_view(StatsView())
        elif symbol == arcade.key.ESCAPE:
            arcade.exit()


class StatsView(BaseView):
    def __init__(self):
        super().__init__()
        self._title_text: arcade.Text | None = None
        self._lines: list[arcade.Text] = []

    def on_show_view(self):
        arcade.set_background_color(C_BG)
        self._rebuild_text()

    def _rebuild_text(self):
        agg = self.storage.aggregate()
        self._title_text = arcade.Text(
            "СТАТИСТИКА",
            SCREEN_WIDTH / 2,
            SCREEN_HEIGHT - 60,
            C_TEXT,
            28,
            anchor_x="center",
            anchor_y="center",
            bold=True,
        )

        # Поэтому делаем строки отдельными Text-объектами.
        self._lines = []
        y0 = SCREEN_HEIGHT / 2 + 60
        step = 28
        items = [
            f"Игр всего: {agg['total']}",
            f"Победы: {agg['wins']}",
            f"Поражения: {agg['loses']}",
            f"Процент побед: {agg['winrate']*100:.1f}%",
            f"Средняя точность: {agg['accuracy']*100:.1f}%",
            "",
            "Esc — назад",
        ]
        for i, line in enumerate(items):
            self._lines.append(
                arcade.Text(
                    line,
                    SCREEN_WIDTH / 2,
                    y0 - i * step,
                    C_TEXT,
                    18,
                    anchor_x="center",
                    anchor_y="center",
                    align="center",
                )
            )

    def on_draw(self):
        self.clear()
        if self._title_text:
            self._title_text.draw()
        for t in self._lines:
            t.draw()

    def on_key_press(self, symbol: int, modifiers: int):
        if symbol == arcade.key.ESCAPE:
            self.window.show_view(MenuView())


class ResultView(BaseView):
    def __init__(self, *, difficulty: str, won: bool, shots: int, hits: int, misses: int, duration: float):
        super().__init__()
        self.difficulty = difficulty
        self.won = won
        self.shots = shots
        self.hits = hits
        self.misses = misses
        self.duration = duration

        self._title_text: arcade.Text | None = None
        self._lines: list[arcade.Text] = []

        self.storage.add_game(
            difficulty=difficulty,
            result="win" if won else "lose",
            shots_total=shots,
            hits=hits,
            misses=misses,
            duration_sec=duration,
        )

    def on_show_view(self):
        arcade.set_background_color(C_BG)
        title = "ПОБЕДА" if self.won else "ПОРАЖЕНИЕ"
        self._title_text = arcade.Text(
            title,
            SCREEN_WIDTH / 2,
            SCREEN_HEIGHT - 70,
            C_TEXT,
            34,
            anchor_x="center",
            anchor_y="center",
            bold=True,
        )

        # Аналогично меню/статистике: строки отдельными объектами.
        self._lines = []
        y0 = SCREEN_HEIGHT / 2 + 70
        step = 28
        items = [
            f"Сложность: {('Лёгкий' if self.difficulty=='easy' else 'Средний')}",
            f"Выстрелов: {self.shots}",
            f"Попаданий: {self.hits}",
            f"Промахов: {self.misses}",
            f"Время: {self.duration:.1f} сек",
            "",
            "Enter — сыграть ещё",
            "Esc — меню",
        ]
        for i, line in enumerate(items):
            self._lines.append(
                arcade.Text(
                    line,
                    SCREEN_WIDTH / 2,
                    y0 - i * step,
                    C_TEXT,
                    18,
                    anchor_x="center",
                    anchor_y="center",
                    align="center",
                )
            )

    def on_draw(self):
        self.clear()
        if self._title_text:
            self._title_text.draw()
        for t in self._lines:
            t.draw()

    def on_key_press(self, symbol: int, modifiers: int):
        if symbol == arcade.key.ENTER:
            self.window.show_view(GameView(difficulty=self.difficulty))
        elif symbol == arcade.key.ESCAPE:
            self.window.show_view(MenuView())


class GameView(BaseView):
    """Основной игровой экран.

    Содержит:
    - два поля (игрок/противник)
    - обработку кликов и ходов
    - простейшую логику AI
    - подсчёт статистики партии (для ResultView)
    """

    def __init__(self, difficulty: str):
        super().__init__()
        self.difficulty = difficulty
        self.ai = ComputerAI(difficulty=difficulty)

        # Время старта партии, нужно для подсчёта длительности.
        self.start_time = time.time()
        # player_turn=True: ожидаем клик игрока по правому полю.
        self.player_turn = True
        # Текст сообщения выводится в HUD.
        self.message = "Ваш ход: кликните по правому полю (противник)."

        self.player_board = Board()
        self.enemy_board = Board()
        auto_place_fleet(self.player_board)
        auto_place_fleet(self.enemy_board)

        # Статистика текущей партии
        self.shots_total = 0
        self.hits = 0
        self.misses = 0


        # используем безопасный режим: если Camera нет, работаем без неё.
        self.camera = arcade.Camera(SCREEN_WIDTH, SCREEN_HEIGHT) if hasattr(arcade, "Camera") else None
        self.gui_camera = arcade.Camera(SCREEN_WIDTH, SCREEN_HEIGHT) if hasattr(arcade, "Camera") else None

        # Тексты заголовков полей (через Text, чтобы не использовать draw_text)
        self._title_player = arcade.Text(
            "Ваше поле",
            PLAYER_GRID_ORIGIN[0],
            SCREEN_HEIGHT - 40,
            C_TEXT,
            18,
            anchor_x="left",
            anchor_y="baseline",
        )
        self._title_enemy = arcade.Text(
            "Поле противника",
            ENEMY_GRID_ORIGIN[0],
            SCREEN_HEIGHT - 40,
            C_TEXT,
            18,
            anchor_x="left",
            anchor_y="baseline",
        )


    def on_show_view(self):
        arcade.set_background_color(C_BG)

    def _grid_from_mouse(self, x: float, y: float, origin: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Преобразует координаты мыши в координаты клетки (gx, gy).

        origin — нижний-левый угол конкретного поля.
        Возвращает None, если клик был вне поля.
        """
        ox, oy = origin
        gx = int((x - ox) // CELL_SIZE)
        gy = int((y - oy) // CELL_SIZE)
        if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
            return gx, gy
        return None

    def _apply_shot_visuals(self, target: str, gx: int, gy: int, hit: bool):
        # Визуальные эффекты убраны в упрощённой версии
        return

    def _enemy_click(self, x: float, y: float):
        """Обработка клика игрока по полю противника."""
        grid = self._grid_from_mouse(x, y, ENEMY_GRID_ORIGIN)
        if not grid:
            # Кликнули мимо сетки — игнорируем.
            return
        gx, gy = grid

        # Делаем выстрел по логической модели.
        res = self.enemy_board.shoot(gx, gy)
        if res.already_shot:
            # Не тратим ход и не меняем счётчики.
            self.message = "Нельзя стрелять повторно в эту клетку."
            return

        # Учитываем статистику только для уникальных выстрелов.
        self.shots_total += 1

        if res.hit:
            self.hits += 1
            # Правило: при попадании игрок ходит ещё раз.
            self.message = "Попадание! Ход продолжается."
            self._apply_shot_visuals("enemy", gx, gy, hit=True)
            if self.enemy_board.all_ships_destroyed():
                self._finish(won=True)
        else:
            self.misses += 1
            # При промахе передаём ход компьютеру.
            self.message = "Промах. Ход компьютера."
            self._apply_shot_visuals("enemy", gx, gy, hit=False)
            self.player_turn = False

    def _computer_turn_step(self):
        """Один шаг хода компьютера.

        AI выбирает координату, мы применяем выстрел к полю игрока.
        Если попадание — компьютер стреляет ещё (player_turn остаётся False).
        """
        x, y = self.ai.next_shot(self.player_board.hits, self.player_board.misses)
        res = self.player_board.shoot(x, y)

        self._apply_shot_visuals("player", x, y, hit=res.hit)

        if res.hit:
            self.message = "Компьютер попал и стреляет ещё."
            # Небольшая пауза, чтобы сообщение успевало читаться.
            self._ai_cooldown = 0.35
            if self.player_board.all_ships_destroyed():
                self._finish(won=False)
            # player_turn остаётся False -> AI продолжает
        else:
            self.message = "Компьютер промахнулся. Ваш ход."
            self.player_turn = True

    def _finish(self, won: bool):
        """Завершение партии: считаем длительность и переключаемся на экран результата."""
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

    def on_update(self, delta_time: float):
        # Игровой цикл Arcade: сюда попадаем примерно 60 раз в секунду.

        # Ход компьютера: делаем паузу между его выстрелами, чтобы UI не "мелькал".
        if getattr(self, "_ai_cooldown", 0) > 0:
            self._ai_cooldown = max(0.0, self._ai_cooldown - delta_time)
        elif not self.player_turn:
            self._computer_turn_step()

        # Камера статическая (заготовка под будущее движение/зум).
        if self.camera:
            self.camera.move_to((0, 0), speed=0.2)

    def on_draw(self):
        # Отрисовка кадра.
        self.clear()

        # Камера мира 
        if self.camera:
            self.camera.use()

        # Верхняя панель (HUD фон)
        top = SCREEN_HEIGHT
        bottom = SCREEN_HEIGHT - TOP_BAR
        if hasattr(arcade, "draw_lrbt_rectangle_filled"):
            arcade.draw_lrbt_rectangle_filled(0, SCREEN_WIDTH, bottom, top, C_PANEL)
        else:
            arcade.draw_lrtb_rectangle_filled(0, SCREEN_WIDTH, top, bottom, C_PANEL)

        # Заголовки полей
        self._title_player.draw()
        self._title_enemy.draw()

        # Рисуем поля примитивами (без спрайтов).
        # reveal_ships=True только для поля игрока.
        self._draw_board(PLAYER_GRID_ORIGIN, self.player_board, reveal_ships=True)
        self._draw_board(ENEMY_GRID_ORIGIN, self.enemy_board, reveal_ships=False)

        # Линии сетки поверх клеток
        self._draw_grid_lines(PLAYER_GRID_ORIGIN)
        self._draw_grid_lines(ENEMY_GRID_ORIGIN)

        # GUI-камера — отдельно от «мира», чтобы не зависеть от движения камеры.
        if self.gui_camera:
            self.gui_camera.use()

        # Строка статуса/подсказки
        self._draw_game_hud()

    def _draw_game_hud(self):
        """Текст в HUD"""
        turn_txt = "Ваш ход" if self.player_turn else "Ход компьютера"

        # Инициализация (или пересоздание, если текст поменялся)
        status_line = f"Сложность: {('Лёгкий' if self.difficulty=='easy' else 'Средний')}    {turn_txt}"
        if getattr(self, "_hud_status", None) is None or self._hud_status.text != status_line:
            self._hud_status = arcade.Text(
                status_line,
                SCREEN_WIDTH / 2,
                SCREEN_HEIGHT - 78,
                C_TEXT,
                16,
                anchor_x="center",
                anchor_y="center",
            )

        if getattr(self, "_hud_message", None) is None or self._hud_message.text != self.message:
            self._hud_message = arcade.Text(
                self.message,
                SCREEN_WIDTH / 2,
                SCREEN_HEIGHT - 100,
                C_TEXT,
                14,
                anchor_x="center",
                anchor_y="center",
            )

        if getattr(self, "_hud_hint", None) is None:
            self._hud_hint = arcade.Text(
                "Esc — меню",
                SCREEN_WIDTH - 20,
                SCREEN_HEIGHT - 24,
                C_TEXT,
                12,
                anchor_x="right",
                anchor_y="center",
            )

        self._hud_status.draw()
        self._hud_message.draw()
        self._hud_hint.draw()

    def _draw_board(self, origin: Tuple[int, int], board: Board, *, reveal_ships: bool):
        """Отрисовывает поле.

        reveal_ships управляет тем, показывать ли корабли (для поля противника — нет).
        """
        ox, oy = origin
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                # Центр клетки в координатах экрана.
                cx = ox + x * CELL_SIZE + CELL_SIZE / 2
                cy = oy + y * CELL_SIZE + CELL_SIZE / 2

                # Приоритет состояния: попадание > промах > корабль (если разрешено) > неизвестно
                if (x, y) in board.hits:
                    color = C_HIT
                elif (x, y) in board.misses:
                    color = C_MISS
                elif reveal_ships and (x, y) in board.ship_cells:
                    color = C_SHIP
                else:
                    color = C_UNKNOWN

                # Рисуем клетку как заполненный прямоугольник.
                # draw_polygon_filled используем как максимально совместимый примитив.
                half = (CELL_SIZE - 4) / 2
                arcade.draw_polygon_filled(
                    [
                        (cx - half, cy - half),
                        (cx + half, cy - half),
                        (cx + half, cy + half),
                        (cx - half, cy + half),
                    ],
                    color,
                )

    def _draw_grid_lines(self, origin: Tuple[int, int]):
        """Рисует линии сетки поверх поля."""
        ox, oy = origin
        w = CELL_SIZE * GRID_SIZE
        h = CELL_SIZE * GRID_SIZE
        for i in range(GRID_SIZE + 1):
            x = ox + i * CELL_SIZE
            arcade.draw_line(x, oy, x, oy + h, C_GRID, 1)
        for j in range(GRID_SIZE + 1):
            y = oy + j * CELL_SIZE
            arcade.draw_line(ox, y, ox + w, y, C_GRID, 1)

    def on_mouse_press(self, x: float, y: float, button: int, modifiers: int):
        """Клик мышью: игрок стреляет только ЛКМ и только в свой ход."""
        if button != arcade.MOUSE_BUTTON_LEFT:
            return
        if not self.player_turn:
            return
        self._enemy_click(x, y)

    def on_key_press(self, symbol: int, modifiers: int):
        """Горячие клавиши игрового экрана."""
        if symbol == arcade.key.ESCAPE:
            self.window.show_view(MenuView())


def main():
    # Точка входа.
    # update_rate=1/60 фиксирует желаемую частоту обновления (логики/анимаций).
    window = arcade.Window(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE, update_rate=1 / 60)

    # Запускаем с меню. Дальше экраны переключают друг друга через window.show_view
    window.show_view(MenuView())
    arcade.run()


if __name__ == "__main__":
    main()
