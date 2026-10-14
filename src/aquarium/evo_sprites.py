"""Спрайты стадий эволюции: клетка, рыба-ёж, рыба-меч, нарвал, акула, кракен.

Тот же стиль, что у art.build_fish: плоские цвета, суперсэмплинг, центр
спрайта = центр существа в симуляции, смотрит вправо. Рыбка (стадия 1)
строится в art.py.
"""
import colorsys
import math

import pygame

SS = 3              # суперсэмплинг
N_FRAMES = 6        # совпадает с art.N_FRAMES
HALF = 4.6          # полуразмер холста в единицах r


def _col(hue, s, v, a=255):
    r, g, b = colorsys.hsv_to_rgb(hue % 1.0, min(1.0, max(0.0, s)),
                                  min(1.0, max(0.0, v)))
    return (int(r * 255), int(g * 255), int(b * 255), a)


def _canvas(r, draw):
    half = int(math.ceil(HALF * r)) + 3
    w_px, h_px = 2 * half * SS, 2 * half * SS
    surf = pygame.Surface((w_px, h_px), pygame.SRCALPHA)
    cx, cy = w_px / 2.0, h_px / 2.0

    def P(x, y):
        return (cx + x * SS, cy + y * SS)

    draw(surf, P, r)
    return pygame.transform.smoothscale(surf, (w_px // SS, h_px // SS))


def _ellipse(surf, P, r, cx, cy, rx, ry, col):
    rect = pygame.Rect(0, 0, int(2 * rx * SS), int(2 * ry * SS))
    rect.center = P(cx, cy)
    pygame.draw.ellipse(surf, col, rect)


def _tail(surf, P, r, hue, bc, rx, swing, length=1.7):
    """Веерный хвост от задней части тела, качается."""
    fin = _col(hue, 0.66, 0.84)
    tx = bc - rx + 0.12 * r
    sw = swing * 0.55 * r
    pts = [P(tx, 0.0)]
    for k in range(9):
        u = k / 4.0 - 1.0
        reach = length * r * (1.0 - 0.28 * abs(u) ** 2.2)
        pts.append(P(tx - reach * (0.62 + 0.38 * (1 - abs(u))),
                     u * 1.3 * r + sw * (0.55 + 0.45 * (1 - abs(u)))))
    pygame.draw.polygon(surf, fin, pts)


def _eye(surf, P, r, x, y, size=0.5):
    pygame.draw.circle(surf, (246, 246, 248, 255), P(x, y),
                       int(size * r * SS))
    pygame.draw.circle(surf, (14, 12, 16, 255), P(x + 0.07 * r, y),
                       max(1, int(size * 0.62 * r * SS)))


# ---------------------------------------------------------------- клетка
def build_cell(hue, r, frame):
    """Милая клетка: закруглённый квадрат, чуть белый и полупрозрачный,
    с двумя глазками и лёгким «дыханием»."""
    phase = 2.0 * math.pi * frame / N_FRAMES
    pulse = 1.0 + 0.03 * math.sin(phase)
    bob = 0.05 * r * math.sin(phase)

    def draw(surf, P, r):
        half = 1.35 * r * pulse
        side = int(2 * half * SS)
        x0, y0 = P(-half, -half + bob)
        rect = pygame.Rect(int(x0), int(y0), side, side)
        radius = int(0.42 * side)
        # тело: почти белое, слегка подкрашено оттенком, полупрозрачное
        body = _col(hue, 0.18, 1.0, 165)
        pygame.draw.rect(surf, body, rect, border_radius=radius)
        # тонкая мембрана по краю
        pygame.draw.rect(surf, _col(hue, 0.45, 0.85, 235), rect,
                         max(1, int(0.1 * r * SS)), border_radius=radius)
        # блик сверху слева
        glow = pygame.Rect(0, 0, int(0.9 * r * SS), int(0.5 * r * SS))
        glow.center = P(-0.45 * r, -0.85 * r + bob)
        pygame.draw.ellipse(surf, (255, 255, 255, 170), glow)
        # глазки и румянец
        cy = -0.05 * r + bob
        for ex in (-0.35 * r, 0.35 * r):
            pygame.draw.circle(surf, (255, 255, 255, 255), P(ex, cy),
                               int(0.26 * r * SS))
            pygame.draw.circle(surf, (30, 26, 40, 255),
                               P(ex + 0.04 * r, cy + 0.03 * r),
                               int(0.15 * r * SS))
            pygame.draw.circle(surf, (255, 255, 255, 255),
                               P(ex - 0.02 * r, cy - 0.06 * r),
                               max(1, int(0.05 * r * SS)))
        for ex in (-0.62 * r, 0.62 * r):
            pygame.draw.circle(surf, (255, 150, 170, 150), P(ex, 0.25 * r + bob),
                               int(0.14 * r * SS))
        # улыбка
        pygame.draw.arc(surf, (60, 40, 60, 255),
                        pygame.Rect(*P(-0.2 * r, 0.12 * r + bob),
                                    int(0.4 * r * SS), int(0.3 * r * SS)),
                        math.pi * 1.15, math.pi * 1.85, max(1, SS // 2))

    return _canvas(r, draw)


# ----------------------------------------------------------- рыба-ёж
def build_hedgehog(hue, r, frame):
    swing = math.sin(2.0 * math.pi * frame / N_FRAMES)
    bc, rx, ry = -0.10 * r, 1.62 * r, 1.48 * r

    def draw(surf, P, r):
        _tail(surf, P, r, hue, bc, rx, swing)
        spine = _col(hue, 0.7, 0.55)
        # иглы по верхней дуге и бокам
        for k in range(11):
            th = math.pi * (0.08 + 0.84 * k / 10.0)
            bx, by = bc + rx * math.cos(th), -ry * math.sin(th)
            nx, ny = math.cos(th) / rx, -math.sin(th) / ry
            nl = math.hypot(nx, ny) or 1.0
            nx, ny = nx / nl, ny / nl
            tx_, ty_ = -ny, nx
            L = (0.55 + 0.1 * math.sin(k * 1.7)) * r
            half_b = 0.17 * r
            tip = (bx + nx * L, by + ny * L)
            pygame.draw.polygon(surf, spine, [
                P(bx + tx_ * half_b, by + ty_ * half_b), P(*tip),
                P(bx - tx_ * half_b, by - ty_ * half_b)])
        _ellipse(surf, P, r, bc, 0.0, rx, ry, _col(hue, 0.6, 0.96))
        for k in range(3):
            y = (-0.25 + 0.3 * k) * r
            pygame.draw.line(surf, _col(hue, 0.85, 0.6), P(bc + 0.15 * r, y),
                             P(bc + 0.5 * r, y + 0.2 * r), max(1, SS // 2))
        _eye(surf, P, r, bc + 0.85 * r, -0.12 * r)

    return _canvas(r, draw)


# ---------------------------------------------------------- рыба-меч
def build_swordfish(hue, r, frame):
    swing = math.sin(2.0 * math.pi * frame / N_FRAMES)
    bc, rx, ry = -0.2 * r, 1.7 * r, 1.2 * r

    def draw(surf, P, r):
        _tail(surf, P, r, hue, bc, rx, swing)
        fin = _col(hue, 0.66, 0.84)
        # высокий спинной плавник
        pygame.draw.polygon(surf, fin, [
            P(bc - 0.1 * r, -ry * 0.8), P(bc - 1.25 * r, -ry * 0.8),
            P(bc - 0.9 * r, -ry - 0.95 * r)])
        # меч: длинный тонкий клюв
        x0 = bc + rx - 0.1 * r
        sword = _col(hue, 0.35, 0.9)
        pygame.draw.polygon(surf, sword, [
            P(x0, -0.24 * r), P(bc + rx + 2.2 * r, 0.0), P(x0, 0.24 * r)])
        _ellipse(surf, P, r, bc, 0.0, rx, ry, _col(hue, 0.4, 0.82))
        pygame.draw.line(surf, _col(hue, 0.6, 0.5), P(bc - 0.8 * r, -0.1 * r),
                         P(bc + 1.1 * r, -0.05 * r), max(1, SS // 2))
        _eye(surf, P, r, bc + 0.9 * r, -0.15 * r, 0.45)

    return _canvas(r, draw)


# ------------------------------------------------------------ нарвал
def build_narwhal(hue, r, frame):
    swing = math.sin(2.0 * math.pi * frame / N_FRAMES)
    bc, rx, ry = -0.1 * r, 1.8 * r, 1.0 * r

    def draw(surf, P, r):
        _tail(surf, P, r, hue, bc, rx, swing)
        # маленький спинной горбик
        _ellipse(surf, P, r, bc - 0.5 * r, -ry * 0.85, 0.35 * r, 0.25 * r,
                 _col(hue, 0.45, 0.45))
        _ellipse(surf, P, r, bc, 0.0, rx, ry, _col(hue, 0.3, 0.5))
        # светлое брюхо
        _ellipse(surf, P, r, bc + 0.1 * r, 0.42 * r, rx * 0.85, 0.45 * r,
                 _col(hue, 0.1, 0.95))
        for sx, sy, sr in ((-0.6, -0.3, 0.18), (-1.0, 0.1, 0.12),
                           (0.2, -0.45, 0.1)):
            _ellipse(surf, P, r, bc + sx * r, sy * r, sr * r * 1.5, sr * r,
                     _col(hue, 0.1, 0.97))
        # бивень: длинный, белый, со спиральными штрихами
        x0, y0 = bc + rx - 0.1 * r, -0.1 * r
        x1 = bc + rx + 2.7 * r
        ivory = _col(0.12, 0.18, 0.98)
        pygame.draw.polygon(surf, ivory, [
            P(x0, y0 - 0.11 * r), P(x1, y0 - 0.02 * r), P(x0, y0 + 0.11 * r)])
        for k in range(4):
            xs = x0 + (0.45 + 0.55 * k) * r
            pygame.draw.line(surf, _col(0.12, 0.5, 0.75),
                             P(xs, y0 + 0.1 * r), P(xs + 0.18 * r, y0 - 0.1 * r),
                             max(1, SS // 2))
        _eye(surf, P, r, bc + 0.9 * r, -0.12 * r, 0.42)

    return _canvas(r, draw)


# ------------------------------------------------------------- акула
def build_shark(hue, r, frame):
    swing = math.sin(2.0 * math.pi * frame / N_FRAMES)
    bc = 0.2 * r

    def draw(surf, P, r):
        body = _col(hue, 0.22, 0.72)
        dark = _col(hue, 0.3, 0.45)
        sw = swing * 0.35 * r
        # хвост: две лопасти
        pygame.draw.polygon(surf, body, [
            P(bc - 1.4 * r, -0.2 * r), P(bc - 2.4 * r, -1.2 * r + sw),
            P(bc - 2.0 * r, 0.0)])
        pygame.draw.polygon(surf, body, [
            P(bc - 1.4 * r, 0.2 * r), P(bc - 2.3 * r, 0.9 * r + sw),
            P(bc - 2.0 * r, 0.0)])
        # спинной плавник (высокий, назад) и брюшной
        pygame.draw.polygon(surf, dark, [
            P(bc + 0.2 * r, -0.7 * r), P(bc - 0.9 * r, -0.7 * r),
            P(bc - 0.5 * r, -1.95 * r)])
        pygame.draw.polygon(surf, dark, [
            P(bc - 0.3 * r, 0.6 * r), P(bc - 0.8 * r, 0.6 * r),
            P(bc - 0.9 * r, 1.05 * r)])
        # тело и заострённая морда
        _ellipse(surf, P, r, bc, 0.0, 1.9 * r, 0.9 * r, body)
        pygame.draw.polygon(surf, body, [
            P(bc + 1.4 * r, -0.6 * r), P(bc + 2.4 * r, 0.0),
            P(bc + 1.4 * r, 0.6 * r)])
        # жаберные щели
        for k in range(3):
            gx = bc + 0.25 * r + k * 0.22 * r
            pygame.draw.line(surf, dark, P(gx, -0.35 * r), P(gx - 0.08 * r,
                             0.15 * r), max(1, SS // 2))
        _eye(surf, P, r, bc + 1.05 * r, -0.22 * r, 0.32)

    return _canvas(r, draw)


# ------------------------------------------------------------ кракен
def build_kraken(hue, r, frame):
    phase = 2.0 * math.pi * frame / N_FRAMES
    bc = 0.35 * r

    def draw(surf, P, r):
        flesh = _col(hue, 0.62, 0.52)
        suck = _col(hue, 0.3, 0.92)
        # щупальца: от задней половины мантии назад, с волной
        for k in range(8):
            a = math.pi * (0.55 + 0.9 * k / 7.0)     # дуга сзади
            bx = bc + 1.0 * r * math.cos(a)
            by = 0.95 * r * math.sin(a)
            path = []
            L = (2.9 + 0.25 * (k % 3)) * r
            for s in range(12):
                u = s / 11.0
                x = bx - L * u
                y = by + math.sin(phase + k * 0.9 + u * 3.0) * 0.6 * r * u
                path.append((x, y, (0.42 - 0.3 * u) * r))
            for x, y, rad in path:
                pygame.draw.circle(surf, flesh, P(x, y),
                                   max(1, int(rad * SS)))
            for x, y, rad in path[3::3]:
                pygame.draw.circle(surf, suck, P(x, y + 0.04 * r),
                                   max(1, int(0.07 * r * SS)))
        # мантия (голова)
        _ellipse(surf, P, r, bc, 0.0, 1.55 * r, 1.35 * r, flesh)
        _ellipse(surf, P, r, bc - 0.25 * r, -0.5 * r, 0.9 * r, 0.35 * r,
                 _col(hue, 0.5, 0.7, 120))
        # глаза
        for ey in (-0.55 * r, 0.55 * r):
            pygame.draw.circle(surf, (250, 246, 230, 255), P(bc + 0.75 * r, ey),
                               int(0.36 * r * SS))
            pygame.draw.circle(surf, (20, 10, 14, 255),
                               P(bc + 0.86 * r, ey), int(0.18 * r * SS))

    return _canvas(r, draw)


# ------------------------------------------------------------ дельфин
def build_dolphin(hue, r, frame):
    swing = math.sin(2.0 * math.pi * frame / N_FRAMES)
    bc, rx, ry = -0.1 * r, 1.9 * r, 0.85 * r

    def draw(surf, P, r):
        _tail(surf, P, r, hue, bc, rx, swing, length=1.5)
        pygame.draw.polygon(surf, _col(hue, 0.4, 0.6), [
            P(bc - 0.1 * r, -ry * 0.9), P(bc - 0.9 * r, -ry * 0.9),
            P(bc - 0.6 * r, -ry - 0.9 * r)])
        _ellipse(surf, P, r, bc, 0.0, rx, ry, _col(hue, 0.35, 0.75))
        _ellipse(surf, P, r, bc + 0.2 * r, 0.4 * r, rx * 0.85, 0.4 * r,
                 _col(hue, 0.1, 0.97))
        pygame.draw.polygon(surf, _col(hue, 0.35, 0.7), [
            P(bc + rx - 0.2 * r, -0.2 * r), P(bc + rx + 0.8 * r, 0.0),
            P(bc + rx - 0.2 * r, 0.2 * r)])
        _eye(surf, P, r, bc + 0.9 * r, -0.2 * r, 0.42)

    return _canvas(r, draw)


# -------------------------------------------------------------- косатка
def build_orca(hue, r, frame):
    swing = math.sin(2.0 * math.pi * frame / N_FRAMES)
    bc, rx, ry = -0.1 * r, 1.8 * r, 1.05 * r

    def draw(surf, P, r):
        _tail(surf, P, r, hue, bc, rx, swing, length=1.4)
        pygame.draw.polygon(surf, _col(0.0, 0.05, 0.1), [
            P(bc - 0.2 * r, -ry * 0.9), P(bc - 0.5 * r, -ry * 0.9),
            P(bc - 0.1 * r, -ry - 1.6 * r)])
        _ellipse(surf, P, r, bc, 0.0, rx, ry, _col(0.0, 0.05, 0.12))
        _ellipse(surf, P, r, bc + 0.7 * r, 0.1 * r, 0.7 * r, 0.3 * r,
                 _col(0.0, 0.0, 0.97))
        _ellipse(surf, P, r, bc - 0.7 * r, 0.1 * r, 0.3 * r, 0.25 * r,
                 _col(0.0, 0.0, 0.97))
        _eye(surf, P, r, bc + 1.0 * r, -0.2 * r, 0.42)

    return _canvas(r, draw)


# ------------------------------------------------------ электрический угорь
def build_eel(hue, r, frame):
    phase = 2.0 * math.pi * frame / N_FRAMES

    def draw(surf, P, r):
        col = _col(hue, 0.5, 0.6)
        for k in range(14):
            u = k / 13.0
            x = 1.9 * r - 4.2 * r * u
            y = 0.6 * r * math.sin(3.0 * u * math.pi + phase)
            rad = (0.55 - 0.3 * u) * r
            pygame.draw.circle(surf, col, P(x, y), max(1, int(rad * SS)))
        pygame.draw.circle(surf, _col(hue, 0.5, 0.75), P(1.9 * r, 0.0),
                           int(0.7 * r * SS))
        _eye(surf, P, r, 1.9 * r + 0.1 * r, -0.2 * r, 0.3)

    return _canvas(r, draw)


# ------------------------------------------------------------------ тунец
def build_tuna(hue, r, frame):
    swing = math.sin(2.0 * math.pi * frame / N_FRAMES)
    bc, rx, ry = 0.0, 1.9 * r, 0.75 * r

    def draw(surf, P, r):
        _tail(surf, P, r, hue, bc, rx, swing, length=1.2)
        for k in range(3):
            pygame.draw.polygon(surf, _col(hue, 0.6, 0.5), [
                P(bc - (0.6 + 0.5 * k) * r, ry * 0.7),
                P(bc - (0.9 + 0.5 * k) * r, ry + 0.3 * r),
                P(bc - (0.3 + 0.5 * k) * r, ry * 0.7)])
        pygame.draw.polygon(surf, _col(hue, 0.6, 0.55), [
            P(bc - 0.2 * r, -ry * 0.9), P(bc - 0.8 * r, -ry * 0.9),
            P(bc - 0.5 * r, -ry - 0.5 * r)])
        _ellipse(surf, P, r, bc, 0.0, rx, ry, _col(hue, 0.6, 0.7))
        _ellipse(surf, P, r, bc + 0.1 * r, 0.35 * r, rx * 0.9, 0.35 * r,
                 _col(hue, 0.05, 0.97))
        _eye(surf, P, r, bc + 1.0 * r, -0.12 * r, 0.38)

    return _canvas(r, draw)


# ------------------------------------------------------------ рыба-фонарь
def build_anglerfish(hue, r, frame):
    bc = -0.2 * r

    def draw(surf, P, r):
        body = _col(hue, 0.5, 0.3)
        _ellipse(surf, P, r, bc, 0.0, 1.4 * r, 1.3 * r, body)
        # приманка на «удочке» перед пастью
        pygame.draw.line(surf, _col(hue, 0.5, 0.4), P(bc + 1.0 * r, -1.0 * r),
                         P(bc + 1.8 * r, -2.2 * r), max(1, SS))
        pygame.draw.circle(surf, (255, 240, 120, 255), P(bc + 1.8 * r, -2.3 * r),
                           int(0.35 * r * SS))
        pygame.draw.circle(surf, (255, 240, 120, 90), P(bc + 1.8 * r, -2.3 * r),
                           int(0.8 * r * SS))
        # пасть и зубы
        pygame.draw.polygon(surf, _col(hue, 0.5, 0.15), [
            P(bc + 0.4 * r, -0.2 * r), P(bc + 1.5 * r, -0.4 * r),
            P(bc + 1.5 * r, 0.6 * r), P(bc + 0.4 * r, 0.5 * r)])
        for k in range(4):
            xt = bc + (0.6 + 0.25 * k) * r
            pygame.draw.polygon(surf, (250, 248, 240, 255), [
                P(xt, -0.2 * r), P(xt + 0.12 * r, -0.2 * r),
                P(xt + 0.06 * r, 0.05 * r)])
        _eye(surf, P, r, bc - 0.4 * r, -0.5 * r, 0.35)

    return _canvas(r, draw)


# ------------------------------------------------------------ морской змей
def build_serpent(hue, r, frame):
    phase = 2.0 * math.pi * frame / N_FRAMES

    def draw(surf, P, r):
        col = _col(hue, 0.55, 0.55)
        ridge = _col(hue, 0.7, 0.4)
        for k in range(12):
            u = k / 11.0
            x = 1.8 * r - 4.0 * r * u
            y = 0.7 * r * math.sin(2.5 * u * math.pi + phase)
            pygame.draw.circle(surf, col, P(x, y), int(0.5 * r * SS))
            if k % 2 == 0:
                pygame.draw.polygon(surf, ridge, [
                    P(x - 0.2 * r, y - 0.4 * r), P(x + 0.2 * r, y - 0.4 * r),
                    P(x, y - 0.95 * r)])
        pygame.draw.circle(surf, _col(hue, 0.55, 0.7), P(1.8 * r, 0.0),
                           int(0.75 * r * SS))
        _eye(surf, P, r, 1.8 * r + 0.15 * r, -0.2 * r, 0.32)

    return _canvas(r, draw)


# ------------------------------------------------------ гигантский кальмар
def build_giant_squid(hue, r, frame):
    phase = 2.0 * math.pi * frame / N_FRAMES
    bc = 0.4 * r

    def draw(surf, P, r):
        flesh = _col(hue, 0.5, 0.75)
        for k in range(10):
            a = math.pi * (0.7 + 0.6 * k / 9.0)
            bx = bc + 0.9 * r * math.cos(a)
            by = 0.6 * r * math.sin(a)
            for s in range(12):
                u = s / 11.0
                x = bx - (3.2 - 0.3 * (k % 2)) * r * u
                y = by + math.sin(phase + k * 0.7 + u * 3.0) * 0.5 * r * u
                pygame.draw.circle(surf, flesh, P(x, y),
                                   max(1, int((0.3 - 0.22 * u) * r * SS)))
        pygame.draw.polygon(surf, _col(hue, 0.5, 0.65), [
            P(bc - 0.8 * r, -0.9 * r), P(bc - 2.0 * r, -0.2 * r),
            P(bc - 0.8 * r, 0.9 * r)])
        _ellipse(surf, P, r, bc, 0.0, 1.25 * r, 0.95 * r, flesh)
        for ey in (-0.45 * r, 0.45 * r):
            pygame.draw.circle(surf, (250, 240, 230, 255),
                               P(bc + 0.6 * r, ey), int(0.4 * r * SS))
            pygame.draw.circle(surf, (20, 10, 14, 255),
                               P(bc + 0.7 * r, ey), int(0.2 * r * SS))

    return _canvas(r, draw)


# -------------------------------------------------------------- левиафан
def build_leviathan(hue, r, frame):
    swing = math.sin(2.0 * math.pi * frame / N_FRAMES)
    bc, rx, ry = 0.0, 2.1 * r, 1.2 * r

    def draw(surf, P, r):
        _tail(surf, P, r, hue, bc, rx, swing, length=1.3)
        _ellipse(surf, P, r, bc, 0.0, rx, ry, _col(hue, 0.3, 0.6))
        _ellipse(surf, P, r, bc + 0.2 * r, 0.55 * r, rx * 0.8, 0.5 * r,
                 _col(hue, 0.1, 0.95))
        # фонтан из дыхала
        for k, rad in enumerate((0.22, 0.28, 0.2)):
            pygame.draw.circle(surf, (235, 245, 255, 230),
                               P(bc + 0.9 * r + k * 0.1 * r,
                                 -ry - (0.4 + 0.5 * k) * r), int(rad * r * SS))
        _eye(surf, P, r, bc + 1.3 * r, -0.25 * r, 0.3)

    return _canvas(r, draw)


# ------------------------------------------------------------- мегалодон
def build_megalodon(hue, r, frame):
    # та же акула, но заметно крупнее и тёмнее
    return build_shark(hue, r * 1.3, frame)


# ----------------------------------------------------- дракон бездны
def build_abyss_dragon(hue, r, frame):
    swing = math.sin(2.0 * math.pi * frame / N_FRAMES)
    bc, rx, ry = -0.1 * r, 1.5 * r, 1.2 * r

    def draw(surf, P, r):
        wing = _col(hue, 0.8, 0.45, 230)
        pygame.draw.polygon(surf, wing, [
            P(bc + 0.2 * r, -ry * 0.6), P(bc - 1.4 * r, -ry - 1.5 * r + swing * 0.3 * r),
            P(bc - 1.2 * r, -ry * 0.2)])
        pygame.draw.polygon(surf, wing, [
            P(bc + 0.2 * r, ry * 0.6), P(bc - 1.4 * r, ry + 1.5 * r - swing * 0.3 * r),
            P(bc - 1.2 * r, ry * 0.2)])
        for k in range(8):
            th = math.pi * (0.1 + 0.8 * k / 7.0)
            bx, by = bc + rx * math.cos(th), -ry * math.sin(th)
            L = 0.9 * r
            pygame.draw.polygon(surf, _col(hue, 0.9, 0.35), [
                P(bx - 0.2 * r, by), P(bx + 0.2 * r, by),
                P(bx + math.cos(th) * L * 0.5, by - math.sin(th) * L - 0.6 * r)])
        _ellipse(surf, P, r, bc, 0.0, rx, ry, _col(hue, 0.8, 0.35))
        for ey in (-0.35 * r, 0.35 * r):
            pygame.draw.circle(surf, (255, 60, 60, 255),
                               P(bc + 0.9 * r, ey), int(0.2 * r * SS))

    return _canvas(r, draw)


BUILDERS = {
    0: build_cell,
    2: build_hedgehog,
    3: build_swordfish,
    4: build_narwhal,
    5: build_shark,
    6: build_kraken,
    7: build_dolphin,
    8: build_orca,
    9: build_eel,
    10: build_tuna,
    11: build_anglerfish,
    12: build_serpent,
    13: build_giant_squid,
    14: build_leviathan,
    15: build_megalodon,
    16: build_abyss_dragon,
}
