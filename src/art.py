"""Процедурные спрайты: рыбки и кусочки хлеба.

Всё рисуется один раз с суперсэмплингом (в 4-6 раз крупнее), сглаживается
и кэшируется, поэтому в основном цикле остаётся только blit.
"""
import colorsys
import math
import random

import numpy as np
import pygame

FISH_SS = 3         # суперсэмплинг рыбки
N_FRAMES = 6        # кадров взмаха хвоста
N_HUE = 24          # корзин цвета
N_PAT = 1           # узор один (минимализм): кэш спрайтов в 4 раза меньше
FISH_SCALE = 1.15   # рыбки визуально крупнее физического размера
N_BREAD = 12        # вариантов формы кусочка хлеба
BREAD_PX = 13       # размер кусочка хлеба на экране


def hsv(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, min(1.0, max(0.0, s)),
                                  min(1.0, max(0.0, v)))
    return np.array([r * 255.0, g * 255.0, b * 255.0])


def _smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _rgb(c, k=1.0):
    return tuple(int(min(255, max(0, v * k))) for v in c)


def _bezier(p0, p1, p2, n=8):
    out = []
    for k in range(n + 1):
        u = k / n
        out.append(((1 - u) ** 2 * p0[0] + 2 * (1 - u) * u * p1[0]
                    + u * u * p2[0],
                    (1 - u) ** 2 * p0[1] + 2 * (1 - u) * u * p1[1]
                    + u * u * p2[1]))
    return out


def build_fish(hue_b, pat, r, frame):
    """Минималистичная плоская рыбка (смотрит вправо), без градиентов.

    Круглое тело, веерный хвост, спинной и нижний плавники, большой глаз,
    несколько тонких штрихов. Центр спрайта = центр существа в симуляции.
    Цвет берётся из гена оттенка; узор (pat) не рисуется.
    """
    ss = FISH_SS
    hue = (hue_b + 0.5) / N_HUE
    swing = math.sin(2.0 * math.pi * frame / N_FRAMES)
    rx, ry = 1.62 * r, 1.48 * r            # полуоси тела
    bc = -0.10 * r                         # центр тела по x
    tl = 1.7 * r                          # длина хвоста
    half_w = int(math.ceil(rx + tl * 1.05 + 3))
    half_h = int(math.ceil(ry + 0.95 * r + 3))
    w_px, h_px = 2 * half_w * ss, 2 * half_h * ss
    cx, cy = w_px / 2.0, h_px / 2.0

    def P(x, y):
        return (cx + x * ss, cy + y * ss)

    body_c = _rgb(hsv(hue, 0.60, 0.96)) + (255,)
    fin_c = _rgb(hsv(hue, 0.66, 0.84)) + (255,)
    line_c = _rgb(hsv(hue, 0.85, 0.60)) + (255,)
    lw = max(1, ss // 2)
    surf = pygame.Surface((w_px, h_px), pygame.SRCALPHA)

    # хвост: веер от задней точки тела, качается
    tx = bc - rx + 0.12 * r
    ty = 0.0
    sw = swing * 0.55 * r
    tail = [P(tx, ty)]
    for k in range(9):                    # округлый веер с небольшой выемкой
        u = k / 4.0 - 1.0                 # -1 .. 1
        reach = tl * (1.0 - 0.28 * abs(u) ** 2.2) * (0.86 if k == 4 else 1.0)
        tail.append(P(tx - reach * (0.62 + 0.38 * (1 - abs(u))),
                      ty + u * 1.3 * r + sw * (0.55 + 0.45 * (1 - abs(u)))))
    pygame.draw.polygon(surf, fin_c, tail)
    for k in (-1, 0, 1):
        pygame.draw.line(surf, line_c, P(tx - 0.2 * r, ty),
                         P(tx - tl * 0.85, ty + k * 0.85 * r + sw * 0.9), lw)

    # спинной плавник (волнистый «гребешок») и нижний плавник
    top_y = -ry * 0.92
    dors = [P(bc - 0.15 * r, top_y), P(bc - 0.35 * r, top_y - 0.55 * r),
            P(bc - 0.85 * r, top_y - 0.85 * r + 0.15 * swing * r),
            P(bc - 1.15 * r, top_y - 0.40 * r), P(bc - 1.25 * r, top_y + 0.2 * r)]
    pygame.draw.polygon(surf, fin_c, dors)
    low = [P(bc - 0.55 * r, ry * 0.85), P(bc - 0.95 * r, ry + 0.45 * r),
           P(bc - 1.25 * r, ry * 0.7)]
    pygame.draw.polygon(surf, fin_c, low)

    # тело
    body = pygame.Rect(0, 0, int(2 * rx * ss), int(2 * ry * ss))
    body.center = P(bc, 0)
    pygame.draw.ellipse(surf, body_c, body)

    # штрихи: жабры и спинной плавник
    for k in range(3):
        y = (-0.25 + 0.3 * k) * r
        pygame.draw.line(surf, line_c, P(bc + 0.15 * r, y + 0.55 * r),
                         P(bc + 0.55 * r - 0.1 * k * r, y + 0.62 * r), lw)
    pygame.draw.line(surf, line_c, P(bc - 0.5 * r, top_y - 0.1 * r),
                     P(bc - 0.8 * r, top_y - 0.6 * r), lw)

    # глаз
    ex, ey, re = bc + 0.85 * r, -0.12 * r, 0.50 * r
    pygame.draw.circle(surf, (246, 246, 248, 255), P(ex, ey), int(re * ss))
    pygame.draw.circle(surf, (14, 12, 16, 255), P(ex + 0.07 * r, ey),
                       int(re * 0.62 * ss))
    pygame.draw.circle(surf, (255, 255, 255, 255),
                       P(ex - 0.02 * r, ey - 0.16 * r),
                       max(1, int(re * 0.2 * ss)))

    return pygame.transform.smoothscale(surf, (w_px // ss, h_px // ss))


def build_bread(variant, size=BREAD_PX):
    """Кусочек хлеба: корочка, мякиш с порами, блик, тень."""
    ss = 6
    n = size * ss
    rnd = random.Random(variant * 7919 + 13)
    surf = pygame.Surface((n, n), pygame.SRCALPHA)
    surf.fill((205, 145, 72, 0))
    c = n / 2.0
    rad = n * 0.34
    expo = rnd.uniform(2.6, 3.6)
    rot = rnd.uniform(0.0, 2.0 * math.pi)
    sx, sy = rnd.uniform(0.92, 1.12), rnd.uniform(0.80, 1.0)
    jit = [rnd.uniform(-0.07, 0.07) for _ in range(20)]
    cr, sr = math.cos(rot), math.sin(rot)

    def outline(scale, dx=0.0, dy=0.0):
        pts = []
        for k in range(20):
            a = 2.0 * math.pi * k / 20
            ca, sa = math.cos(a), math.sin(a)
            rr = (abs(ca) ** expo + abs(sa) ** expo) ** (-1.0 / expo)
            rr *= 1.0 + jit[k]
            x, y = rr * ca * rad * scale * sx, rr * sa * rad * scale * sy
            pts.append((c + x * cr - y * sr + dx, c + x * sr + y * cr + dy))
        return pts

    pygame.draw.polygon(surf, (10, 20, 30, 80),
                        outline(1.0, 0.8 * ss, 1.0 * ss))
    pygame.draw.polygon(surf, (146, 84, 32, 255), outline(1.0))
    pygame.draw.polygon(surf, (204, 140, 62, 255),
                        outline(0.93, -0.18 * ss, -0.22 * ss))
    pygame.draw.polygon(surf, (247, 222, 164, 255),
                        outline(0.74, -0.25 * ss, -0.30 * ss))
    for _ in range(7):
        a = rnd.uniform(0, 2 * math.pi)
        d = rnd.uniform(0.0, 0.42) * rad
        rr = rnd.uniform(0.035, 0.075) * n
        pygame.draw.circle(surf, (226, 186, 116, 255),
                           (c + math.cos(a) * d - 0.25 * ss,
                            c + math.sin(a) * d - 0.3 * ss), max(1, int(rr)))
    pygame.draw.circle(surf, (255, 243, 205, 255),
                       (c - 0.32 * rad, c - 0.36 * rad), int(0.075 * n))
    return pygame.transform.smoothscale(surf, (size, size))


class SpriteBank:
    """Ленивый кэш спрайтов рыб и хлеба."""

    def __init__(self, builds_per_frame=8):
        self.fish = {}
        self.bread = [build_bread(i) for i in range(N_BREAD)]
        self.cap = builds_per_frame
        self.left = builds_per_frame

    def new_frame(self):
        self.left = self.cap

    def get_fish(self, hue_b, pat, rb, frame):
        """(спрайт, зеркальный спрайт) или None, если бюджет кадра исчерпан."""
        key = (hue_b, pat, rb, frame)
        hit = self.fish.get(key)
        if hit is not None:
            return hit
        if self.left <= 0:
            return None
        self.left -= 1
        spr = build_fish(hue_b, pat, (rb + 0.5) / 1.5 * FISH_SCALE, frame)
        self.fish[key] = (spr, pygame.transform.flip(spr, False, True))
        return self.fish[key]

    def get_any(self, hue_b, pat, rb, frame):
        """Любой уже готовый кадр этой рыбки (пока нужный не достроился)."""
        for k in range(N_FRAMES):
            hit = self.fish.get((hue_b, pat, rb, (frame + k) % N_FRAMES))
            if hit is not None:
                return hit
        return None