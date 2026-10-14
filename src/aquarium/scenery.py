"""Фон аквариума: вода, лучи света, блики, песок, камни, водоросли, пузырьки.

Статичное (вода, песок, камни) рисуется один раз в буфер. Движущееся
(лучи, блики, растения, пузырьки, струя фильтра, частицы) обновляется
каждый кадр дёшево: аддитивные текстуры и векторизованные частицы.
"""
import math

import numpy as np
import pygame
from pygame import gfxdraw

ADD = pygame.BLEND_RGB_ADD


def _smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def value_noise(w, h, cell, rng):
    """Гладкий шум (h, w) со значениями 0..1."""
    gw, gh = w // cell + 3, h // cell + 3
    g = rng.random((gh, gw)).astype(np.float32)
    xs = np.arange(w, dtype=np.float32) / cell
    ys = np.arange(h, dtype=np.float32) / cell
    x0, y0 = xs.astype(int), ys.astype(int)
    fx, fy = xs - x0, ys - y0
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    a, b = g[y0][:, x0], g[y0][:, x0 + 1]
    c, d = g[y0 + 1][:, x0], g[y0 + 1][:, x0 + 1]
    top = a * (1 - fx) + b * fx
    bot = c * (1 - fx) + d * fx
    return top * (1 - fy[:, None]) + bot * fy[:, None]


def fractal(w, h, rng, cells=(64, 24, 9, 3), weights=(0.45, 0.3, 0.17, 0.08)):
    out = np.zeros((h, w), np.float32)
    for c, k in zip(cells, weights):
        out += k * value_noise(w, h, c, rng)
    return out


def rgb_surface(arr):
    """arr (h, w, 3) uint8 -> Surface."""
    h, w = arr.shape[:2]
    return pygame.image.frombuffer(np.ascontiguousarray(arr).tobytes(),
                                   (w, h), "RGB").convert()


def rgba_surface(arr):
    h, w = arr.shape[:2]
    return pygame.image.frombuffer(np.ascontiguousarray(arr).tobytes(),
                                   (w, h), "RGBA").convert_alpha()


def make_rock(w, h, seed, tint=(1.0, 1.0, 1.0)):
    """Камень с объёмным освещением (сверху слева) и налётом водорослей."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xx - w / 2.0) / (w / 2.0)
    v = (yy - h * 0.60) / (h * 0.58)
    ang = np.arctan2(v, u)
    k = (1 + 0.10 * np.sin(3 * ang + rng.uniform(0, 6.3))
         + 0.07 * np.sin(5 * ang + rng.uniform(0, 6.3))
         + 0.04 * np.sin(9 * ang + rng.uniform(0, 6.3)))
    d = np.sqrt(u * u + v * v) / k
    mask = np.clip((1.0 - d) * 14.0, 0.0, 1.0)
    nz = fractal(w, h, rng, (40, 16, 7, 3), (0.5, 0.3, 0.14, 0.06))
    z = np.sqrt(np.clip(1.0 - d ** 2, 0, 1)) * (w * 0.20) + nz * (w * 0.05)
    gy, gx = np.gradient(z)
    inv = 1.0 / np.sqrt(gx * gx + gy * gy + 1.0)
    lx, ly, lz = -0.55, -0.70, 0.55
    ln = math.sqrt(lx * lx + ly * ly + lz * lz)
    shade = np.clip((-gx * lx - gy * ly + lz) * inv / ln, 0.0, 1.0)
    base = np.array([92, 108, 118], np.float32)
    col = base * (0.22 + 0.95 * shade[..., None])
    col *= (0.80 + 0.40 * nz)[..., None]
    col += np.array([6, 18, 30], np.float32) * (1.0 - shade[..., None])
    up = np.clip(inv * 1.4 - 0.45, 0, 1) * (v < 0.15)
    moss = (up * _smooth(0.45, 0.8, nz))[..., None]
    col = col * (1 - 0.55 * moss) + np.array([34, 96, 52], np.float32) \
        * (0.4 + 0.8 * shade[..., None]) * 0.55 * moss
    col *= np.array(tint, np.float32)
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = np.clip(col, 0, 255)
    out[..., 3] = (mask * 255).astype(np.uint8)
    return rgba_surface(out)


class Particles:
    """Пузырьки/частицы: x, y, vx, vy, rad, age, life, phase, amp."""

    def __init__(self, cap):
        self.cap = cap
        self.d = np.zeros((0, 9))
        self.acc = 0.0

    def emit(self, rows):
        if len(rows) == 0:
            return
        room = self.cap - len(self.d)
        if room <= 0:
            return
        self.d = np.vstack([self.d, np.asarray(rows)[:room]])

    def keep(self, mask):
        self.d = self.d[mask]


class Scenery:
    def __init__(self, w, h, seed=11):
        self.W, self.H = w, h
        self.rng = np.random.default_rng(seed)
        self.floor_y = int(h * 0.885)
        self.surface_y = 12
        self.t_prev = None
        self._build_static()
        self._build_light()
        self.plants = []
        self._build_flow()
        self._build_glass()

    # ------------------------------------------------------------------
    # статичная часть
    # ------------------------------------------------------------------
    def sand_top(self, x):
        x = np.asarray(x, np.float32)
        return (self.floor_y + 16 * np.sin(x / 210.0 + 0.8)
                + 9 * np.sin(x / 83.0 + 2.1) + 4 * np.sin(x / 31.0))

    def _build_static(self):
        W, H, rng = self.W, self.H, self.rng
        y = np.linspace(0, 1, H, dtype=np.float32)
        stops = [0.0, 0.16, 0.52, 1.0]
        cols = np.array([[62, 172, 190], [30, 132, 160], [11, 80, 112],
                         [4, 32, 54]], np.float32)
        base = np.stack([np.interp(y, stops, cols[:, k]) for k in range(3)],
                        axis=1)[:, None, :] * np.ones((1, W, 1), np.float32)
        x = np.linspace(0, 1, W, dtype=np.float32)[None, :]
        glow = np.exp(-((x - 0.40) / 0.36) ** 2) * (1 - y[:, None]) ** 2.2
        base += glow[..., None] * np.array([30, 44, 36], np.float32)
        edge = 1 - 0.30 * (np.abs(x - 0.5) * 2) ** 2.6
        base *= edge[..., None]

        # дальний план: размытые силуэты скал и растений
        sw, sh = W // 6, H // 6
        far = pygame.Surface((sw, sh), pygame.SRCALPHA)
        for cx, wd, ht in ((0.10, 0.22, 0.20), (0.34, 0.16, 0.12),
                           (0.58, 0.20, 0.16), (0.82, 0.24, 0.22)):
            pts = [(cx * sw - wd * sw / 2, sh)]
            for k in range(9):
                a = k / 8
                pts.append((cx * sw + (a - 0.5) * wd * sw,
                            sh - ht * sh * math.sin(a * math.pi) ** 0.7
                            * (0.8 + 0.4 * rng.random())))
            pts.append((cx * sw + wd * sw / 2, sh))
            pygame.draw.polygon(far, (6, 44, 62, 150), pts)
        for _ in range(18):
            fx = rng.uniform(0, sw)
            fh = rng.uniform(0.25, 0.55) * sh
            lean = rng.uniform(-5, 5)
            pygame.draw.polygon(far, (8, 62, 70, 90), [
                (fx - 2.2, sh), (fx + lean * 0.6, sh - fh * 0.55),
                (fx + lean, sh - fh), (fx + lean * 0.6 + 0.6, sh - fh * 0.55),
                (fx + 2.2, sh)])
        far = pygame.transform.smoothscale(far, (sw // 2, sh // 2))
        far = pygame.transform.smoothscale(far, (W, H))
        far_a = pygame.surfarray.array_alpha(far).T.astype(np.float32) / 255
        far_c = pygame.surfarray.array3d(far).transpose(1, 0, 2)
        base = base * (1 - far_a[..., None]) + far_c * far_a[..., None]

        # песок
        top = self.sand_top(np.arange(W))
        y0 = int(top.min()) - 8
        rows = H - y0
        yy = (np.arange(y0, H, dtype=np.float32)[:, None])
        depth = (yy - top[None, :])
        m = np.clip(depth * 0.7 + 0.5, 0, 1)
        k = np.clip(depth / (H - top.min()) * 1.25, 0, 1)[..., None]
        light = np.array([204, 182, 132], np.float32)
        deep = np.array([108, 96, 72], np.float32)
        sand = light * (1 - k) + deep * k
        grain = fractal(W, rows, rng, (6, 3, 2), (0.4, 0.35, 0.25))
        rip = np.sin(depth * 0.33 + np.arange(W)[None, :] * 0.012
                     + fractal(W, rows, rng, (70, 25), (0.6, 0.4)) * 8.0)
        sand = sand * (0.86 + 0.28 * grain[..., None]) + rip[..., None] * 5
        sand *= np.array([0.86, 0.94, 0.97], np.float32)
        band = base[y0:H]
        base[y0:H] = band * (1 - m[..., None]) + sand * m[..., None]

        self.bg = rgb_surface(np.clip(base, 0, 255).astype(np.uint8))

        # камни
        rocks = []
        for cxf, rw, rh, seed, tint in (
                (0.075, 380, 250, 3, (1.0, 1.0, 1.04)),
                (0.19, 190, 120, 8, (1.02, 1.0, 1.0)),
                (0.50, 150, 92, 14, (0.9, 1.0, 1.06)),
                (0.80, 210, 130, 21, (1.0, 1.0, 1.04)),
                (0.945, 400, 270, 5, (0.96, 1.0, 1.06))):
            rock = make_rock(rw, rh, seed, tint)
            cx = int(cxf * W)
            by = int(self.sand_top(cx)) + 34
            self.bg.blit(rock, (cx - rw // 2, by - rh))
            rocks.append((cx - rw // 2, by - rh, rock))
        self._build_sand_mask(top, y0, rocks)
        # галька
        for _ in range(110):
            px = int(rng.uniform(0, W))
            py = int(self.sand_top(px) + rng.uniform(8, H - self.floor_y - 6))
            rw = int(rng.uniform(4, 13))
            rh = max(3, int(rw * rng.uniform(0.55, 0.8)))
            c = rng.choice([(150, 140, 126), (120, 110, 100), (176, 166, 150),
                            (98, 94, 90), (160, 128, 98)])
            c = tuple(int(v * s) for v, s in zip(c, (0.8, 0.92, 0.98)))
            pebble = pygame.Surface((rw * 4, rh * 4), pygame.SRCALPHA)
            pygame.draw.ellipse(pebble, (10, 20, 24, 90),
                                (rw * 0.4, rh * 1.0, rw * 3.4, rh * 3.0))
            pygame.draw.ellipse(pebble, tuple(int(v * 0.6) for v in c)
                                + (255,), (0, 0, rw * 3.6, rh * 3.4))
            pygame.draw.ellipse(pebble, c + (255,),
                                (rw * 0.2, rh * 0.15, rw * 2.8, rh * 2.5))
            pygame.draw.ellipse(pebble, (235, 235, 225, 255),
                                (rw * 0.7, rh * 0.5, rw * 0.9, rh * 0.6))
            pebble = pygame.transform.smoothscale(pebble, (rw, rh))
            self.bg.blit(pebble, (px - rw // 2, py - rh // 2))
        # воздушный камень и труба фильтра
        self.stone_x = int(0.67 * W)
        sy = int(self.sand_top(self.stone_x)) + 20
        self.stone_y = sy
        pygame.draw.ellipse(self.bg, (84, 90, 94), (self.stone_x - 18,
                                                    sy - 6, 36, 16))
        pygame.draw.ellipse(self.bg, (132, 140, 142), (self.stone_x - 14,
                                                       sy - 6, 28, 9))
        self.pipe_x = int(0.065 * W)
        self.nozzle = (self.pipe_x + 44, 66)
        self._draw_pipe()

    def _build_sand_mask(self, top, y0, rocks):
        """Маска «только песок»: мягкий верхний край, камни вырезаны."""
        W, H = self.W, self.H
        yy = np.arange(y0, H, dtype=np.float32)[:, None]
        m = _smooth(-6.0, 22.0, yy - top[None, :]).astype(np.float32)
        for rx, ry, rock in rocks:
            al = pygame.surfarray.array_alpha(rock).T.astype(np.float32) / 255
            h_, w_ = al.shape
            x0, x1 = max(0, rx), min(W, rx + w_)
            y_0, y_1 = max(y0, ry), min(H, ry + h_)
            if x0 >= x1 or y_0 >= y_1:
                continue
            sub = al[y_0 - ry:y_1 - ry, x0 - rx:x1 - rx]
            m[y_0 - y0:y_1 - y0, x0:x1] *= 1.0 - sub
        g = (m * 255).astype(np.uint8)
        self.sand_y0 = y0
        self.sand_mask = rgb_surface(np.repeat(g[..., None], 3, axis=2))
        self.sand_tmp = pygame.Surface((W, H - y0)).convert()

    def _draw_pipe(self):
        s = self.bg
        x, ny = self.pipe_x, self.nozzle[1]
        for i, c in enumerate(((52, 62, 70), (84, 98, 108), (126, 142, 152),
                               (96, 110, 120), (60, 70, 80))):
            pygame.draw.line(s, c, (x + 4 * i, 0), (x + 4 * i, ny), 4)
        for i, c in enumerate(((52, 62, 70), (100, 114, 124), (130, 146, 156),
                               (70, 82, 92))):
            pygame.draw.line(s, c, (x, ny - 8 + 4 * i), (x + 46, ny - 8 + 4 * i),
                             4)
        pygame.draw.circle(s, (60, 70, 80), (x + 8, ny - 2), 10)
        pygame.draw.circle(s, (126, 142, 152), (x + 6, ny - 4), 4)
        pygame.draw.rect(s, (30, 38, 44), (x + 44, ny - 10, 6, 20),
                         border_radius=2)

    # ------------------------------------------------------------------
    # свет: лучи и блики-каустики
    # ------------------------------------------------------------------
    def _build_light(self):
        W, H, rng = self.W, self.H, self.rng
        rh = int(H * 0.82)
        yy = np.arange(rh, dtype=np.float32)[None, :]
        xx = np.arange(W, dtype=np.float32)[:, None]
        fall = (1.0 - yy / rh) ** 1.7 * _smooth(0, 40, yy)
        self.rays = []
        for ks, slope, gain in (((5, 9, 14), 0.30, 30.0),
                                ((7, 11, 17), 0.22, 22.0)):
            acc = np.zeros((W, rh), np.float32)
            for k in ks:
                ph = rng.uniform(0, 6.28)
                beam = np.maximum(0.0, np.cos(2 * np.pi * k
                                              * (xx - slope * yy) / W + ph))
                acc += beam ** 4 * rng.uniform(0.6, 1.0)
            acc = np.clip(acc * fall * gain / 1.6, 0, 255)
            arr = np.zeros((W, rh, 3), np.uint8)
            arr[..., 0] = acc * 0.62
            arr[..., 1] = acc * 0.92
            arr[..., 2] = acc
            self.rays.append(pygame.surfarray.make_surface(arr).convert())

        C = 512
        yy, xx = np.mgrid[0:C, 0:C].astype(np.float32)
        f = np.zeros((C, C), np.float32)
        for _ in range(6):
            while True:
                kx, ky = int(rng.integers(-6, 7)), int(rng.integers(-6, 7))
                if kx * kx + ky * ky >= 9:
                    break
            f += np.cos(2 * np.pi * (kx * xx + ky * yy) / C
                        + rng.uniform(0, 6.28))
        c = np.exp(-(f / 0.46) ** 2) ** 1.3
        self.caustics = []
        for tile, gain in ((c, 38.0), (np.rot90(c).copy(), 30.0)):
            for g in (gain, gain * 0.09):
                arr = np.zeros((C, C, 3), np.uint8)
                arr[..., 0] = tile * g * 0.78
                arr[..., 1] = tile * g * 0.98
                arr[..., 2] = tile * g
                self.caustics.append(pygame.surfarray.make_surface(arr)
                                     .convert())
        self.C = C
        glow = np.zeros((H // 7, W, 3), np.uint8)
        gy = np.linspace(0, 1, glow.shape[0], dtype=np.float32)[:, None]
        g = (1 - gy) ** 2.4 * 70
        glow[..., 0], glow[..., 1], glow[..., 2] = g * 0.7, g * 0.95, g
        self.glow = rgb_surface(glow)
        self.wave = pygame.Surface((W, 34), pygame.SRCALPHA)

    def _blit_tiled(self, scr, tile, ox, oy, area):
        scr.set_clip(area)
        C = self.C
        x = -(ox % C)
        while x < self.W:
            y = area.top - ((area.top + oy) % C)
            while y < area.bottom:
                scr.blit(tile, (x, y), special_flags=ADD)
                y += C
            x += C
        scr.set_clip(None)

    # ------------------------------------------------------------------
    # растения
    # ------------------------------------------------------------------
    def _make_blades(self, clusters, front):
        rng = self.rng
        rows = []
        greens = [((22, 92, 58), (70, 168, 96)), ((30, 112, 66), (96, 190, 100)),
                  ((16, 96, 88), (66, 172, 150)), ((70, 108, 40), (150, 190, 70)),
                  ((120, 44, 60), (206, 98, 96))]
        for cx, n, kind, hmin, hmax in clusters:
            for _ in range(n):
                x = cx + rng.normal(0, 28 if kind == 0 else 22)
                h = rng.uniform(hmin, hmax)
                gi = int(rng.integers(0, 4)) if rng.random() > 0.12 else 4
                dk, lt = greens[gi]
                if front:
                    dk = tuple(int(v * 0.55) for v in dk)
                    lt = tuple(int(v * 0.7) for v in lt)
                fog = 0.78 if not front else 1.0
                tint = (0.8, 0.96, 1.02)
                dk = tuple(int(v * fog * t_) for v, t_ in zip(dk, tint))
                lt = tuple(int(v * fog * t_) for v, t_ in zip(lt, tint))
                rows.append((x, self.sand_top(x) + rng.uniform(10, 40), h,
                             rng.uniform(11, 19) if kind == 0
                             else rng.uniform(22, 36), kind,
                             rng.uniform(0, 6.28), rng.uniform(0.7, 1.4),
                             rng.uniform(0.025, 0.06), rng.normal(0, 0.06),
                             dk, lt))
        return rows

    @staticmethod
    def _band_colors(dk, lt):
        """Основание темнее, кончик светлее и чуть прозрачнее в воде."""
        out = []
        for k in range(5):
            f = k / 4.0
            out.append(tuple(int(d * (0.76 + 0.24 * f) + (l - d) * 0.55 * f
                                 * f) for d, l in zip(dk, lt)))
        return out

    def _build_plants(self):
        W = self.W
        back = [(0.045 * W, 4, 0, 200, 400), (0.135 * W, 6, 0, 230, 450),
                (0.285 * W, 8, 0, 210, 430), (0.37 * W, 3, 1, 140, 190),
                (0.455 * W, 5, 0, 180, 370), (0.585 * W, 7, 0, 230, 460),
                (0.735 * W, 6, 0, 200, 420), (0.84 * W, 3, 1, 150, 200),
                (0.905 * W, 7, 0, 210, 440), (0.985 * W, 5, 0, 180, 390)]
        front = [(0.02 * W, 4, 0, 130, 230), (0.975 * W, 4, 0, 130, 240),
                 (0.50 * W, 2, 1, 80, 120)]
        self.plants = []
        for blades, is_front in ((self._make_blades(back, False), False),
                                 (self._make_blades(front, True), True)):
            a = {k: np.array([b[i] for b in blades], np.float64)
                 for i, k in enumerate(("x", "y", "h", "w", "kind", "ph",
                                        "sp", "amp", "lean"))}
            a["dk"] = [b[9] for b in blades]
            a["lt"] = [b[10] for b in blades]
            a["bands"] = [self._band_colors(b[9], b[10]) for b in blades]
            a["broad_i"] = np.flatnonzero(a["kind"] == 1)
            a["lt_broad"] = [a["lt"][i] for i in a["broad_i"]]
            self.plants.append(a)

    def _draw_plants(self, scr, a, t):
        n = 10
        s = np.linspace(0, 1, n + 1)[None, :]
        sway = (a["amp"][:, None] * np.sin(t * a["sp"][:, None]
                                           + a["ph"][:, None] + s * 2.4
                                           + a["x"][:, None] * 0.004)
                * s ** 1.5 + a["lean"][:, None] * s ** 1.7)
        X = a["x"][:, None] + sway * a["h"][:, None]
        Y = a["y"][:, None] - s * a["h"][:, None] * (1 - 0.12 * np.abs(sway))
        rib = 1.0 - 0.78 * s ** 1.3
        broad = np.sin(np.pi * np.clip(0.10 + 0.9 * s, 0, 1) ** 0.85) ** 0.7
        prof = np.where(a["kind"][:, None] == 0, rib, broad)
        wd = a["w"][:, None] * prof
        dX, dY = np.gradient(X, axis=1), np.gradient(Y, axis=1)
        nr = np.hypot(dX, dY) + 1e-6
        nx, ny = -dY / nr, dX / nr
        broad_i = a["broad_i"]
        a_dk, a_lt = a["dk"], a["lt"]
        for scale, shift, key in ((1.0, 0.0, "dk"), (0.42, 0.14, "lt")):
            sel = slice(None) if key == "dk" else broad_i
            hw = wd[sel] * 0.5 * scale
            off = wd[sel] * 0.5 * shift
            Xs, Ys, nxs, nys = X[sel], Y[sel], nx[sel], ny[sel]
            lx, ly = Xs + nxs * (hw + off), Ys + nys * (hw + off)
            rx, ry = Xs - nxs * (hw - off), Ys - nys * (hw - off)
            poly = np.concatenate([np.stack([lx, ly], -1),
                                   np.stack([rx, ry], -1)[:, ::-1]],
                                  axis=1).astype(np.int32).tolist()
            cols = a[key] if key == "dk" else a["lt_broad"]
            if key == "dk":
                m = len(poly[0]) // 2          # точек на сторону = n + 1
                for pts, (dk_, lt_) in zip(poly, zip(a_dk, a_lt)):
                    for bi in range(m - 1):
                        f = bi / (m - 2)
                        col = tuple(int(d * (0.72 + 0.28 * f)
                                        + (l - d) * 0.5 * f * f)
                                    for d, l in zip(dk_, lt_))
                        part = [pts[bi], pts[bi + 1],
                                pts[2 * m - 2 - bi], pts[2 * m - 1 - bi]]
                        pygame.draw.polygon(scr, col, part)
                    gfxdraw.aapolygon(scr, pts, tuple(
                        int(d * 0.9) for d in dk_))
            else:                      # блик только у широких листьев
                for pts, c in zip(poly, cols):
                    pygame.draw.polygon(scr, c, pts)

    # ------------------------------------------------------------------
    # течение: пузырьки, струя фильтра, частицы
    # ------------------------------------------------------------------
    def _build_flow(self):
        self.air = Particles(140)
        self.jet = Particles(360)
        self.puff = Particles(90)
        self.bub = {}
        for r in range(1, 10):
            ss = 4
            n = (2 * r + 4) * ss
            s = pygame.Surface((n, n), pygame.SRCALPHA)
            s.fill((220, 245, 255, 0))
            c = n / 2
            pygame.draw.circle(s, (200, 235, 255, 34), (c, c), r * ss)
            pygame.draw.circle(s, (232, 248, 255, 190), (c, c), r * ss,
                               max(ss // 2, 2))
            pygame.draw.circle(s, (255, 255, 255, 235),
                               (c - r * ss * 0.38, c - r * ss * 0.38),
                               max(2, int(r * ss * 0.22)))
            self.bub[r] = pygame.transform.smoothscale(s, (2 * r + 4,
                                                           2 * r + 4))
        self.puff_s = []
        for r in (16, 24, 34):
            yy, xx = np.mgrid[-r:r, -r:r].astype(np.float32)
            a = np.clip(1 - np.hypot(xx, yy) / r, 0, 1) ** 1.6
            arr = np.zeros((2 * r, 2 * r, 4), np.uint8)
            arr[..., :3] = (214, 240, 250)
            arr[..., 3] = a * 255
            self.puff_s.append(rgba_surface(arr))
        n = 130
        self.plk = np.stack([self.rng.uniform(0, self.W, n),
                             self.rng.uniform(40, self.H * 0.9, n),
                             self.rng.uniform(0.4, 1.0, n),
                             self.rng.uniform(0, 6.28, n)], axis=1)
        self.dot = []
        for r in (1, 2):
            s = pygame.Surface((2 * r + 2, 2 * r + 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (225, 245, 255, 150), (r + 1, r + 1), r)
            self.dot.append(s)

    def _update_flow(self, t, dt):
        rng = self.rng
        # воздушный камень
        a = self.air
        a.acc += dt * 9.0
        k = int(a.acc)
        a.acc -= k
        rows = []
        for _ in range(k):
            rad = float(rng.choice([1, 1, 2, 2, 3, 4, 5, 6]))
            rows.append((self.stone_x + rng.normal(0, 3), self.stone_y - 4,
                         rng.normal(0, 4), -(46 + 30 * rad), rad, 0.0, 99.0,
                         rng.uniform(0, 6.28), rng.uniform(1.0, 3.0)))
        a.emit(rows)
        d = a.d
        if len(d):
            d[:, 5] += dt
            d[:, 0] += d[:, 2] * dt + np.cos(d[:, 7] + d[:, 5] * 4.5) \
                * d[:, 8] * dt * 4.0
            d[:, 1] += d[:, 3] * dt
            a.keep(d[:, 1] > self.surface_y + 4)
        # струя фильтра
        j = self.jet
        j.acc += dt * 70.0
        k = int(j.acc)
        j.acc -= k
        nx, ny = self.nozzle
        rows = []
        for _ in range(k):
            sp = rng.uniform(150, 260)
            an = rng.normal(0.18, 0.16)
            rows.append((nx + rng.normal(0, 2), ny + rng.normal(0, 4),
                         sp * math.cos(an), sp * math.sin(an),
                         float(rng.choice([1, 1, 2, 2, 3])), 0.0,
                         rng.uniform(2.2, 4.2), rng.uniform(0, 6.28),
                         rng.uniform(0.5, 1.6)))
        j.emit(rows)
        d = j.d
        if len(d):
            d[:, 5] += dt
            damp = math.exp(-0.85 * dt)
            d[:, 2] *= damp
            d[:, 3] = d[:, 3] * damp - 34.0 * dt
            d[:, 0] += d[:, 2] * dt + np.cos(d[:, 7] + d[:, 5] * 3) \
                * d[:, 8] * dt * 6
            d[:, 1] += d[:, 3] * dt
            j.keep((d[:, 5] < d[:, 6]) & (d[:, 1] > self.surface_y + 4))
        p = self.puff
        p.acc += dt * 10.0
        k = int(p.acc)
        p.acc -= k
        rows = []
        for _ in range(k):
            sp = rng.uniform(110, 190)
            an = rng.normal(0.22, 0.12)
            rows.append((nx + 10, ny + rng.normal(0, 3), sp * math.cos(an),
                         sp * math.sin(an), float(rng.integers(0, 3)), 0.0,
                         rng.uniform(1.8, 3.0), 0, 0))
        p.emit(rows)
        d = p.d
        if len(d):
            d[:, 5] += dt
            damp = math.exp(-1.25 * dt)
            d[:, 2] *= damp
            d[:, 3] = d[:, 3] * damp - 18.0 * dt
            d[:, 0] += d[:, 2] * dt
            d[:, 1] += d[:, 3] * dt
            p.keep(d[:, 5] < d[:, 6])
        # частицы в воде: плавно плывут по течению
        pl = self.plk
        pl[:, 0] += (7.0 + 9.0 * np.sin(pl[:, 1] * 0.006 + t * 0.35)) \
            * pl[:, 2] * dt
        pl[:, 1] += (2.0 + 6.0 * np.sin(pl[:, 0] * 0.005 + t * 0.27 + pl[:, 3])) \
            * pl[:, 2] * dt
        pl[:, 0] %= self.W
        pl[:, 1] = np.where(pl[:, 1] > self.floor_y, 40, pl[:, 1])

    def _draw_bubbles(self, scr, ps, life_fade):
        d = ps.d
        if not len(d):
            return
        seq = []
        for x, y, rad, age, life in zip(d[:, 0].tolist(), d[:, 1].tolist(),
                                        d[:, 4].tolist(), d[:, 5].tolist(),
                                        d[:, 6].tolist()):
            s = self.bub[int(rad)]
            if life_fade:
                f = min(1.0, (life - age) / (0.35 * life), age * 5.0)
                if f < 1.0:
                    s = s.copy()
                    s.set_alpha(int(255 * max(0.0, f)))
            seq.append((s, (x - s.get_width() / 2, y - s.get_height() / 2)))
        scr.blits(seq, False)

    def _draw_puffs(self, scr):
        d = self.puff.d
        for x, y, i, age, life in zip(d[:, 0].tolist(), d[:, 1].tolist(),
                                      d[:, 4].tolist(), d[:, 5].tolist(),
                                      d[:, 6].tolist()):
            s = self.puff_s[int(i)]
            f = min(1.0, age / 0.4, (life - age) / (0.5 * life))
            s.set_alpha(int(52 * max(0.0, f)))
            scr.blit(s, (x - s.get_width() / 2, y - s.get_height() / 2))

    # ------------------------------------------------------------------
    # стекло и поверхность
    # ------------------------------------------------------------------
    def _build_glass(self):
        W, H = self.W, self.H
        self.edges = []
        e = 90
        g = (1 - np.linspace(0, 1, e, dtype=np.float32)) ** 2.2
        for side in ("l", "r", "b"):
            if side == "b":
                arr = np.zeros((e, W, 4), np.uint8)
                arr[..., 3] = (g[::-1, None] * 120)
                arr[..., 2] = 14
            else:
                arr = np.zeros((H, e, 4), np.uint8)
                a = g[None, :] if side == "l" else g[None, ::-1]
                arr[..., 3] = a * 96
                arr[..., 2] = 14
            self.edges.append((side, rgba_surface(arr)))
        self.streaks = []
        for x0, wd, al in ((0.30, 90, 9), (0.74, 40, 8)):
            sw = int(wd + 70)
            s = pygame.Surface((sw, H), pygame.SRCALPHA)
            pygame.draw.polygon(s, (255, 255, 255, al),
                                [(70, 0), (70 + wd, 0), (wd, H), (0, H)])
            self.streaks.append((s, int(x0 * W)))

    def _draw_surface(self, scr, t):
        W = self.W
        xs = np.arange(0, W + 24, 24)
        ys = (self.surface_y + 3.0 * np.sin(xs * 0.021 + t * 1.5)
              + 2.0 * np.sin(xs * 0.047 - t * 2.2)
              + 1.2 * np.sin(xs * 0.11 + t * 3.1))
        wv = self.wave
        wv.fill((0, 0, 0, 0))
        pts = [(0, 0)] + list(zip(xs.tolist(), ys.tolist())) + [(W, 0)]
        pygame.draw.polygon(wv, (212, 240, 248, 150), pts)
        line = list(zip(xs.tolist(), ys.tolist()))
        pygame.draw.lines(wv, (255, 255, 255, 235), False, line, 2)
        pygame.draw.lines(wv, (170, 226, 240, 110), False,
                          [(x, y + 3) for x, y in line], 3)
        scr.blit(wv, (0, 0))

    # ------------------------------------------------------------------
    # публичный интерфейс
    # ------------------------------------------------------------------
    def draw_back(self, scr, t, fx=True):
        dt = 0.016 if self.t_prev is None else min(0.05, max(0.0, t - self.t_prev))
        self.t_prev = t
        W, H = self.W, self.H
        scr.blit(self.bg, (0, 0))
        if not fx:
            return
        self._update_flow(t, dt)
        scr.blit(self.glow, (0, 0), special_flags=ADD)
        for i, ray in enumerate(self.rays):
            off = int((t * (14 if i == 0 else -9)) % W)
            scr.blit(ray, (-off, 0), special_flags=ADD)
            scr.blit(ray, (W - off, 0), special_flags=ADD)
        c = self.caustics
        water = pygame.Rect(0, self.surface_y, W, H - self.surface_y)
        self._blit_tiled(scr, c[1], int(t * 26), int(t * 14), water)
        self._blit_tiled(scr, c[3], -int(t * 20), int(t * 11), water)
        # усиленные блики только на песке (через маску, без шва на камнях)
        tmp, y0 = self.sand_tmp, self.sand_y0
        tmp.fill((0, 0, 0))
        area = pygame.Rect(0, 0, W, tmp.get_height())
        self._blit_tiled(tmp, c[0], int(t * 26), int(t * 14) + y0, area)
        self._blit_tiled(tmp, c[2], -int(t * 20), int(t * 11) + y0, area)
        tmp.blit(self.sand_mask, (0, 0), special_flags=pygame.BLEND_RGB_MULT)
        scr.blit(tmp, (0, y0), special_flags=ADD)
        seq = []
        for x, y, sz, ph in self.plk.tolist():
            tw = 0.6 + 0.4 * math.sin(t * 1.6 + ph)
            if tw > 0.5:
                seq.append((self.dot[1 if sz > 0.75 else 0], (x, y)))
        scr.blits(seq, False)

    def draw_front(self, scr, t, fx=True):
        if not fx:
            return
        self._draw_puffs(scr)
        self._draw_bubbles(scr, self.jet, True)
        self._draw_bubbles(scr, self.air, False)
        for s, x in self.streaks:
            scr.blit(s, (x, 0))
        for side, s in self.edges:
            if side == "l":
                scr.blit(s, (0, 0))
            elif side == "r":
                scr.blit(s, (self.W - s.get_width(), 0))
            else:
                scr.blit(s, (0, self.H - s.get_height()))
        self._draw_surface(scr, t)