"""Отрисовка pygame: красивый аквариум, графики, панель управления.

Мир рисуется из scenery.py (вода, свет, песок, растения, пузырьки) и art.py
(рыбки по генам, хлеб вместо еды). Правая панель оформлена по системе из
landing-page-design: шрифт Manrope, размеры 12/14/16/18/20, отступы 2/4/8/12/
16/24/32/40/48, вложенные радиусы, плоские фоны из палитры, hover/active/focus.
"""
import math
import os
import time

import numpy as np
import pygame
from pygame import gfxdraw

from . import art
from .scenery import Scenery
from .stats import GRAZER, PARASITE, PREDATOR, classify

HERE = os.path.dirname(os.path.abspath(__file__))

# палитра (только разрешённые фоны) и цвета данных
BG0, BG1, BG2, BG3, BG4 = ((0, 0, 0), (24, 24, 24), (31, 31, 31),
                           (39, 39, 39), (49, 49, 49))
WHITE, MUTED, INK = (255, 255, 255), (155, 155, 155), (0, 0, 0)
C_ALL, C_GRAZE = (240, 240, 240), (90, 230, 120)
C_PRED, C_PARA = (255, 70, 70), (230, 70, 255)
C_GOLD, C_ORANGE, C_BLUE = (255, 220, 80), (255, 160, 80), (120, 220, 255)
ROLE_RING = {}
ROLE_RU = ["травоядная", "хищник", "паразит", "молодая"]


def mix(a, b, k):
    return tuple(int(a[i] + (b[i] - a[i]) * k) for i in range(len(a)))


def fnum(v, nd=1):
    return ("%.*f" % (nd, v)).replace(".", ",")


_RR = {}


def rr(w, h, r, color, border=None):
    """Скруглённый прямоугольник со сглаживанием (суперсэмплинг + кэш)."""
    w, h = int(w), int(h)
    if w < 1 or h < 1:
        return pygame.Surface((1, 1), pygame.SRCALPHA)
    key = (w, h, r, color, border)
    s = _RR.get(key)
    if s is not None:
        return s
    if len(_RR) > 500:
        _RR.clear()
    k = 3
    big = pygame.Surface((w * k, h * k), pygame.SRCALPHA)
    big.fill(((border or color)[:3]) + (0,))
    full = big.get_rect()
    rad = min(r, w // 2, h // 2) * k
    if border:
        pygame.draw.rect(big, border, full, border_radius=rad)
        pygame.draw.rect(big, color, full.inflate(-2 * k, -2 * k),
                         border_radius=max(0, rad - k))
    else:
        pygame.draw.rect(big, color, full, border_radius=rad)
    s = pygame.transform.smoothscale(big, (w, h))
    _RR[key] = s
    return s


class Fonts:
    SIZES = {"xs": 12, "sm": 14, "base": 16, "lg": 18, "xl": 20}

    def __init__(self):
        pygame.font.init()
        self.f, self.cache = {}, {}
        for weight in ("Medium", "SemiBold"):
            path = os.path.join(HERE, "fonts", "Manrope-%s.ttf" % weight)
            if not os.path.exists(path):
                path = pygame.font.match_font("manrope,geist,poppins")
            for k, s in self.SIZES.items():
                f = pygame.font.Font(path, s)
                if path is None and weight == "SemiBold":
                    f.set_bold(True)
                self.f[(weight, k)] = f

    def text(self, size, s, color, weight="Medium"):
        key = (weight, size, s, color)
        surf = self.cache.get(key)
        if surf is None:
            if len(self.cache) > 600:
                self.cache.clear()
            surf = self.f[(weight, size)].render(s, True, color)
            self.cache[key] = surf
        return surf


class Renderer:
    FOCUS_ORDER = ("pause", "slower", "faster", "fx", "rays")

    def __init__(self, cfg, screen):
        self.cfg = cfg
        self.screen = screen
        self.F = Fonts()
        self.show_rays = True
        self.fx = True
        self.seen_rounds = 0
        self.banner_until = 0.0
        self.banner_text = ""
        self.storm = pygame.Surface((cfg.world_w, cfg.world_h),
                                    pygame.SRCALPHA)
        self._zone_key = None
        self.scen = Scenery(cfg.world_w, cfg.world_h)
        self.bank = art.SpriteBank(builds_per_frame=16)
        self.bread = list(self.bank.bread)
        self.ph = np.zeros(cfg.max_creatures)
        self.rot_cache = {}
        self.t = 0.0
        self.t_prev = time.time()
        self.hits = {}
        self.focus = None
        self.focus_visible = False

    # ------------------------------------------------------------ ввод
    def pick(self, w, pos):
        idx = np.flatnonzero(w.alive)
        if len(idx) == 0:
            return None
        d = (w.x[idx] - pos[0]) ** 2 + (w.y[idx] - pos[1]) ** 2
        k = int(np.argmin(d))
        return int(idx[k]) if d[k] < 30 ** 2 else None

    def click(self, pos):
        """Клик по панели. Возвращает 'pause' | 'slower' | 'faster' | None."""
        self.focus_visible = False
        for name, r in self.hits.items():
            if r.collidepoint(pos):
                self.focus = name
                return self.activate(name)
        return None

    def activate(self, name):
        if name == "fx":
            self.fx = not self.fx
            return None
        if name == "rays":
            self.show_rays = not self.show_rays
            return None
        return name

    def move_focus(self, step):
        o = self.FOCUS_ORDER
        i = o.index(self.focus) if self.focus in o else (-1 if step > 0 else 0)
        self.focus = o[(i + step) % len(o)]
        self.focus_visible = True

    def activate_focus(self):
        if self.focus and self.focus_visible:
            return self.activate(self.focus)
        return None

    # --------------------------------------------------------- рисование
    def draw(self, w, stats, sel, speed, paused, fps):
        c, scr = self.cfg, self.screen
        now = time.time()
        if not paused:
            self.t += min(0.1, now - self.t_prev)
        self.t_prev = now
        self.bank.new_frame()
        self.scen.draw_back(scr, self.t, self.fx)
        role = classify(w.S, w.alive, w.age, c.role_min_age, c.role_bite_frac)
        if c.battle:
            self._draw_zone(w)
        self._draw_food(w)
        self._draw_fish(w, role)
        if sel is not None and w.alive[sel]:
            self._draw_selected_world(w, sel)
        if w.player_idx is not None and w.alive[w.player_idx]:
            self._draw_player(w, w.player_idx)
        self.scen.draw_front(scr, self.t, self.fx)
        self._banner(w)
        self._panel(w, stats, sel, role, speed, paused, fps)

    def _draw_food(self, w):
        idx = np.flatnonzero(w.fa)
        if not len(idx):
            return
        bob = np.sin(self.t * 1.4 + idx * 0.7) * 1.5
        xs = (w.fx[idx] - 6).tolist()
        ys = (w.fy[idx] - 6 + bob).tolist()
        var = (idx % art.N_BREAD).tolist()
        bread = self.bread
        self.screen.blits([(bread[v], (x, y)) for v, x, y in zip(var, xs, ys)],
                          False)

    def _draw_fish(self, w, role):
        scr, bank = self.screen, self.bank
        idx = np.flatnonzero(w.alive)
        if not len(idx):
            return
        T = w.T[idx]
        hue = (T[:, 2] * art.N_HUE).astype(int) % art.N_HUE
        pat = np.clip((T[:, 3] * art.N_PAT).astype(int), 0, art.N_PAT - 1)
        rb = np.clip(np.rint(5.0 + 5.0 * T[:, 0]), 7, 20).astype(int)
        self.ph[idx] += 0.12 + 0.30 * np.clip(w.spd[idx] / self.cfg.max_speed,
                                              0, 1.5)
        frame = self.ph[idx].astype(int) % art.N_FRAMES
        ang = w.ang[idx]
        flip = np.cos(ang) < 0
        q = np.rint(-np.degrees(ang) / 4.0).astype(int)
        cache = self.rot_cache
        if len(cache) > 6000:
            cache.clear()
        order = np.argsort(rb, kind="stable")
        px, py = w.x[idx].tolist(), w.y[idx].tolist()
        seq, rings = [], []
        for k in order.tolist():
            h_, p_, r_, f_ = int(hue[k]), int(pat[k]), int(rb[k]), int(frame[k])
            got = bank.get_fish(h_, p_, r_, f_) or bank.get_any(h_, p_, r_, f_)
            if got is None:
                self._fallback_fish(w, idx[k], T[k])
                continue
            fl, qq = bool(flip[k]), int(q[k])
            ck = (h_, p_, r_, f_, fl, qq)
            spr = cache.get(ck)
            if spr is None:
                spr = pygame.transform.rotozoom(got[1] if fl else got[0],
                                                qq * 4.0, 1.0)
                cache[ck] = spr
            seq.append((spr, (px[k] - spr.get_width() / 2,
                              py[k] - spr.get_height() / 2)))
            ring = ROLE_RING.get(int(role[idx[k]]))
            if ring:
                rings.append((int(px[k]), int(py[k]),
                              int(7 + 8 * T[k, 0]), ring))
        scr.blits(seq, False)
        for x, y, r, col in rings:
            gfxdraw.aacircle(scr, x, y, r, col)
            gfxdraw.aacircle(scr, x, y, r + 1, col)

    def _fallback_fish(self, w, i, T):
        """Пока спрайт достраивается: простой силуэт, чтобы не мигало."""
        px, py, a = w.x[i], w.y[i], w.ang[i]
        r = 3 + 3 * T[0]
        col = tuple(int(v * 255) for v in
                    __import__("colorsys").hsv_to_rgb(T[2] % 1.0, 0.7, 0.95))
        pts = [(px + math.cos(a) * r * 2, py + math.sin(a) * r * 2),
               (px + math.cos(a + 1.57) * r, py + math.sin(a + 1.57) * r),
               (px - math.cos(a) * r * 1.6, py - math.sin(a) * r * 1.6),
               (px + math.cos(a - 1.57) * r, py + math.sin(a - 1.57) * r)]
        pygame.draw.polygon(self.screen, col, pts)

    def _draw_zone(self, w):
        r = int(w.zone_radius())
        zx, zy = int(w.zone_c[0]), int(w.zone_c[1])
        if (zx, zy, r) != self._zone_key:
            self.storm.fill((220, 30, 40, 75))
            if r > 0:
                pygame.draw.circle(self.storm, (0, 0, 0, 0), (zx, zy), r)
            self._zone_key = (zx, zy, r)
        self.screen.blit(self.storm, (0, 0))
        if 0 < r < 4000:
            for rad in (r, r + 1):
                gfxdraw.aacircle(self.screen, zx, zy, rad, (255, 255, 255))

    def _banner(self, w):
        n = len(w.winners)
        if n > self.seen_rounds:
            r = w.winners[-1]
            self.banner_text = ("Раунд %d: победитель продержался %d с, "
                                "убийств %d, еды %d"
                                % (r[0], r[1] / 60, r[2], r[6]))
            self.banner_until = time.time() + 3.0
        self.seen_rounds = n
        left = self.banner_until - time.time()
        if left > 0:
            t = self.F.text("base", self.banner_text, WHITE, "SemiBold")
            bw, bh = t.get_width() + 48, 48
            pill = rr(bw, bh, 24, (0, 0, 0, 204), (49, 49, 49, 255)).copy()
            pill.set_alpha(int(255 * min(1.0, left * 3)))
            x = (self.cfg.world_w - bw) // 2
            self.screen.blit(pill, (x, 24))
            t.set_alpha(int(255 * min(1.0, left * 3)))
            self.screen.blit(t, (x + 24, 24 + (bh - t.get_height()) // 2))

    def _draw_selected_world(self, w, i):
        px, py = int(w.x[i]), int(w.y[i])
        gfxdraw.aacircle(self.screen, px, py, 26, (255, 255, 255))
        gfxdraw.aacircle(self.screen, px, py, 27, (255, 255, 255))
        if self.show_rays:
            for k in range(5):
                a = w.ang[i] + math.radians(-60 + 30 * k)
                pygame.draw.aaline(
                    self.screen, (255, 255, 160), (px, py),
                    (px + math.cos(a) * self.cfg.vision,
                     py + math.sin(a) * self.cfg.vision))

    def _draw_player(self, w, i):
        px, py = int(w.x[i]), int(w.y[i])
        r = int(3 + 3 * w.T[i, 0])
        pygame.draw.circle(self.screen, (255, 215, 0), (px, py), r + 12, 2)
        self._text("ВЫ", px - 10, py - r - 34, (255, 215, 0))
        cap = self.cfg.energy_cap * w.T[i, 0]
        frac = max(0.0, min(1.0, w.energy[i] / cap))
        bw = 36
        bx, by = px - bw // 2, py + r + 14
        self.screen.blit(rr(bw, 4, 2, BG4), (bx, by))
        self.screen.blit(rr(max(4, int(bw * frac)), 4, 2, C_GRAZE), (bx, by))

    def _text(self, s, x, y, col=(230, 230, 230), big=False):
        t = self.F.text("lg" if big else "sm", s, col,
                        "SemiBold" if big else "Medium")
        self.screen.blit(t, (x, y))

    # ----------------------------------------------------------- панель
    def _hover(self, name):
        r = self.hits.get(name)
        return bool(r and r.collidepoint(pygame.mouse.get_pos()))

    def _button(self, name, rect, label=None, primary=False, glyph=None):
        scr, F = self.screen, self.F
        self.hits[name] = rect
        hot = self._hover(name)
        down = hot and pygame.mouse.get_pressed()[0]
        fill = (217, 217, 217) if (primary and hot) else WHITE if primary \
            else BG4 if hot else BG3
        ink = INK if primary else WHITE
        r = rect.inflate(-rect.w * 0.04, -rect.h * 0.04) if down else rect
        scr.blit(rr(r.w, r.h, 8, fill), r.topleft)
        cx, cy = r.center
        if label:
            t = F.text("base", label, ink, "SemiBold")
            scr.blit(t, (cx - t.get_width() // 2, cy - t.get_height() // 2))
        if glyph == "minus":
            pygame.draw.line(scr, ink, (cx - 6, cy), (cx + 5, cy), 2)
        elif glyph == "plus":
            pygame.draw.line(scr, ink, (cx - 6, cy), (cx + 5, cy), 2)
            pygame.draw.line(scr, ink, (cx, cy - 6), (cx, cy + 5), 2)
        self._focus_ring(name, r)

    def _focus_ring(self, name, rect, radius=8):
        if self.focus_visible and self.focus == name:
            fr = rect.inflate(8, 8)
            self.screen.blit(rr(fr.w, fr.h, radius + 4, (0, 0, 0, 0),
                                (255, 255, 255, 255)), fr.topleft)

    def _switch(self, name, x, y, w, label, on):
        scr, F = self.screen, self.F
        row = pygame.Rect(x, y, w, 32)
        self.hits[name] = row
        hot = self._hover(name)
        t = F.text("sm", label, WHITE if hot else MUTED)
        scr.blit(t, (x, y + (32 - t.get_height()) // 2))
        track = WHITE if on else BG4
        if hot:
            track = mix(track, WHITE, 0.15)
        tx = x + w - 40
        scr.blit(rr(40, 24, 12, track), (tx, y + 4))
        scr.blit(rr(16, 16, 8, INK if on else MUTED),
                 (tx + (20 if on else 4), y + 8))
        self._focus_ring(name, pygame.Rect(tx, y + 4, 40, 24), 12)

    def _card(self, x, y, w, h):
        self.screen.blit(rr(w, h, 24, BG1, BG4), (x, y))

    def _graph(self, x, y, w, h, series, colors):
        scr = self.screen
        scr.blit(rr(w, h, 8, BG2), (x, y))
        ymax = max([float(s.max()) for s in series if len(s)] + [1.0])
        for g in (0.25, 0.5, 0.75):
            gy = y + 4 + int((h - 8) * g)
            pygame.draw.line(scr, BG3, (x + 4, gy), (x + w - 5, gy))
        for s, col in zip(series, colors):
            if len(s) < 2:
                continue
            pts = [(x + 4 + (w - 8) * k / (len(s) - 1),
                    y + h - 4 - (h - 8) * float(v) / ymax)
                   for k, v in enumerate(s)]
            pygame.draw.lines(scr, col, False, pts, 2)
        return ymax

    def _legend(self, x, y, items):
        for label, col in items:
            gfxdraw.filled_circle(self.screen, x + 4, y + 10, 4, col)
            gfxdraw.aacircle(self.screen, x + 4, y + 10, 4, col)
            t = self.F.text("xs", label, MUTED)
            self.screen.blit(t, (x + 14, y + 10 - t.get_height() // 2))
            x += 14 + t.get_width() + 12

    def _title(self, x, y, title, right=None):
        t = self.F.text("base", title, WHITE, "SemiBold")
        self.screen.blit(t, (x, y))
        if right:
            r = self.F.text("xs", right, MUTED)
            self.screen.blit(r, (x + self.PW - r.get_width(),
                                 y + (t.get_height() - r.get_height()) // 2))

    def _tile(self, x, y, w, label, value, dot=None):
        scr, F = self.screen, self.F
        scr.blit(rr(w, 56, 8, BG3), (x, y))
        lx = x + 12
        if dot:
            gfxdraw.filled_circle(scr, lx + 4, y + 20, 4, dot)
            gfxdraw.aacircle(scr, lx + 4, y + 20, 4, dot)
            lx += 14
        scr.blit(F.text("xs", label, MUTED), (lx, y + 8))
        scr.blit(F.text("xl", value, WHITE, "SemiBold"), (x + 12, y + 26))

    def _panel(self, w, stats, sel, role, speed, paused, fps):
        c, scr, F = self.cfg, self.screen, self.F
        x0 = c.world_w
        pygame.draw.rect(scr, BG0, (x0, 0, c.panel_w, c.window_h))
        self.hits = {}
        cx = x0 + 16
        cw = c.panel_w - 32
        self.PW = cw - 32
        ix = cx + 16
        a = stats.arrays()
        win = slice(-400, None)
        n = w.n_alive
        cnt = [int(((role == r) & w.alive).sum()) for r in range(3)]
        y = 16

        # --- обзор
        h = 152
        self._card(cx, y, cw, h)
        t = F.text("lg", "Аквариум", WHITE, "SemiBold")
        scr.blit(t, (ix, y + 16))
        chip_x = cx + cw - 16
        state = "Пауза" if paused else "×%d" % speed
        for label in (state, "Битва" if c.battle else "Песочница"):
            ct = F.text("xs", label, WHITE, "SemiBold")
            cwid = ct.get_width() + 16
            chip_x -= cwid
            scr.blit(rr(cwid, 24, 8, BG3), (chip_x, y + 16))
            scr.blit(ct, (chip_x + 8, y + 16 + (24 - ct.get_height()) // 2))
            chip_x -= 8
        if c.battle:
            sub = "Раунд %d, прошло %d с" % (w.round_no, w.round_tick / 60)
        else:
            sub = "Тик %d" % w.tick
        scr.blit(F.text("sm", sub, MUTED), (ix, y + 16 + 28 + 4))
        tw = (self.PW - 16) // 3
        ty = y + h - 16 - 56
        total = c.round_size if c.battle else c.max_creatures
        self._tile(ix, ty, tw, "Живых", "%d/%d" % (n, total))
        self._tile(ix + tw + 8, ty, tw, "Хлеба", str(int(w.fa.sum())))
        self._tile(ix + 2 * (tw + 8), ty, tw, "Кадров/с", "%d" % fps)
        y += h + 12

        # --- зона
        if c.battle:
            h = 96
            self._card(cx, y, cw, h)
            if w.round_tick <= c.zone_delay:
                left = (c.zone_delay - w.round_tick) / 60
                right = "шторм через %d с" % left
                frac = 1.0 - (c.zone_delay - w.round_tick) / c.zone_delay
            else:
                right = "радиус %d" % w.zone_radius()
                frac = 1.0 - w.zone_radius() / max(1.0, w.zone_r0)
            self._title(ix, y + 16, "Безопасная зона", right)
            by = y + 16 + 24 + 8
            scr.blit(rr(self.PW, 8, 4, BG3), (ix, by))
            scr.blit(rr(max(8, int(self.PW * max(0.0, min(1.0, frac)))), 8,
                        4, C_PRED if w.round_tick > c.zone_delay else MUTED),
                     (ix, by))
            cap = ("зона сужается" if w.round_tick > c.zone_delay
                   else "зона ждёт начала сужения")
            scr.blit(F.text("xs", cap, MUTED), (ix, by + 16))
            y += h + 12

        # --- роли
        h = 16 + 24 + 8 + 56 + 16
        self._card(cx, y, cw, h)
        self._title(ix, y + 16, "Роли по поведению")
        ty = y + h - 16 - 56
        self._tile(ix, ty, tw, "Травоядные", str(cnt[GRAZER]), C_GRAZE)
        self._tile(ix + tw + 8, ty, tw, "Хищники", str(cnt[PREDATOR]), C_PRED)
        self._tile(ix + 2 * (tw + 8), ty, tw, "Паразиты", str(cnt[PARASITE]),
                   C_PARA)
        y += h + 12

        # --- население
        h = 132
        self._card(cx, y, cw, h)
        self._title(ix, y + 16, "Население")
        self._legend(ix + 100, y + 16 - 2, [("все", C_ALL), ("трав.", C_GRAZE),
                                            ("хищ.", C_PRED),
                                            ("пар.", C_PARA)])
        self._graph(ix, y + 16 + 28, self.PW, h - 16 - 28 - 16,
                    [a["pop"][win], a["grazer"][win], a["predator"][win],
                     a["parasite"][win]], [C_ALL, C_GRAZE, C_PRED, C_PARA])
        y += h + 12

        if sel is not None and w.alive[sel]:
            y = self._selected_card(w, sel, role, cx, y, cw)
        else:
            if c.battle:
                wa = np.array(w.winners, dtype=np.float64).reshape(-1, 8)[-80:]
                g1 = ("Обучение победителей", [wa[:, 3] * 100.0], [C_GOLD])
                g2 = ("Еда: победитель и среднее", [wa[:, 6], wa[:, 7]],
                      [C_ORANGE, C_GRAZE])
            else:
                g1 = ("Скорость обучения", [a["mean_eta"][win] * 100.0],
                      [C_GOLD])
                g2 = ("Укусы и убийства", [a["bites"][win], a["kills"][win]],
                      [C_ORANGE, C_PRED])
            for title, series, cols in (g1, g2):
                h = 108
                self._card(cx, y, cw, h)
                ymax = max([float(s.max()) for s in series if len(s)] + [1.0])
                self._title(ix, y + 16, title, "макс %s" % fnum(ymax))
                self._graph(ix, y + 16 + 24, self.PW, h - 16 - 24 - 16,
                            series, cols)
                y += h + 12
            if c.battle and w.winners:
                rows = w.winners[-3:][::-1]
                h = 16 + 24 + 8 + 20 * len(rows) + 16 - 4
                self._card(cx, y, cw, h)
                self._title(ix, y + 16, "Последние победители")
                yy = y + 16 + 24 + 8
                for r in rows:
                    line = "Р%d  %d с  убийств %d  еды %d  размер %s" % (
                        r[0], r[1] / 60, r[2], r[6], fnum(r[4], 2))
                    scr.blit(F.text("xs", line, WHITE), (ix, yy))
                    yy += 20
                y += h + 12

        # --- управление
        h = 16 + 40 + 12 + 32 + 16
        y = max(y, c.window_h - h - 16) if y + h + 16 <= c.window_h else y
        self._card(cx, y, cw, h)
        by = y + 16
        sp_w = 40 + 56 + 40
        self._button("pause", pygame.Rect(ix, by, self.PW - sp_w - 8, 40),
                     "Пуск" if paused else "Пауза", primary=True)
        sx = ix + self.PW - sp_w
        self._button("slower", pygame.Rect(sx, by, 40, 40), glyph="minus")
        lab = F.text("base", "×%d" % speed, WHITE, "SemiBold")
        scr.blit(lab, (sx + 40 + (56 - lab.get_width()) // 2,
                       by + (40 - lab.get_height()) // 2))
        self._button("faster", pygame.Rect(sx + 96, by, 40, 40), glyph="plus")
        half = (self.PW - 16) // 2
        self._switch("fx", ix, by + 52, half, "Эффекты воды", self.fx)
        self._switch("rays", ix + half + 16, by + 52, half, "Лучи зрения",
                     self.show_rays)

    def _selected_card(self, w, i, role, cx, y, cw):
        scr, F = self.screen, self.F
        ix = cx + 16
        S = w.S[i]
        lines = ("Размер %s   Обучение %s   Поколение %d"
                 % (fnum(w.T[i, 0], 2), fnum(w.T[i, 1], 3), int(w.gen[i])),
                 "Энергия %d   Возраст %d" % (w.energy[i], w.age[i]),
                 "Уровень %d   Опыт %d" % (int(w.lvl[i]), int(w.xp[i])),
                 "От еды %d   От убийств %d   Высосано %d"
                 % (S[0], S[1], S[2]),
                 "Укусов %d   Убийств %d" % (S[3], S[4]),
                 "Съедено еды %d   Рост +%s" % (S[5], fnum(S[6], 2)))
        groups = (("Входы", w.inp[i], 0.0), ("Скрытый слой", w.h[i], -1.0),
                  ("Выходы", w.out[i], -1.0))
        h = 16 + 28 + 8 + 20 * len(lines) + 8 + 3 * (18 + 28 + 4) + 12
        self._card(cx, y, cw, h)
        yy = y + 16
        scr.blit(F.text("base", "Рыбка #%d, %s" % (i, ROLE_RU[int(role[i])]),
                        WHITE, "SemiBold"), (ix, yy))
        yy += 28 + 8
        for line in lines:
            scr.blit(F.text("sm", line, WHITE), (ix, yy))
            yy += 20
        yy += 8
        for label, vec, lo in groups:
            scr.blit(F.text("xs", label, MUTED), (ix, yy))
            yy += 18
            bw = min(14, (self.PW - (len(vec) - 1)) // len(vec))
            for k, v in enumerate(vec):
                frac = (float(v) - lo) / (1.0 - lo)
                hh = int(28 * min(1.0, max(0.0, frac)))
                bx = ix + k * (bw + 1)
                scr.blit(rr(bw, 28, 2, BG3), (bx, yy))
                if hh:
                    scr.blit(rr(bw, max(2, hh), 2, C_BLUE),
                             (bx, yy + 28 - max(2, hh)))
            yy += 28 + 4
        return y + h + 12
