"""Мир: хранение состояния (структура массивов) и шаг симуляции."""
import math
import os

import numpy as np

from . import kernels as K
from .config import Config
from .genome import (N_H, N_IN, N_OUT, N_PARAMS, N_TRAITS, SIZE_MIN,
                     expand_genome, mutate_traits, mutate_weights,
                     random_genomes, random_traits, seed_numba)

STATE_KEYS = ["alive", "x", "y", "ang", "spd", "energy", "age", "T", "G", "L",
              "mem", "gen", "S", "inp", "h", "out", "gain", "fx", "fy", "fa",
              "died_t", "in_round", "xp", "lvl"]


def _new_weight_mask(old_n_h, new_n_h, n_in, n_out):
    off_b1 = n_in * new_n_h
    off_w2 = off_b1 + new_n_h
    off_b2 = off_w2 + new_n_h * n_out
    mask = np.zeros(off_b2 + n_out, np.int8)
    w1 = mask[0:off_b1].reshape(n_in, new_n_h)
    w1[:, old_n_h:] = 1
    mask[off_b1:off_w2][old_n_h:] = 1
    w2 = mask[off_w2:off_b2].reshape(new_n_h, n_out)
    w2[old_n_h:, :] = 1
    return mask

WIN_COLS = ["round", "ticks", "kills", "eta", "size", "gen", "food",
            "mean_food"]


class World:
    def __init__(self, cfg=None, seed=0):
        self.cfg = cfg or Config()
        c = self.cfg
        self.rng = np.random.default_rng(seed)
        seed_numba(seed)
        self.P = K.make_params(c)
        n = c.max_creatures
        self.alive = np.zeros(n, bool)
        self.x = np.zeros(n)
        self.y = np.zeros(n)
        self.ang = np.zeros(n)
        self.spd = np.zeros(n)
        self.energy = np.zeros(n)
        self.age = np.zeros(n)
        self.gain = np.zeros(n)
        self.gen = np.zeros(n, np.int32)
        self.xp = np.zeros(n)
        self.lvl = np.zeros(n, np.int32)
        self.boost_mask = np.zeros(N_PARAMS, np.int8)
        self.boost_until = -1
        self.bite_on = np.zeros(n, bool)
        self.T = np.zeros((n, N_TRAITS))
        self.T[:, 0] = 1.0
        self.G = np.zeros((n, N_PARAMS))
        self.L = np.zeros((n, N_PARAMS))
        self.mem = np.zeros((n, 2))
        self.S = np.zeros((n, 7))
        self.inp = np.zeros((n, N_IN))
        self.h = np.zeros((n, N_H))
        self.out = np.zeros((n, N_OUT))
        m = c.max_food
        self.fx = np.zeros(m)
        self.fy = np.zeros(m)
        self.fa = np.zeros(m, bool)
        self.gx = int(math.ceil(c.world_w / c.cell))
        self.gy = int(math.ceil(c.world_h / c.cell))
        self.died_t = np.full(n, np.inf)
        self.in_round = np.zeros(n, bool)
        self.round_no = 1
        self.round_tick = 0
        self.winners = []
        self._set_zone(c.world_w / 2.0, c.world_h / 2.0)
        self.parents = None
        self.ev = np.zeros(3)
        self.tick = 0
        self.births = 0
        self.deaths = 0
        self.rescues = 0
        self.player_idx = None
        self.player_action = None
        if c.battle:
            self.start_round()
        else:
            self.spawn_random(c.init_creatures)
            K.spawn_food(self.fx, self.fy, self.fa, c.init_food, self.P)

    @property
    def n_alive(self):
        return int(self.alive.sum())

    def spawn_random(self, k):
        c = self.cfg
        idx = np.flatnonzero(~self.alive)[:k]
        k = len(idx)
        if k == 0:
            return 0
        self.G[idx] = random_genomes(self.rng, k, c.init_w_std)
        self.L[idx] = self.G[idx]
        self.T[idx] = random_traits(self.rng, k, c.init_eta_max)
        self.x[idx] = self.rng.uniform(0, c.world_w, k)
        self.y[idx] = self.rng.uniform(0, c.world_h, k)
        self.ang[idx] = self.rng.uniform(-math.pi, math.pi, k)
        self.spd[idx] = 0.0
        self.energy[idx] = 0.5 * c.energy_cap * self.T[idx, 0]
        self.age[idx] = 0.0
        self.gen[idx] = 0
        self.xp[idx] = 0.0
        self.lvl[idx] = 0
        for arr in (self.mem, self.S, self.inp, self.h, self.out):
            arr[idx] = 0.0
        self.alive[idx] = True
        return k

    def _set_zone(self, cx, cy):
        c = self.cfg
        self.zone_c = (float(cx), float(cy))
        self.zone_r0 = math.hypot(max(cx, c.world_w - cx),
                                  max(cy, c.world_h - cy)) + 1.0

    def zone_radius(self):
        c = self.cfg
        if not c.battle:
            return 1e9
        t = self.round_tick
        if t <= c.zone_delay:
            return self.zone_r0
        return self.zone_r0 * max(0.0, 1.0 - (t - c.zone_delay) /
                                  c.zone_shrink)

    def _update_zone(self):
        self.P[K.P_ZX], self.P[K.P_ZY] = self.zone_c
        self.P[K.P_ZR] = self.zone_radius()

    def start_round(self, parents=None):
        c = self.cfg
        self.alive[:] = False
        self.fa[:] = False
        self.energy[:] = 0.0
        self.gain[:] = 0.0
        self.died_t[:] = np.inf
        self.in_round[:] = False
        self.round_tick = 0
        self._set_zone(self.rng.uniform(0.25, 0.75) * c.world_w,
                       self.rng.uniform(0.25, 0.75) * c.world_h)
        k = self.spawn_random(c.round_size)
        if parents is not None:
            pg, pt, pgen = parents
            m = len(pg)
            wts = 1.0 / np.arange(1, m + 1)
            pick = self.rng.choice(m, size=k, p=wts / wts.sum())
            pick[0] = 0
            for i in range(k):
                self.G[i] = pg[pick[i]]
                self.T[i] = pt[pick[i]]
                self.gen[i] = pgen[pick[i]] + 1
                if i > 0:
                    boosted = self.tick < self.boost_until
                    rmult = c.migrate_boost_rate_mult if boosted else 1.0
                    smult = c.migrate_boost_sigma_mult if boosted else 1.0
                    mutate_weights(self.G[i], c.mut_rate, c.mut_sigma,
                                   c.mut_big_p, c.mut_big_mult,
                                   c.weight_clip, self.boost_mask, rmult,
                                   smult)
                    mutate_traits(self.T[i], c.trait_mut)
            self.L[:k] = self.G[:k]
            self.energy[:k] = 0.5 * c.energy_cap * self.T[:k, 0]
        self.in_round[:k] = True
        self._update_zone()
        K.spawn_food(self.fx, self.fy, self.fa, c.battle_init_food, self.P)

    def ranking(self):
        idx = np.flatnonzero(self.in_round)
        t = np.where(self.alive[idx], np.inf, self.died_t[idx])
        order = np.lexsort((-self.energy[idx], -self.S[idx, 4], -t))
        return idx[order]

    def end_round(self):
        rk = self.ranking()
        win = int(rk[0])
        self.winners.append((self.round_no, self.round_tick,
                             float(self.S[win, 4]), float(self.T[win, 1]),
                             float(self.T[win, 0]), int(self.gen[win]),
                             float(self.S[win, 5]),
                             float(self.S[rk, 5].mean())))
        top = rk[:self.cfg.hof_size]
        pt = self.T[top].copy()
        pt[:, 0] = np.maximum(SIZE_MIN, pt[:, 0] - self.S[top, 6])
        self.parents = (self.G[top].copy(), pt, self.gen[top].copy())
        self.round_no += 1
        self.start_round(self.parents)

    def step(self):
        c, P = self.cfg, self.P
        self._update_zone()
        K.learn(self.alive, self.gain, self.T[:, 1], self.G, self.L,
                self.inp, self.h, self.out, P)
        self.gain[:] = 0.0
        cs, ci = K.build_grid(self.x, self.y, self.alive, c.cell, self.gx,
                              self.gy)
        fs, fi = K.build_grid(self.fx, self.fy, self.fa, c.cell, self.gx,
                              self.gy)
        
        # 1. Заполняем внешние сенсоры в начале вектора входов
        K.sense(self.x, self.y, self.ang, self.T, self.alive, self.spd,
                self.energy, self.mem, cs, ci, self.fx, self.fy, self.fa, fs,
                fi, self.gx, self.gy, c.cell, P, self.inp)
        
        # 2. Подключаем рекуррентную память (RNN): передаем предыдущие скрытые состояния в хвост входов
        n_ext_in = N_IN - N_H
        self.inp[:, n_ext_in:] = self.h

        # 3. Рассчитываем нейросеть с учётом памяти
        K.think(self.alive, self.inp, self.L, self.h, self.out)

        if self.player_idx is not None and self.alive[self.player_idx]:
            turn, thrust, bite = self.player_action or (0.0, -1.0, False)
            pi = self.player_idx
            self.out[pi, 0] = turn
            self.out[pi, 1] = thrust
            self.out[pi, 2] = 1.0 if bite else -1.0
        prev_alive = self.alive.copy()
        K.move(self.alive, self.x, self.y, self.ang, self.spd, self.T,
               self.energy, self.age, self.out, self.bite_on, self.mem,
               self.gain, self.xp, self.lvl, P)
        order = self.rng.permutation(np.flatnonzero(self.alive)).astype(
            np.int32)
        K.interact(order, self.x, self.y, self.ang, self.T, self.energy,
                   self.alive, self.bite_on, self.gain, self.S, self.ev,
                   self.fx, self.fy, self.fa, cs, ci, fs, fi, self.gx,
                   self.gy, c.cell, self.xp, self.lvl, P)
        self.deaths += K.reap(self.alive, self.energy, self.age, P)
        if not c.battle:
            boosted = self.tick < self.boost_until
            rmult = c.migrate_boost_rate_mult if boosted else 1.0
            smult = c.migrate_boost_sigma_mult if boosted else 1.0
            self.births += K.reproduce(
                self.alive, self.energy, self.age, self.T, self.G, self.L,
                self.x, self.y, self.ang, self.spd, self.mem, self.gen,
                self.S, self.inp, self.h, self.out, self.xp, self.lvl,
                self.boost_mask, rmult, smult, P)
        rate = c.battle_food_per_tick if c.battle else c.food_per_tick
        n = int(rate)
        if self.rng.random() < rate - n:
            n += 1
        K.spawn_food(self.fx, self.fy, self.fa, n, P)
        if not c.battle and self.n_alive < c.rescue_below:
            self.spawn_random(c.rescue_count)
            self.rescues += 1
        self.tick += 1
        if c.battle:
            self.died_t[prev_alive & ~self.alive] = self.round_tick
            self.round_tick += 1
            if self.n_alive <= 1 or self.round_tick >= c.max_round_ticks:
                self.end_round()

    def save(self, path):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        wins = np.array(self.winners, dtype=np.float64).reshape(-1, 8)
        np.savez_compressed(path, tick=self.tick, round_no=self.round_no,
                            round_tick=self.round_tick, winners=wins,
                            zone_c=np.array(self.zone_c),
                            **{k: getattr(self, k) for k in STATE_KEYS})

    def load(self, path):
        d = np.load(path)
        for k in STATE_KEYS:
            if k not in d.files:
                continue
            arr = d[k]
            cur = getattr(self, k)
            if arr.shape == cur.shape:
                cur[...] = arr
                continue
            if k in ("G", "L") and arr.ndim == 2 and arr.shape[0] == \
                    cur.shape[0]:
                old_params = arr.shape[1]
                denom = N_IN + 1 + N_OUT
                old_n_h = (old_params - N_OUT) // denom
                if (old_n_h * denom + N_OUT == old_params
                        and 0 < old_n_h <= N_H):
                    cur[...] = expand_genome(arr, old_n_h, N_H, N_IN,
                                             N_OUT, self.rng)
                    if old_n_h < N_H:
                        self.boost_mask = _new_weight_mask(old_n_h, N_H,
                                                           N_IN, N_OUT)
                        self._pending_boost = True
                    continue
            if k in ("h", "inp", "out") and arr.shape[0] == cur.shape[0]:
                continue
            raise ValueError(f"save incompatible with config: {k}")
        self.tick = int(d["tick"])
        if getattr(self, "_pending_boost", False):
            self.boost_until = self.tick + self.cfg.migrate_boost_ticks
            self._pending_boost = False
        self.round_no = int(d["round_no"])
        self.round_tick = int(d["round_tick"])
        self.winners = [(int(r[0]), int(r[1]), r[2], r[3], r[4], int(r[5]),
                         r[6], r[7]) for r in d["winners"]]
        self._set_zone(*d["zone_c"])