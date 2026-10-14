"""Мир: хранение состояния (структура массивов) и шаг симуляции."""
import math
import os

import numpy as np

from . import kernels as K
from .config import Config
from .evolution import N_STAGES
from .genome import (N_H, N_IN, N_OUT, N_PARAMS, N_TRAITS, SIZE_MIN,
                     canonical, expand_genome, mutate_traits, mutate_weights,
                     pairwise_dist, random_genomes, random_traits, seed_numba)

STATE_KEYS = ["alive", "x", "y", "ang", "spd", "energy", "age", "T", "G", "L",
              "mem", "gen", "S", "inp", "h", "out", "gain", "fx", "fy", "fa",
              "died_t", "in_round", "xp", "lvl", "species", "stage"]


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
        self.species = np.zeros(n, np.int32)
        self.stage = np.zeros(n, np.int32)      # стадия эволюции (0 = клетка)
        self.evo_log = []                       # (раунд, было, стало)
        self.evo_base = np.zeros((n, 2))        # еда/убийства на последней эволюции
        self.auto_restart = True                # GUI выключает: даёт показать WIN
        self.match_over = False
        self.win_idx = -1
        self.next_sp = 1                       # следующий свободный id вида
        self.sp_reps = np.zeros((0, N_PARAMS))  # представители видов (канонические)
        self.sp_ids = np.zeros(0, np.int32)     # их id
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
        self.soc_own = np.zeros(n)
        self.soc_other = np.zeros(n)
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
        self.species[idx] = 0
        self.stage[idx] = 0
        self.S[idx] = 0.0          # счётчики матча
        self.evo_base[idx] = 0.0
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
        # зоны нет: радиус бесконечный (ни шторма, ни ограничения спавна еды)
        return 1e9

    def _update_zone(self):
        self.P[K.P_ZX], self.P[K.P_ZY] = self.zone_c
        self.P[K.P_ZR] = self.zone_radius()

    def start_round(self, parents=None, stages=None):
        """parents = (genomes, traits, gens, species) от предыдущего раунда,
        stages = стадии эволюции этих родителей."""
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
            pg, pt, pgen, psp = parents
            m = len(pg)
            wts = 1.0 / np.arange(1, m + 1)
            pick = self.rng.choice(m, size=k, p=wts / wts.sum())
            pick[0] = 0
            for i in range(k):
                self.G[i] = pg[pick[i]]
                self.T[i] = pt[pick[i]]
                self.gen[i] = pgen[pick[i]] + 1
                self.species[i] = psp[pick[i]]
                self.stage[i] = 0 if stages is None else stages[pick[i]]
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

    def assign_species(self):
        """Перекластеризация всей популяции раунда. Возвращает ranking().
        1) Leader-кластеризация по матрице расстояний канонических геномов.
        2) Каждый новый кластер получает id вида, чей прежний представитель
           ближе всего (и не занят), иначе новый id. Так виды сохраняют
           идентичность между раундами, а не переназначаются заново."""
        c = self.cfg
        rk = self.ranking()
        m = len(rk)
        if m == 0:
            return rk
        if c.spec_dist <= 0:
            self.species[rk] = 0
            return rk
        Gc = canonical(self.G[rk])
        D = pairwise_dist(Gc, Gc)
        labels = np.full(m, -1, np.int64)
        leaders = []
        for j in range(m):                 # rk уже по рейтингу
            if labels[j] >= 0:
                continue
            members = (labels < 0) & (D[j] < c.spec_dist)
            labels[members] = len(leaders)
            leaders.append(j)
        ids = np.empty(len(leaders), np.int32)
        taken = set()
        prev_d = None
        if len(self.sp_ids):
            prev_d = pairwise_dist(Gc[leaders], self.sp_reps)
        for li in range(len(leaders)):
            sid = None
            if prev_d is not None:
                for p in np.argsort(prev_d[li]):
                    if prev_d[li, p] >= c.spec_dist:
                        break
                    pid = int(self.sp_ids[p])
                    if pid not in taken:
                        sid = pid
                        break
            if sid is None:
                sid = self.next_sp
                self.next_sp += 1
            taken.add(sid)
            ids[li] = sid
        self.species[rk] = ids[labels]
        self.sp_reps = Gc[leaders].copy()
        self.sp_ids = ids.copy()
        return rk

    def pick_parents(self, rk):
        """Выбор родителей. При spec_dist <= 0 — топ hof_size.
        Иначе из spec_pool лучших берём не более spec_quota рыб с каждого вида."""
        c = self.cfg
        if c.spec_dist <= 0:
            top = rk[:c.hof_size]
            return top, np.zeros(len(top), np.int32)
        pool = rk[:c.spec_pool]
        counts = {}
        chosen, chosen_sp = [], []
        for i in pool:                     # уже в порядке рейтинга
            s = int(self.species[i])
            if counts.get(s, 0) < c.spec_quota:
                counts[s] = counts.get(s, 0) + 1
                chosen.append(i)
                chosen_sp.append(s)
        chosen = chosen[:c.hof_size]
        chosen_sp = chosen_sp[:c.hof_size]
        return np.array(chosen, np.int64), np.array(chosen_sp, np.int32)

    def end_round(self):
        rk = self.assign_species()
        win = int(rk[0])
        self.win_idx = win
        self.winners.append((self.round_no, self.round_tick,
                             float(self.S[win, 4]), float(self.T[win, 1]),
                             float(self.T[win, 0]), int(self.gen[win]),
                             float(self.S[win, 5]),
                             float(self.S[rk, 5].mean())))
        top, sp = self.pick_parents(rk)
        pt = self.T[top].copy()
        pt[:, 0] = np.maximum(SIZE_MIN, pt[:, 0] - self.S[top, 6])
        self.parents = (self.G[top].copy(), pt, self.gen[top].copy(), sp)
        self.round_no += 1
        if self.auto_restart:
            self.restart_match()
        else:
            self.match_over = True      # GUI покажет WIN, потом вызовет restart_match

    def restart_match(self):
        """Новый матч: все снова клетки, геномы — от победителей прошлого."""
        self.match_over = False
        self.start_round(self.parents)

    def expert_usage(self):
        """Средняя доля каждого эксперта по живым рыбам (для статистики).
        ВНИМАНИЕ: порядок раскладки Wg не проверен против kernels.think."""
        from .genome import N_EXPERTS, N_IN as _NI, OFF_GATE_B
        idx = np.flatnonzero(self.alive)
        if len(idx) == 0:
            return np.zeros(N_EXPERTS)
        Wg = self.G[idx, 0:OFF_GATE_B].reshape(-1, _NI, N_EXPERTS)
        bg = self.G[idx, OFF_GATE_B:OFF_GATE_B + N_EXPERTS]
        z = np.einsum("ki,kie->ke", self.inp[idx], Wg) + bg
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        return p.mean(axis=0)

    def step(self):
        if self.match_over:
            return
        c, P = self.cfg, self.P
        self._update_zone()
        K.learn(self.alive, self.gain, self.T[:, 1], self.G, self.L,
                self.inp, self.h, self.out, P)
        self.gain[:] = 0.0
        cs, ci = K.build_grid(self.x, self.y, self.alive, c.cell, self.gx,
                              self.gy)
        fs, fi = K.build_grid(self.fx, self.fy, self.fa, c.cell, self.gx,
                              self.gy)

        # 1. Внешние сенсоры в начале вектора входов
        K.sense(self.x, self.y, self.ang, self.T, self.alive, self.spd,
                self.energy, self.mem, cs, ci, self.fx,
                self.fy, self.fa, fs, fi, self.gx, self.gy, c.cell, P,
                self.inp)

        # 2. Рекуррентная память (RNN)
        n_ext_in = N_IN - N_H
        self.inp[:, n_ext_in:] = self.h

        # 3. Нейросеть с учётом памяти
        K.think(self.alive, self.inp, self.L, self.h, self.out)

        # рефлексы: бегство от сильных и охота на слабых (игрок не затрагивается)
        K.reflex(self.alive, self.x, self.y, self.ang, self.T, self.stage,
                 cs, ci, self.gx, self.gy, c.cell, P, self.out)
        if self.player_idx is not None and self.alive[self.player_idx]:
            turn, thrust, bite = self.player_action or (0.0, -1.0, False)
            pi = self.player_idx
            self.out[pi, 0] = turn
            self.out[pi, 1] = thrust
            self.out[pi, 2] = 1.0 if bite else -1.0
        prev_alive = self.alive.copy()
        K.move(self.alive, self.x, self.y, self.ang, self.spd, self.T, self.energy, self.age, self.out, self.bite_on, self.mem, self.gain, self.xp, self.lvl, self.stage, P)


        order = self.rng.permutation(np.flatnonzero(self.alive)).astype(
            np.int32)
        K.interact(order, self.x, self.y, self.ang, self.T, self.energy,
                    self.alive, self.bite_on, self.gain, self.S, self.ev,
                    self.fx, self.fy, self.fa, cs, ci, fs, fi, self.gx,
                    self.gy, c.cell, self.xp, self.lvl, self.stage, P)







                   
                   
                   
                   
        if c.battle:
            # эволюция по порогам (еда/убийства с прошлой эволюции), без сброса
            ev = K.evolve(self.alive, self.stage, self.S, self.evo_base,
                          N_STAGES - 1, c.evo_food, c.evo_kills)
            for i in ev:
                self.evo_log.append((self.round_no, int(self.stage[i]) - 1,
                                     int(self.stage[i])))
        self.deaths += K.reap(self.alive, self.energy, self.age, P)
        if not c.battle:
            boosted = self.tick < self.boost_until
            rmult = c.migrate_boost_rate_mult if boosted else 1.0
            smult = c.migrate_boost_sigma_mult if boosted else 1.0
            self.births += K.reproduce(
                self.alive, self.energy, self.age, self.T, self.G, self.L,
                self.x, self.y, self.ang, self.spd, self.mem, self.gen,
                self.S, self.inp, self.h, self.out, self.xp, self.lvl,
                self.stage, self.boost_mask, rmult, smult, P)
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

    def sprinkle(self, x, y, k=2, spread=10.0):
        """Бросить k кусочков корма рядом с точкой (x, y) внутри зоны."""
        zx, zy = self.zone_c
        zr = self.zone_radius()
        if (x - zx) ** 2 + (y - zy) ** 2 > zr * zr:
            return 0
        slots = np.flatnonzero(~self.fa)[:k]
        if len(slots) < k:
            # карта заполнена едой: новый корм занимает места старых кусочков
            busy = np.flatnonzero(self.fa)
            if len(busy):
                extra = self.rng.choice(busy, size=min(len(busy), k - len(slots)),
                                        replace=False)
                slots = np.concatenate([slots, extra])
        free = slots
        c = self.cfg
        for f in free:
            a = self.rng.uniform(0.0, 2.0 * math.pi)
            d = spread * math.sqrt(self.rng.random())
            self.fx[f] = min(c.world_w, max(0.0, x + d * math.cos(a)))
            self.fy[f] = min(c.world_h, max(0.0, y + d * math.sin(a)))
            self.fa[f] = True
        return len(free)

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
        if not os.path.exists(path):
            # Если файл автосейва не найден – просто продолжаем с пустым миром
            return
        d = np.load(path)
        for k in STATE_KEYS:
            if k not in d.files:
                continue
            arr = d[k]
            cur = getattr(self, k)
            if arr.shape == cur.shape:
                cur[...] = arr
                continue
            # сейв с меньшим лимитом существ/еды (например 150 -> 260):
            # старые слоты копируем, новые остаются пустыми (как при старте)
            if (arr.ndim == cur.ndim and arr.ndim >= 1
                    and arr.shape[1:] == cur.shape[1:]
                    and arr.shape[0] < cur.shape[0]):
                cur[:arr.shape[0]] = arr
                continue
            # старые сейвы: T был (n, 5) без sociality, теперь (n, 6)
            if k == "T" and arr.ndim == 2 and cur.ndim == 2 and \
                    arr.shape[0] == cur.shape[0] and \
                    arr.shape[1] == 5 and cur.shape[1] == 6:
                cur[:, :5] = arr
                cur[:, 5] = self.rng.uniform(0.0, 1.0, cur.shape[0])
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
        if "stage" not in d.files:            # старый сейв: все начинают клеткой
            self.stage[:] = 0
        self.tick = int(d["tick"])
        if getattr(self, "_pending_boost", False):
            self.boost_until = self.tick + self.cfg.migrate_boost_ticks
            self._pending_boost = False
        self.round_no = int(d["round_no"])
        self.round_tick = int(d["round_tick"])
        self.winners = [(int(r[0]), int(r[1]), r[2], r[3], r[4], int(r[5]),
                         r[6], r[7]) for r in d["winners"]]
        self._set_zone(*d["zone_c"])
        # представители видов не сохраняются: после загрузки виды
        # переназначатся в конце первого раунда, id продолжают нумерацию
        self.next_sp = int(self.species.max()) + 1
        self.sp_reps = np.zeros((0, N_PARAMS))
        self.sp_ids = np.zeros(0, np.int32)