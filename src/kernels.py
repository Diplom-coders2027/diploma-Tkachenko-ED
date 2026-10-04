"""Numba-ядра: сетка, сенсоры, нейросеть, физика, укусы, размножение."""
import math

import numpy as np
from numba import njit, prange

from .genome import (N_H, N_IN, N_OUT, N_PARAMS, N_RAYS, N_TRAITS, OFF_B1,
                     OFF_B2, OFF_W1, OFF_W2, SIZE_MAX, SIZE_MIN, mutate_traits,
                     mutate_weights, N_EXPERTS, N_HE, OFF_GATE_W, OFF_GATE_B, 
                     OFF_GATE_OUT, SINGLE_EXPERT_SIZE)

(P_FOOD_E, P_EAT_R0, P_EAT_R1, P_BITE_R0, P_BITE_R1, P_BITE_DMG, P_BITE_EFF,
 P_KILL_BONUS, P_BITE_COST, P_BITE_THR, P_CONE, P_TURN, P_MAXSPD, P_C_BASE,
 P_C_MOVE, P_C_BRAIN, P_ECAP, P_REPRO, P_MIN_AGE, P_MAXAGE, P_VISION,
 P_RSCALE, P_WCLIP, P_DECAY, P_MUT_RATE, P_MUT_SIGMA, P_MUT_BIG_P,
 P_MUT_BIG_MULT, P_TRAIT_MUT, P_W, P_H, P_LIGHT, P_GROW_EVERY, P_GROW, P_ZX,
 P_ZY, P_ZR, P_STORM, P_RB_FOOD, P_RB_BITE, P_RB_KILL, P_XP_FOOD, P_XP_BITE,
 P_XP_KILL, P_XP_BASE, P_XP_GROWTH, P_LVL_MAX, P_LVL_DMG, P_LVL_SPD,
 P_LVL_ECAP, P_MOB_BONUS, P_MOB_MAX) = range(52)
N_KP = 52


def make_params(c):
    v = [c.food_energy, c.eat_r0, c.eat_r1, c.bite_r0, c.bite_r1, c.bite_dmg,
         c.bite_eff, c.kill_bonus, c.bite_cost, c.bite_threshold,
         math.radians(c.bite_cone_deg), c.turn_rate, c.max_speed, c.cost_base,
         c.cost_move, c.cost_brain, c.energy_cap, c.repro_frac,
         c.min_repro_age, c.max_age, c.vision, c.reward_scale, c.weight_clip,
         c.plasticity_decay, c.mut_rate, c.mut_sigma, c.mut_big_p,
         c.mut_big_mult, c.trait_mut, c.world_w, c.world_h, c.light_pow,
         c.grow_every, c.grow_amount, c.world_w / 2.0, c.world_h / 2.0, 1e9,
         c.storm_damage, c.learn_bonus_food, c.learn_bonus_bite,
         c.learn_bonus_kill, c.xp_per_food, c.xp_per_bite, c.xp_per_kill,
         c.xp_level_base, c.xp_level_growth, c.lvl_max, c.lvl_dmg_per_lvl,
         c.lvl_speed_per_lvl, c.lvl_ecap_per_lvl, c.mob_bonus_per_attacker,
         c.mob_bonus_max]
    assert len(v) == N_KP
    return np.array(v, dtype=np.float64)


@njit(cache=True)
def level_of(xp, base, growth, maxlvl):
    lvl = 0
    need = base
    rem = xp
    while lvl < maxlvl and rem >= need:
        rem -= need
        need *= growth
        lvl += 1
    return lvl


@njit(cache=True)
def wrap_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


@njit(cache=True)
def build_grid(x, y, active, cs, gx, gy):
    ncell = gx * gy
    n = x.shape[0]
    start = np.zeros(ncell + 1, np.int32)
    cid = np.full(n, -1, np.int32)
    for i in range(n):
        if active[i]:
            cx = min(gx - 1, max(0, int(x[i] / cs)))
            cy = min(gy - 1, max(0, int(y[i] / cs)))
            c = cy * gx + cx
            cid[i] = c
            start[c + 1] += 1
    for c in range(ncell):
        start[c + 1] += start[c]
    fill = start[:ncell].copy()
    items = np.empty(start[ncell], np.int32)
    for i in range(n):
        c = cid[i]
        if c >= 0:
            items[fill[c]] = i
            fill[c] += 1
    return start, items


@njit(cache=True, parallel=True)
def learn(alive, gain, eta, G, L, inp, h, out, P):
    rs = P[P_RSCALE]
    wc = P[P_WCLIP]
    dec = P[P_DECAY]
    for i in prange(alive.shape[0]):
        if not alive[i]:
            continue
        for p in range(N_PARAMS):
            L[i, p] += dec * (G[i, p] - L[i, p])
        r = min(1.0, max(-1.0, gain[i] * rs))
        e = eta[i] * r
        if e == 0.0:
            continue
        for a in range(N_IN):
            xa = inp[i, a]
            if xa == 0.0:
                continue
            for j in range(N_H):
                p = OFF_W1 + a * N_H + j
                L[i, p] = min(wc, max(-wc, L[i, p] + e * xa * h[i, j]))
        for j in range(N_H):
            for b in range(N_OUT):
                p = OFF_W2 + j * N_OUT + b
                L[i, p] = min(wc, max(-wc, L[i, p] + e * h[i, j] * out[i, b]))


@njit(cache=True, parallel=True)
def sense(x, y, ang, T, alive, spd, energy, mem, cstart, citems, fx, fy, fa,
          fstart, fitems, gx, gy, cs, P, inp):
    R = P[P_VISION]
    R2 = R * R
    RW = math.pi / 6.0
    
    for i in prange(x.shape[0]):
        if not alive[i]:
            continue
        fd = np.empty(N_RAYS)
        cd = np.empty(N_RAYS)
        csz = np.empty(N_RAYS)
        ckin = np.empty(N_RAYS)
        for k in range(N_RAYS):
            fd[k] = R
            cd[k] = R
            csz[k] = 0.0
            ckin[k] = 0.0
        cx = min(gx - 1, max(0, int(x[i] / cs)))
        cy = min(gy - 1, max(0, int(y[i] / cs)))
        for yy in range(max(0, cy - 1), min(gy - 1, cy + 1) + 1):
            for xx in range(max(0, cx - 1), min(gx - 1, cx + 1) + 1):
                c = yy * gx + xx
                for t in range(cstart[c], cstart[c + 1]):
                    j = citems[t]
                    if j == i or not alive[j]:
                        continue
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    d2 = dx * dx + dy * dy
                    if d2 >= R2:
                        continue
                    rel = wrap_angle(math.atan2(dy, dx) - ang[i])
                    k = int(math.floor((rel + 2.5 * RW) / RW))
                    if k < 0 or k >= N_RAYS:
                        continue
                    d = math.sqrt(d2)
                    if d < cd[k]:
                        cd[k] = d
                        csz[k] = T[j, 0] / T[i, 0]
                        ckin[k] = 1.0 - (abs(T[i, 2] - T[j, 2]) +
                                         abs(T[i, 3] - T[j, 3]) +
                                         abs(T[i, 4] - T[j, 4])) / 3.0
                for t in range(fstart[c], fstart[c + 1]):
                    f = fitems[t]
                    if not fa[f]:
                        continue
                    dx = fx[f] - x[i]
                    dy = fy[f] - y[i]
                    d2 = dx * dx + dy * dy
                    if d2 >= R2:
                        continue
                    rel = wrap_angle(math.atan2(dy, dx) - ang[i])
                    k = int(math.floor((rel + 2.5 * RW) / RW))
                    if k < 0 or k >= N_RAYS:
                        continue
                    d = math.sqrt(d2)
                    if d < fd[k]:
                        fd[k] = d
        for k in range(N_RAYS):
            inp[i, k * 4] = 1.0 - fd[k] / R
            inp[i, k * 4 + 1] = 1.0 - cd[k] / R
            if cd[k] < R:
                inp[i, k * 4 + 2] = min(3.0, csz[k]) / 3.0
                inp[i, k * 4 + 3] = ckin[k]
            else:
                inp[i, k * 4 + 2] = 0.0
                inp[i, k * 4 + 3] = 0.0
        inp[i, N_RAYS * 4] = energy[i] / (P[P_ECAP] * T[i, 0])
        inp[i, N_RAYS * 4 + 1] = spd[i] / P[P_MAXSPD]
        inp[i, N_RAYS * 4 + 2] = mem[i, 0]
        inp[i, N_RAYS * 4 + 3] = mem[i, 1]
        zx = P[P_ZX] - x[i]
        zy = P[P_ZY] - y[i]
        zd = math.sqrt(zx * zx + zy * zy)
        inp[i, N_RAYS * 4 + 4] = min(1.0, max(-1.0, (P[P_ZR] - zd) / 200.0))
        zr = wrap_angle(math.atan2(zy, zx) - ang[i])
        inp[i, N_RAYS * 4 + 5] = math.cos(zr)
        inp[i, N_RAYS * 4 + 6] = math.sin(zr)


@njit(cache=True, parallel=True)
def think(alive, inp, L, h, out):
    """Модульное мышление (Mixture of Experts + RNN)."""
    n_agents = alive.shape[0]
    
    for i in prange(n_agents):
        if not alive[i]:
            continue
        
        # 1. Диспетчер (Gating Network): вычисляет веса важности для каждого эксперта
        gate_logits = np.empty(N_EXPERTS)
        gate_sum = 0.0
        for e in range(N_EXPERTS):
            s = L[i, OFF_GATE_B + e]
            for a in range(N_IN):
                s += inp[i, a] * L[i, OFF_GATE_W + (a * N_EXPERTS + e)]
            gate_logits[e] = s
            # Стабильный exp для softmax
            gate_sum += math.exp(min(20.0, max(-20.0, s)))
        
        # Softmax для диспетчера (суммарный вес экспертов = 1.0)
        gate_weights = np.empty(N_EXPERTS)
        for e in range(N_EXPERTS):
            val = math.exp(min(20.0, max(-20.0, gate_logits[e])))
            gate_weights[e] = val / (gate_sum + 1e-9)

        # 2. Просчет экспертных модулей и сборка итоговых выходов
        # Обнуляем буферы ответов агента
        for b in range(N_OUT):
            out[i, b] = 0.0
        for j in range(N_H):
            h[i, j] = 0.0

        # Проходим по каждому эксперту
        for e in range(N_EXPERTS):
            exp_base = OFF_W1 + e * SINGLE_EXPERT_SIZE
            w1_offset = exp_base
            b1_offset = exp_base + (OFF_B1 - OFF_W1)
            w2_offset = exp_base + (OFF_W2 - OFF_W1)
            b2_offset = exp_base + (OFF_B2 - OFF_W1)
            
            gw = gate_weights[e] # Степень доверия диспетчера этому эксперту
            
            # Скрытый слой конкретного эксперта (размером N_HE)
            h_start = e * N_HE
            for j in range(N_HE):
                s = L[i, b1_offset + j]
                for a in range(N_IN):
                    s += inp[i, a] * L[i, w1_offset + a * N_HE + j]
                
                h_val = math.tanh(s)
                h[i, h_start + j] = h_val # Записываем в общую память h для RNN
                
                # Вклад эксперта в выходные действия (умноженный на вес диспетчера)
                for b in range(N_OUT):
                    out_s = L[i, b2_offset + b]
                    # Добавляем вклад через веса второго слоя эксперта
                    # (для упрощения суммируем выходы с учетом веса диспетчера)
                    pass # Выходы эксперта посчитаем ниже корректно через скрытые состояния

        # Пересчитываем итоговые `out` как комбинацию выходов экспертов, взвешенных диспетчером
        for b in range(N_OUT):
            combined_out = 0.0
            for e in range(N_EXPERTS):
                exp_base = OFF_W1 + e * SINGLE_EXPERT_SIZE
                w2_offset = exp_base + (OFF_W2 - OFF_W1)
                b2_offset = exp_base + (OFF_B2 - OFF_W1)
                
                # Считаем сырой выход этого эксперта
                exp_out_b = L[i, b2_offset + b]
                h_start = e * N_HE
                for j in range(N_HE):
                    exp_out_b += h[i, h_start + j] * L[i, w2_offset + j * N_OUT + b]
                
                combined_out += gate_weights[e] * math.tanh(exp_out_b)
            
            out[i, b] = combined_out

@njit(cache=True, parallel=True)
def move(alive, x, y, ang, spd, T, energy, age, out, bite_on, mem, gain, xp,
         lvl, P):
    W = P[P_W]
    H = P[P_H]
    for i in prange(alive.shape[0]):
        if not alive[i]:
            continue
        lvl[i] = level_of(xp[i], P[P_XP_BASE], P[P_XP_GROWTH], P[P_LVL_MAX])
        ang[i] = wrap_angle(ang[i] + out[i, 0] * P[P_TURN])
        thrust = (out[i, 1] + 1.0) * 0.5
        spd_mult = 1.0 + lvl[i] * P[P_LVL_SPD]
        s = thrust * P[P_MAXSPD] * spd_mult / T[i, 0] ** 0.3
        spd[i] = s
        x[i] += math.cos(ang[i]) * s
        y[i] += math.sin(ang[i]) * s
        if x[i] < 0.0:
            x[i] = -x[i]
            ang[i] = wrap_angle(math.pi - ang[i])
        elif x[i] > W:
            x[i] = 2.0 * W - x[i]
            ang[i] = wrap_angle(math.pi - ang[i])
        if y[i] < 0.0:
            y[i] = -y[i]
            ang[i] = wrap_angle(-ang[i])
        elif y[i] > H:
            y[i] = 2.0 * H - y[i]
            ang[i] = wrap_angle(-ang[i])
        x[i] = min(W, max(0.0, x[i]))
        y[i] = min(H, max(0.0, y[i]))
        bite_on[i] = out[i, 2] > P[P_BITE_THR]
        mem[i, 0] = out[i, 3]
        mem[i, 1] = out[i, 4]
        energy[i] -= (P[P_C_BASE] * T[i, 0] + P[P_C_MOVE] * s * s * T[i, 0]
                      + P[P_C_BRAIN])
        zx = x[i] - P[P_ZX]
        zy = y[i] - P[P_ZY]
        if zx * zx + zy * zy > P[P_ZR] * P[P_ZR]:
            energy[i] -= P[P_STORM]
            gain[i] -= P[P_STORM]
        age[i] += 1.0


@njit(cache=True)
def interact(order, x, y, ang, T, energy, alive, bite_on, gain, S, ev, fx, fy,
             fa, cstart, citems, fstart, fitems, gx, gy, cs, xp, lvl, P):
    n = alive.shape[0]
    hits = np.zeros(n, np.int32)
    zx = P[P_ZX]
    zy = P[P_ZY]
    zr = P[P_ZR]
    for f in range(fa.shape[0]):
        if not fa[f]:
            continue
        dx = fx[f] - zx
        dy = fy[f] - zy
        if dx * dx + dy * dy > zr * zr:
            fa[f] = False

    for o in range(order.shape[0]):
        i = order[o]
        if not alive[i] or energy[i] <= 0.0:
            continue
        cap = P[P_ECAP] * T[i, 0] * (1.0 + lvl[i] * P[P_LVL_ECAP])
        cx = min(gx - 1, max(0, int(x[i] / cs)))
        cy = min(gy - 1, max(0, int(y[i] / cs)))
        y0 = max(0, cy - 1)
        y1 = min(gy - 1, cy + 1)
        x0 = max(0, cx - 1)
        x1 = min(gx - 1, cx + 1)
        er = P[P_EAT_R0] + P[P_EAT_R1] * T[i, 0]
        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                c = yy * gx + xx
                for t in range(fstart[c], fstart[c + 1]):
                    f = fitems[t]
                    if not fa[f]:
                        continue
                    dx = fx[f] - x[i]
                    dy = fy[f] - y[i]
                    if dx * dx + dy * dy < er * er:
                        fa[f] = False
                        e = P[P_FOOD_E]
                        energy[i] = min(cap, energy[i] + e)
                        gain[i] += e + P[P_RB_FOOD]
                        S[i, 0] += e
                        S[i, 5] += 1.0
                        xp[i] += P[P_XP_FOOD]
                        if int(S[i, 5]) % int(P[P_GROW_EVERY]) == 0:
                            gr = min(P[P_GROW], SIZE_MAX - T[i, 0])
                            if gr > 0.0:
                                T[i, 0] += gr
                                S[i, 6] += gr
        if not bite_on[i]:
            continue
        energy[i] -= P[P_BITE_COST]
        best = -1
        bd = 1e18
        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                c = yy * gx + xx
                for t in range(cstart[c], cstart[c + 1]):
                    j = citems[t]
                    if j == i or not alive[j] or energy[j] <= 0.0:
                        continue
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    d = math.sqrt(dx * dx + dy * dy)
                    reach = P[P_BITE_R0] + P[P_BITE_R1] * (T[i, 0] + T[j, 0])
                    if d >= reach or d >= bd:
                        continue
                    rel = wrap_angle(math.atan2(dy, dx) - ang[i])
                    if abs(rel) > P[P_CONE]:
                        continue
                    best = j
                    bd = d
        if best < 0:
            continue
        j = best
        dmg_mult = 1.0 + lvl[i] * P[P_LVL_DMG]
        mob_mult = min(P[P_MOB_MAX], 1.0 + hits[j] * P[P_MOB_BONUS])
        steal = min(P[P_BITE_DMG] * T[i, 0] * dmg_mult * mob_mult, energy[j])
        hits[j] += 1
        energy[j] -= steal
        gain[j] -= steal
        g = steal * P[P_BITE_EFF]
        S[i, 3] += 1.0
        ev[0] += 1.0
        xp[i] += P[P_XP_BITE]
        if energy[j] <= 1e-9:
            energy[j] = 0.0
            alive[j] = False
            g += P[P_KILL_BONUS] * T[j, 0]
            S[i, 1] += g
            S[i, 4] += 1.0
            ev[1] += 1.0
            xp[i] += P[P_XP_KILL]
        else:
            S[i, 2] += g
            ev[2] += g
        energy[i] = min(cap, energy[i] + g)
        gain[i] += g + P[P_RB_BITE]
        if not alive[j]:
            gain[i] += P[P_RB_KILL]


@njit(cache=True)
def reap(alive, energy, age, P):
    d = 0
    for i in range(alive.shape[0]):
        if alive[i] and (energy[i] <= 0.0 or age[i] > P[P_MAXAGE]):
            alive[i] = False
            d += 1
    return d


@njit(cache=True)
def reproduce(alive, energy, age, T, G, L, x, y, ang, spd, mem, gen, S, inp,
              h, out, xp, lvl, boost_mask, boost_rate_mult, boost_sigma_mult,
              P):
    n = alive.shape[0]
    births = 0
    free = 0
    for i in range(n):
        if not alive[i]:
            continue
        cap = P[P_ECAP] * T[i, 0]
        if energy[i] < P[P_REPRO] * cap or age[i] < P[P_MIN_AGE]:
            continue
        while free < n and alive[free]:
            free += 1
        if free >= n:
            break
        c = free
        e = energy[i] * 0.5
        energy[i] = e
        energy[c] = e
        alive[c] = True
        age[c] = 0.0
        spd[c] = 0.0
        gen[c] = gen[i] + 1
        for p in range(N_PARAMS):
            G[c, p] = G[i, p]
        mutate_weights(G[c], P[P_MUT_RATE], P[P_MUT_SIGMA], P[P_MUT_BIG_P],
                       P[P_MUT_BIG_MULT], P[P_WCLIP], boost_mask,
                       boost_rate_mult, boost_sigma_mult)
        for p in range(N_PARAMS):
            L[c, p] = G[c, p]
        for k in range(N_TRAITS):
            T[c, k] = T[i, k]
        T[c, 0] = max(SIZE_MIN, T[i, 0] - S[i, 6])
        mutate_traits(T[c], P[P_TRAIT_MUT])
        x[c] = min(P[P_W], max(0.0, x[i] + (np.random.random() - 0.5) * 10.0))
        y[c] = min(P[P_H], max(0.0, y[i] + (np.random.random() - 0.5) * 10.0))
        ang[c] = (np.random.random() * 2.0 - 1.0) * math.pi
        for k in range(2):
            mem[c, k] = 0.0
        for k in range(S.shape[1]):
            S[c, k] = 0.0
        for k in range(inp.shape[1]):
            inp[c, k] = 0.0
        for k in range(h.shape[1]):
            h[c, k] = 0.0
        for k in range(out.shape[1]):
            out[c, k] = 0.0
        xp[c] = 0.0
        lvl[c] = 0
        births += 1
    return births


@njit(cache=True)
def spawn_food(fx, fy, fa, count, P):
    zx, zy = P[P_ZX], P[P_ZY]
    zr = max(0.0, P[P_ZR] - 40.0)
    zr2 = zr * zr
    placed = 0
    for f in range(fx.shape[0]):
        if placed >= count:
            break
        if not fa[f]:
            for _ in range(30):
                x = np.random.random() * P[P_W]
                y = P[P_H] * np.random.random() ** P[P_LIGHT]
                if (x - zx) ** 2 + (y - zy) ** 2 <= zr2:
                    fa[f] = True
                    fx[f] = x
                    fy[f] = y
                    placed += 1
                    break
    return placed