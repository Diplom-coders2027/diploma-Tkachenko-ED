"""Геном: раскладка весов сети, инициализация и мутации."""
import numpy as np
from numba import njit

N_RAYS = 5
N_EXTERNAL_IN = N_RAYS * 4 + 7   # лучи*4 + энергия + скорость + 2 памяти + 3 зона
N_H = 24                # Суммарно скрытых нейронов
N_IN = N_EXTERNAL_IN + N_H       # Внешние входы + рекуррентная память RNN
N_OUT = 5               # поворот, тяга, укус, 2 ячейки памяти

# --- МОДУЛЬНАЯ АРХИТЕКТУРА (Mixture of Experts) ---
N_EXPERTS = 3           # Количество экспертных модулей (0: Еда, 1: Атака, 2: Бегство)
N_HE = N_H // N_EXPERTS # Нейронов в каждом эксперте (24 / 3 = 8)

# Раскладка весов в плоском векторе генома:
# 1. Веса Диспетчера (определяет важность каждого эксперта от входов)
OFF_GATE_W = 0
OFF_GATE_B = OFF_GATE_W + N_IN * N_EXPERTS
OFF_GATE_OUT = OFF_GATE_B + N_EXPERTS

# 2. Веса Экспертов (у каждого свои W1, B1, W2, B2)
OFF_W1 = OFF_GATE_OUT
OFF_B1 = OFF_W1 + N_IN * N_HE
OFF_W2 = OFF_B1 + N_HE
OFF_B2 = OFF_W2 + N_HE * N_OUT
SINGLE_EXPERT_SIZE = OFF_B2 + N_OUT - OFF_W1

# Общий размер параметров генома = Диспетчер + (Эксперты * 3)
N_PARAMS = OFF_GATE_OUT + SINGLE_EXPERT_SIZE * N_EXPERTS

N_TRAITS = 6            # размер, eta, r, g, b, sociality
SIZE_MIN, SIZE_MAX = 0.7, 2.5


def random_genomes(rng, k, std):
    g = rng.normal(0.0, std, (k, N_PARAMS))
    # Обнуляем смещения (biases) для стабильного старта
    g[:, OFF_GATE_B:OFF_GATE_OUT] = 0.0
    for e in range(N_EXPERTS):
        base = OFF_W1 + e * SINGLE_EXPERT_SIZE
        g[:, base + (OFF_B1 - OFF_W1):base + (OFF_W2 - OFF_W1)] = 0.0
        g[:, base + (OFF_B2 - OFF_W1):base + SINGLE_EXPERT_SIZE] = 0.0
    return g


def canonical(G):
    """Копия генома, где нейроны каждого эксперта отсортированы по убыванию
    нормы входящих весов W1. Нужна ТОЛЬКО для сравнения геномов: перестановка
    скрытых нейронов меняет рекуррентные входы, поэтому сам геном не переставляем.
    G: (k, N_PARAMS)."""
    out = G.copy()
    k = G.shape[0]
    for e in range(N_EXPERTS):
        base = OFF_W1 + e * SINGLE_EXPERT_SIZE
        w1 = G[:, base:base + N_IN * N_HE].reshape(k, N_IN, N_HE)
        b1 = G[:, base + N_IN * N_HE:base + N_IN * N_HE + N_HE]
        o_w2 = base + N_IN * N_HE + N_HE
        w2 = G[:, o_w2:o_w2 + N_HE * N_OUT].reshape(k, N_HE, N_OUT)
        norm = np.sqrt((w1 * w1).sum(axis=1))          # (k, N_HE)
        order = np.argsort(-norm, axis=1)
        w1p = np.take_along_axis(w1, order[:, None, :], axis=2)
        b1p = np.take_along_axis(b1, order, axis=1)
        w2p = np.take_along_axis(w2, order[:, :, None], axis=1)
        out[:, base:base + N_IN * N_HE] = w1p.reshape(k, -1)
        out[:, base + N_IN * N_HE:base + N_IN * N_HE + N_HE] = b1p
        out[:, o_w2:o_w2 + N_HE * N_OUT] = w2p.reshape(k, -1)
    return out


def pairwise_dist(A, B):
    """Евклидовы расстояния между строками A (n, d) и B (m, d) одним вызовом."""
    a2 = (A * A).sum(axis=1)[:, None]
    b2 = (B * B).sum(axis=1)[None, :]
    d2 = a2 + b2 - 2.0 * (A @ B.T)
    return np.sqrt(np.maximum(d2, 0.0))


def expand_genome(old, old_n_h, new_n_h, n_in, n_out, rng, std=0.25):
    """Переносит геном(ы) из сети со старым числом скрытых нейронов в сеть
    с новым (большим) числом — "тёплый старт"."""
    assert new_n_h >= old_n_h
    k = old.shape[0]
    off_b1_o = n_in * old_n_h
    off_w2_o = off_b1_o + old_n_h
    off_b2_o = off_w2_o + old_n_h * n_out

    w1_o = old[:, 0:off_b1_o].reshape(k, n_in, old_n_h)
    b1_o = old[:, off_b1_o:off_w2_o]
    w2_o = old[:, off_w2_o:off_b2_o].reshape(k, old_n_h, n_out)
    b2_o = old[:, off_b2_o:off_b2_o + n_out]

    w1_n = rng.normal(0.0, std, (k, n_in, new_n_h))
    b1_n = np.zeros((k, new_n_h))
    w2_n = rng.normal(0.0, std, (k, new_n_h, n_out))
    w1_n[:, :, :old_n_h] = w1_o
    b1_n[:, :old_n_h] = b1_o
    w2_n[:, :old_n_h, :] = w2_o
    b2_n = b2_o

    off_b1_n = n_in * new_n_h
    off_w2_n = off_b1_n + new_n_h
    off_b2_n = off_w2_n + new_n_h * n_out
    n_params_new = off_b2_n + n_out
    out = np.zeros((k, n_params_new))
    out[:, 0:off_b1_n] = w1_n.reshape(k, -1)
    out[:, off_b1_n:off_w2_n] = b1_n
    out[:, off_w2_n:off_b2_n] = w2_n.reshape(k, -1)
    out[:, off_b2_n:off_b2_n + n_out] = b2_n
    return out


def random_traits(rng, k, eta_max):
    t = np.empty((k, N_TRAITS))
    t[:, 0] = np.clip(np.exp(rng.normal(0.0, 0.1, k)), SIZE_MIN, SIZE_MAX)
    t[:, 1] = rng.uniform(0.0, eta_max, k)
    t[:, 2:5] = rng.uniform(0.0, 1.0, (k, 3))
    t[:, 5] = rng.uniform(0.0, 1.0, k)   # sociality: 0=территориальный, 1=стайный
    return t


@njit(cache=True)
def seed_numba(s):
    np.random.seed(s)


@njit(cache=True)
def mutate_weights(w, rate, sigma, big_p, big_mult, wclip, boost_mask,
                   boost_rate_mult, boost_sigma_mult):
    for k in range(w.shape[0]):
        r = rate
        s = sigma
        if boost_mask[k]:
            r = min(1.0, rate * boost_rate_mult)
            s = sigma * boost_sigma_mult
        if np.random.random() < r:
            if np.random.random() < big_p:
                s = s * big_mult
            v = w[k] + np.random.normal(0.0, s)
            w[k] = min(wclip, max(-wclip, v))


@njit(cache=True)
def mutate_traits(t, scale, eta_max=0.1, sd_size=0.06, sd_eta=0.004,
                  sd_col=0.04, sd_soc=0.05):
    # значения по умолчанию совпадают с config.Config
    v = t[0] * np.exp(np.random.normal(0.0, sd_size * scale))
    t[0] = min(SIZE_MAX, max(SIZE_MIN, v))
    v = t[1] + np.random.normal(0.0, sd_eta * scale)
    t[1] = min(eta_max, max(0.0, v))
    for k in range(2, 5):
        v = t[k] + np.random.normal(0.0, sd_col * scale)
        t[k] = min(1.0, max(0.0, v))
    v = t[5] + np.random.normal(0.0, sd_soc * scale)
    t[5] = min(1.0, max(0.0, v))