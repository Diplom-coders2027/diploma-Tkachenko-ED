"""Классификация ролей (по поведению, не по генам) и история популяции."""
import numpy as np

GRAZER, PREDATOR, PARASITE, UNKNOWN = 0, 1, 2, 3
ROLE_NAMES = ["grazer", "predator", "parasite", "young"]
COLUMNS = ["tick", "pop", "food", "grazer", "predator", "parasite", "young",
           "mean_size", "mean_eta", "max_gen", "bites", "kills", "drained"]


def classify(S, alive, age, min_age, bite_frac=0.25):
    """Грейзер (травоядный): <bite_frac энергии от укусов. Иначе хищник, если энергии
    с убийств >= высасывания, иначе паразит."""
    kill, drain = S[:, 1], S[:, 2]
    bite = kill + drain
    total = S[:, 0] + bite
    role = np.full(S.shape[0], UNKNOWN, dtype=np.int8)
    ok = alive & (age >= min_age) & (total > 0)
    graz = ok & (bite < bite_frac * total)
    hunt = ok & ~graz
    role[graz] = GRAZER
    role[hunt & (kill >= drain)] = PREDATOR
    role[hunt & (kill < drain)] = PARASITE
    return role


class Stats:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rows = []

    def record(self, w):
        c = self.cfg
        alive = w.alive
        role = classify(w.S, alive, w.age, c.role_min_age, c.role_bite_frac)
        cnt = [int(((role == r) & alive).sum()) for r in range(4)]
        n = int(alive.sum())
        # молодые без данных тоже считаются как UNKNOWN
        cnt[UNKNOWN] = n - cnt[0] - cnt[1] - cnt[2]
        if n:
            ms = float(w.T[alive, 0].mean())
            me = float(w.T[alive, 1].mean())
            mg = int(w.gen[alive].max())
        else:
            ms = me = 0.0
            mg = 0
        self.rows.append((w.tick, n, int(w.fa.sum()), *cnt, ms, me, mg,
                          float(w.ev[0]), float(w.ev[1]), float(w.ev[2])))
        w.ev[:] = 0.0

    def arrays(self):
        if not self.rows:
            return {k: np.zeros(0) for k in COLUMNS}
        a = np.array(self.rows, dtype=np.float64)
        return {k: a[:, i] for i, k in enumerate(COLUMNS)}

    def to_csv(self, path):
        import os
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(",".join(COLUMNS) + "\n")
            for r in self.rows:
                f.write(",".join(str(v) for v in r) + "\n")
