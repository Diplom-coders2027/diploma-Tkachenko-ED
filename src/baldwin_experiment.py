"""
Эксперимент Baldwin effect (Hinton & Nowlan, 1987).

Гипотеза: популяция, которой разрешено обучаться в течение жизни
(Hebbian-пластичность через eta > 0), генетически адаптируется быстрее
по поколениям, чем популяция без обучения (eta = 0, чистый генетический
поиск) — даже если сами веса обучения не наследуются напрямую.

Запуск из корня репозитория (там, где лежит пакет aquarium):
    python -m baldwin_experiment
(либо поправь путь импорта под структуру твоего проекта)

Результат: два CSV в saves/ с метриками победителей по раундам —
baldwin_on.csv и baldwin_off.csv. Дальше их сравниваешь по оси "раунд"
(= поколение) против колонок ticks / food / size — ожидаем, что в
baldwin_on.csv рост происходит быстрее (за меньшее число раундов).
"""
import csv
import os
import time

from aquarium.config import Config
from aquarium.world import World, WIN_COLS

N_ROUNDS = 200          # сколько раундов битвы прогонять в каждом варианте
SEED = 42
OUT_DIR = "saves/baldwin"


def make_config(learning_enabled: bool) -> Config:
    c = Config()
    c.battle = True
    if not learning_enabled:
        # полностью выключаем обучение в течение жизни:
        # eta у всех будет жёстко 0, Hebbian-правило в kernels.learn()
        # обнулится само (e = eta[i] * r = 0), геном при этом
        # продолжает эволюционировать как обычно.
        c.eta_max = 0.0
        c.init_eta_max = 0.0
        c.trait_eta_sd = 0.0
    return c


def run_experiment(label: str, learning_enabled: bool):
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = make_config(learning_enabled)
    w = World(cfg, seed=SEED)
    path = os.path.join(OUT_DIR, f"{label}.csv")

    t0 = time.time()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(WIN_COLS)
        last_round = w.round_no
        while w.round_no <= N_ROUNDS:
            w.step()
            # как только стартовал новый раунд — предыдущий завершился,
            # его данные уже лежат в w.winners[-1]
            if w.round_no != last_round and w.winners:
                writer.writerow(w.winners[-1])
                f.flush()
                last_round = w.round_no
                if (last_round % 10) == 0:
                    dt = time.time() - t0
                    print(f"[{label}] round {last_round}/{N_ROUNDS} "
                          f"({dt:.1f}s)")
    save_path = os.path.join(OUT_DIR, f"{label}.npz")
    w.save(save_path)
    print(f"[{label}] done -> {path}")
    print(f"[{label}] save -> {save_path}")


def main():
    print("=== Baldwin effect experiment ===")
    run_experiment("baldwin_on", learning_enabled=True)
    run_experiment("baldwin_off", learning_enabled=False)
    print("Готово. Сравни saves/baldwin/baldwin_on.csv и "
          "saves/baldwin/baldwin_off.csv по колонкам 'round' vs "
          "'food'/'size'/'ticks'.")


if __name__ == "__main__":
    main()