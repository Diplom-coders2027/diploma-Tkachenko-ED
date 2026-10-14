"""Строит сравнительный график baldwin_on vs baldwin_off по раундам.
Запуск: python plot_baldwin.py
Нужен matplotlib: pip install matplotlib
"""
import csv

import matplotlib.pyplot as plt

PATHS = {
    "С обучением (eta > 0)": "saves/baldwin/baldwin_on.csv",
    "Без обучения (eta = 0)": "saves/baldwin/baldwin_off.csv",
}

# какую метрику смотрим: "food" (еды собрано победителем), "size", "ticks"
METRIC = "food"


def load(path):
    rounds, vals = [], []
    with open(path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rounds.append(float(row["round"]))
            vals.append(float(row[METRIC]))
    return rounds, vals


def moving_avg(vals, window=10):
    out = []
    for i in range(len(vals)):
        lo = max(0, i - window + 1)
        out.append(sum(vals[lo:i + 1]) / (i - lo + 1))
    return out


def main():
    plt.figure(figsize=(9, 5))
    for label, path in PATHS.items():
        rounds, vals = load(path)
        smooth = moving_avg(vals, window=10)
        plt.plot(rounds, smooth, label=label)
    plt.xlabel("Раунд (поколение)")
    plt.ylabel(f"{METRIC} победителя раунда (скользящее среднее, окно=10)")
    plt.title("Baldwin effect: влияние пожизненного обучения на скорость "
              "генетической адаптации")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("saves/baldwin/comparison.png", dpi=150)
    print("Сохранено: saves/baldwin/comparison.png")
    plt.show()


if __name__ == "__main__":
    main()