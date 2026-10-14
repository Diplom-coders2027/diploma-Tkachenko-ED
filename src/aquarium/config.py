"""Все параметры симуляции в одном месте."""
from dataclasses import dataclass


@dataclass
class Config:
    # окно
    window_w: int = 1920
    window_h: int = 1080
    world_w: int = 3840            # карта в 4 раза больше окна по площади
    world_h: int = 2160
    panel_w: int = 400
    max_creatures: int = 260
    init_creatures: int = 100
    rescue_below: int = 25
    rescue_count: int = 25
    # еда
    max_food: int = 4000
    init_food: int = 1200
    food_per_tick: float = 3.0
    food_energy: float = 20.0
    light_pow: float = 1.5
    battle_light_pow: float = 1.0   # в бою еда по всей высоте, не только у поверхности
    # восприятие
    vision: float = 100.0
    cell: float = 100.0
    # физика и энергия
    energy_cap: float = 100.0
    turn_rate: float = 0.35
    max_speed: float = 3.5
    cost_base: float = 0.03
    cost_move: float = 0.012
    cost_brain: float = 0.01
    eat_r0: float = 4.0
    eat_r1: float = 4.0
    # укус
    bite_r0: float = 4.0
    bite_r1: float = 5.0
    bite_dmg: float = 14.0
    bite_eff: float = 0.2
    kill_bonus: float = 5.0
    bite_cost: float = 0.8
    bite_threshold: float = 0.3
    bite_cone_deg: float = 50.0
    # размножение и возраст
    repro_frac: float = 0.75
    min_repro_age: float = 150.0
    max_age: float = 1e18           # без ограничения по возрасту: живут сколько могут
    # мутации весов
    mut_rate: float = 0.15
    mut_sigma: float = 0.4
    mut_big_p: float = 0.03
    mut_big_mult: float = 4.0
    # мутации черт (размер, eta, цвет)
    trait_mut: float = 1.0          # общий множитель силы мутации черт
    trait_size_sd: float = 0.06     # шум логарифма размера
    trait_eta_sd: float = 0.004     # шум скорости обучения
    trait_col_sd: float = 0.04      # шум компонент цвета
    eta_max: float = 0.1            # верхняя граница скорости обучения
    # обучение в течение жизни (Hebbian + награда)
    reward_scale: float = 0.2
    weight_clip: float = 8.0
    plasticity_decay: float = 0.0005
    init_w_std: float = 1.0
    init_eta_max: float = 0.07
    # доп. поощрение обучающего сигнала (Hebbian-награды), поверх
    # фактического прироста энергии: не влияет на физику/энергию,
    # только усиливает закрепление этих действий в сети
    learn_bonus_food: float = 1.5    # за каждый съеденный кусок корма
    learn_bonus_bite: float = 2.0    # за удачный укус (без убийства)
    learn_bonus_kill: float = 5.0    # за убийство другой рыбы
    # режим «королевская битва»: стартуют round_size, побеждает последний
    battle: bool = False
    round_size: int = 200          # стартуют сразу 200 клеток
    battle_init_food: int = 2000
    battle_food_per_tick: float = 0.8
    zone_delay: int = 1800          # 30 с до начала сужения (зона невидима)
    zone_shrink: int = 7000         # за столько тиков радиус сужается до нуля
    zone_shrink: int = 7000         # за столько тиков радиус сужается до нуля
    storm_damage: float = 0.8       # потеря энергии за тик вне зоны
    autosave_seconds: float = 300.0
    stats_interval: int = 100
    max_round_ticks: int = 10 ** 15  # без лимита: матч идёт, пока не останется одна рыба
    hof_size: int = 10              # сколько последних выживших становятся родителями
    # бонус за еду: каждые N съеденных кусков размер растёт (не наследуется)
    grow_every: int = 100000000000000
    grow_amount: float = 0.0
    # опыт и уровни: прокачка в течение жизни, НЕ передаётся по наследству
    xp_per_food: float = 1.0
    xp_per_bite: float = 3.0        # за удачный укус (высасывание без убийства)
    xp_per_kill: float = 12.0
    xp_level_base: float = 40.0     # опыта нужно на 1-й уровень
    xp_level_growth: float = 1.35   # во сколько раз растёт порог на след. уровень
    lvl_max: int = 12
    lvl_dmg_per_lvl: float = 0.06   # +6% урона укуса за уровень
    lvl_speed_per_lvl: float = 0.02 # +2% к макс. скорости за уровень
    lvl_ecap_per_lvl: float = 0.03  # +3% к запасу энергии за уровень
    # "стая": если цель уже укушена в этот же тик другими, следующие
    # атакующие получают бонус к эффективности укуса
    mob_bonus_per_attacker: float = 0.35
    mob_bonus_max: float = 2.2
    # "тёплый старт" после миграции сейва в сеть с бОльшим N_H
    migrate_boost_ticks: int = 300000
    # социальность: территориальность vs стайность
    trait_soc_sd: float = 0.05       # шум мутации sociality
    social_radius: float = 60.0      # радиус подсчёта соседей своего вида
    social_bonus_mult: float = 0.35  # макс. снижение cost_move/cost_base при стайности и скоплении
    territorial_bite_mult: float = 0.5  # бонус bite_dmg против сородичей при territorial
    crowd_penalty_mult: float = 0.25 # энергоштраф territorial-особи от скученности чужого вида
    role_min_age: float = 300.0
    role_bite_frac: float = 0.25
    # видовая кластеризация (0 = выключена, родители — топ hof_size)
    spec_dist: float = 0.0
    spec_pool: int = 30
    spec_quota: int = 3
    # множители мутаций во время «тёплого старта» после миграции
    migrate_boost_rate_mult: float = 1.0
    migrate_boost_sigma_mult: float = 1.0
    # эволюция внутри матча: переход на след. стадию по порогам
    evo_food: int = 30              # кусков еды с последней эволюции
    evo_kills: int = 5              # убийств с последней эволюции

    @property
    def view_w(self):
        """Ширина видимой части карты (левее панели)."""
        return self.window_w - self.panel_w

    @property
    def view_h(self):
        return self.window_h
