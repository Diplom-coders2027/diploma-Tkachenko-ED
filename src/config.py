"""Все параметры симуляции в одном месте."""
from dataclasses import dataclass


@dataclass
class Config:
    # окно
    window_w: int = 1920
    window_h: int = 1080
    panel_w: int = 400
    # популяция
    max_creatures: int = 150
    init_creatures: int = 100
    rescue_below: int = 25
    rescue_count: int = 25
    # еда
    max_food: int = 2500
    init_food: int = 1200
    food_per_tick: float = 3.0
    food_energy: float = 20.0
    light_pow: float = 1.5
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
    max_age: float = 9000.0
    # мутации
    mut_rate: float = 0.15
    mut_sigma: float = 0.4
    mut_big_p: float = 0.03
    mut_big_mult: float = 4.0
    trait_mut: float = 1.0
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
    round_size: int = 100
    battle_init_food: int = 500
    battle_food_per_tick: float = 0.8
    # зона (считаем, что при x1 идёт 60 тиков/с)
    zone_delay: int = 1800          # 30 с до начала сужения
    zone_shrink: int = 7000         # за столько тиков радиус сужается до нуля
    storm_damage: float = 0.8       # потеря энергии за тик вне зоны
    max_round_ticks: int = 14000
    hof_size: int = 10              # сколько последних выживших становятся родителями
    # бонус за еду: каждые N съеденных кусков размер растёт (не наследуется)
    grow_every: int = 100000000000000
    grow_amount: float = 0.0
    # опыт и уровни: прокачка в течение жизни, НЕ передаётся по наследству
    # (в отличие от generic/весов сети, которые эволюционируют между раундами)
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
    # атакующие получают бонус к эффективности укуса — так группа слабых
    # может завалить одну крупную/сильную рыбу
    mob_bonus_per_attacker: float = 0.35
    mob_bonus_max: float = 2.2
    # "тёплый старт" после миграции сейва в сеть с бОльшим N_H: веса новых
    # нейронов временно мутируют чаще и сильнее, чтобы быстрее найти
    # полезную роль, не трогая уже обученные старые веса
    migrate_boost_ticks: int = 300000
    migrate_boost_rate_mult: float = 4.0
    migrate_boost_sigma_mult: float = 5.0
    # статистика / сохранения
    stats_interval: int = 100
    role_min_age: float = 300.0
    role_bite_frac: float = 0.25
    autosave_seconds: float = 300.0

    def __post_init__(self):
        if self.cell < self.vision:
            raise ValueError("cell must be >= vision")
        if self.battle:
            if self.round_size > self.max_creatures:
                raise ValueError("round_size > max_creatures")
        elif self.init_creatures > self.max_creatures:
            raise ValueError("init_creatures > max_creatures")

    @property
    def world_w(self) -> int:
        return self.window_w - self.panel_w

    @property
    def world_h(self) -> int:
        return self.window_h
