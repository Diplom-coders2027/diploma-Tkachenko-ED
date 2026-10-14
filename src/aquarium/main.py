"""Точка входа: окно pygame или безоконный режим."""
import argparse
import os
import time

from .config import Config
from .stats import Stats
from .world import WIN_COLS, World

SPEEDS = [1, 2, 5, 10, 25, 50, 100, 300]


def save_winners(w, path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(WIN_COLS) + "\n")
        for r in w.winners:
            f.write(",".join(str(v) for v in r) + "\n")


def run_headless(cfg, seed, ticks, csv_path=None, load=None, rounds=0):
    w = World(cfg, seed)
    if load:
        w.load(load)
    st = Stats(cfg)
    while w.tick < ticks and (not rounds or len(w.winners) < rounds):
        w.step()
        if w.tick % cfg.stats_interval == 0:
            st.record(w)
    if csv_path:
        st.to_csv(csv_path)
    return w, st


FEED_PERIOD = 0.25      # секунд между порциями корма при зажатой ЛКМ
FEED_AMOUNT = 2         # кусочков в одной порции
CAM_SPEED = 900.0       # px/с, камера по WASD
WIN_SECONDS = 3.0       # сколько показывать WIN перед новым матчем


def run_gui(cfg, seed, load=None, fullscreen=False, save_dir="saves",
            play=False):
    import pygame
    from .render import Renderer
    pygame.init()
    flags = pygame.FULLSCREEN if fullscreen else 0
    screen = pygame.display.set_mode((cfg.window_w, cfg.window_h), flags)
    pygame.display.set_caption("Aquarium evolution")
    w = World(cfg, seed)
    w.auto_restart = False          # матч заканчивается WIN-паузой, см. ниже
    if load:
        w.load(load)
    if play:
        w.player_idx = 0
    st = Stats(cfg)
    rend = Renderer(cfg, screen)
    clock = pygame.time.Clock()
    si, paused, render_on, sel = 0, False, True, None
    last_save = time.time()
    last_feed = 0.0
    win_hold = None                 # момент, когда закончится пауза WIN
    running = True
    while running:
        dt = clock.get_time() / 1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_SPACE and not play:
                    paused = not paused
                elif e.key == pygame.K_p and play:
                    paused = not paused
                elif e.key == pygame.K_UP and not play:
                    si = min(len(SPEEDS) - 1, si + 1)
                elif e.key == pygame.K_DOWN and not play:
                    si = max(0, si - 1)
                elif e.key == pygame.K_r:
                    render_on = not render_on
                elif e.key == pygame.K_e:
                    rend.fx = not rend.fx
                elif e.key == pygame.K_TAB:
                    rend.move_focus(-1 if e.mod & pygame.KMOD_SHIFT else 1)
                elif e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    act = rend.activate_focus()
                    if act == "pause" and not play:
                        paused = not paused
                    elif act == "slower" and not play:
                        si = max(0, si - 1)
                    elif act == "faster" and not play:
                        si = min(len(SPEEDS) - 1, si + 1)
                elif e.key == pygame.K_z:
                    rend.fit = not rend.fit      # Z: вся карта / крупно
                elif e.key == pygame.K_v:
                    rend.show_rays = not rend.show_rays
                elif e.key == pygame.K_s and e.mod & pygame.KMOD_CTRL:
                    w.save(os.path.join(save_dir, "manual.npz"))   # Ctrl+S
            elif (e.type == pygame.MOUSEBUTTONDOWN and e.pos[0] < cfg.view_w
                  and e.button not in (1, 2)):
                # ПКМ (и колесо) в аквариуме — выбрать рыбку
                sel = rend.pick(w, e.pos)
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 2:
                rend.cam_drag = e.pos       # средняя кнопка — двигать камеру
            elif e.type == pygame.MOUSEBUTTONUP and e.button == 2:
                rend.cam_drag = None
            elif e.type == pygame.MOUSEMOTION and rend.cam_drag is not None:
                rend.pan(-e.rel[0], -e.rel[1])
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                act = rend.click(e.pos)
                if act == "pause" and not play:
                    paused = not paused
                elif act == "slower" and not play:
                    si = max(0, si - 1)
                elif act == "faster" and not play:
                    si = min(len(SPEEDS) - 1, si + 1)
        # камера: WASD двигает по карте (в режиме игрока камера следует за ним)
        if not play:
            keys = pygame.key.get_pressed()
            px = (keys[pygame.K_d] - keys[pygame.K_a]) * CAM_SPEED * dt
            py = (keys[pygame.K_s] - keys[pygame.K_w]) * CAM_SPEED * dt
            if px or py:
                rend.pan(px, py)
        elif w.player_idx is not None and w.alive[w.player_idx]:
            rend.center_on(w.x[w.player_idx], w.y[w.player_idx])
        # ЛКМ в аквариуме: посыпать немного корма (зажатая кнопка — порциями)
        if not play and pygame.mouse.get_pressed()[0]:
            mx, my = pygame.mouse.get_pos()
            now = time.time()
            if mx < cfg.view_w and now - last_feed >= FEED_PERIOD:
                wx, wy = rend.to_world((mx, my))
                w.sprinkle(wx, wy, k=FEED_AMOUNT)
                last_feed = now
        if play and w.player_idx is not None:
            keys = pygame.key.get_pressed()
            turn = 0.0
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                turn -= 1.0
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                turn += 1.0
            thrust = 1.0 if (keys[pygame.K_UP] or keys[pygame.K_w]) else -1.0
            bite = keys[pygame.K_SPACE]
            w.player_action = (turn, thrust, bite)
            if not w.alive[w.player_idx] and w.in_round[w.player_idx]:
                pass  # ждём следующий раунд, player_idx=0 переиспользуется
        if w.match_over:
            # победитель 3 секунды виден с короной и надписью WIN, потом старт с клеток
            now = time.time()
            if win_hold is None:
                win_hold = now + WIN_SECONDS
            elif now >= win_hold:
                w.restart_match()
                win_hold = None
        elif not paused:
            t0 = time.time()
            done = 0
            budget = SPEEDS[si] if not play else 1
            while done < budget and time.time() - t0 < 0.05:
                w.step()
                done += 1
                if w.tick % cfg.stats_interval == 0:
                    st.record(w)
                if w.match_over:
                    break
        if render_on:
            rend.draw(w, st, sel if not play else w.player_idx,
                      SPEEDS[si] if not play else 1, paused, clock.get_fps())
            if play:
                rend._text("WASD/arrows move  SPACE bite  P pause",
                           20, cfg.window_h - 26, (200, 200, 150))
            else:
                rend._text("ЛКМ корм · ПКМ выбрать · Z вся карта/крупно · WASD/СКМ камера",
                           20, cfg.window_h - 26, (160, 160, 160))
        else:
            screen.fill((10, 10, 14))
            rend._text(f"render off | tick {w.tick} alive {w.n_alive} "
                       "(press R)", 20, 20)
        pygame.display.flip()
        clock.tick(60)
        if time.time() - last_save > cfg.autosave_seconds:
            w.save(os.path.join(save_dir, "autosave.npz"))
            last_save = time.time()
    w.save(os.path.join(save_dir, "autosave.npz"))
    st.to_csv(os.path.join(save_dir, "history.csv"))
    save_winners(w, os.path.join(save_dir, "winners.csv"))
    pygame.quit()


def main(argv=None):
    p = argparse.ArgumentParser(description="Aquarium evolution simulator")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--headless", type=int, default=0,
                   help="N тиков без окна, затем печать итогов")
    p.add_argument("--rounds", type=int, default=0,
                   help="безоконно: сыграть N раундов битвы")
    p.add_argument("--sandbox", action="store_true",
                   help="обычный режим без битвы: размножение и популяция")
    p.add_argument("--play", action="store_true",
                   help="управлять рыбкой #0 в королевской битве против ИИ")
    p.add_argument("--csv", default=None)
    p.add_argument("--load", default=None)
    p.add_argument("--fullscreen", action="store_true")
    a = p.parse_args(argv)
    cfg = Config(battle=not a.sandbox)
    if a.headless or a.rounds:
        w, st = run_headless(cfg, a.seed, a.headless or 10 ** 9, a.csv,
                             a.load, a.rounds)
        print("tick", w.tick, "alive", w.n_alive, "rows", len(st.rows))
        for r in w.winners:
            print("round %d: %.0fs, kills %d, winner food %d, avg food %.1f, "
                  "eta %.3f, size %.2f, gen %d"
                  % (r[0], r[1] / 60.0, r[2], r[6], r[7], r[3], r[4], r[5]))
        if st.rows and a.sandbox:
            print(dict(zip(["tick", "pop", "food", "grazer", "predator",
                            "parasite", "young"], st.rows[-1][:7])))
    else:
        run_gui(cfg, a.seed, a.load, a.fullscreen, play=a.play)
