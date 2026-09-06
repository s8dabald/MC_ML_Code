"""
Gymnasium Environment fuer den Minecraft-Bot.
Free-Running: Das Game laeuft bei 20 TPS, das Modell mithaltet.
Kein Freeze, kein Tick-Step — jede Aktion wird auf den lebenden Bot angewendet.
"""
import time
import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from javascript import require

from config import (
    STATE_SIZE, BASIC_ACTIONS, MAX_ROTATION, MAX_STEPS_PER_EPISODE,
    ACTION_NAMES, SURVIVE_REWARD, DEATH_PENALTY, DAMAGE_PENALTY,
    MOVEMENT_REWARD_ENABLED, MOVEMENT_BONUS_PER_BLOCK,
    MOVEMENT_MAX_ACCUM_DIST, MOVEMENT_ANCHOR_REFRESH_DIST,
    INVENTORY_REWARD_ENABLED, INVENTORY_REWARD_PER_SLOT, INVENTORY_REWARD_MAX_SLOTS,
    DIG_DOWN_GUIDE_ENABLED, DIG_DOWN_PENALTY_PER_BLOCK, DIG_DOWN_GUIDE_STEPS,
)
from skills import craft_item, find_food_slot

Vec3 = require("vec3")
HELPERS = require("./env_helpers.js")


class MinecraftBotEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, bot_module, server_manager, mc_data, debug_actions=False, game_port=None):
        super().__init__()

        self.bot_mod = bot_module
        self.server = server_manager
        self.bot = bot_module.bot
        self.mcData = mc_data
        self.debug_actions = debug_actions
        self.game_port = game_port

        # Craftbare Items aus mcData.recipes ermitteln
        self.craftable_items = []
        self._build_craftable_list()

        self.num_craft_items = len(self.craftable_items)
        self.action_high = 29.0 + (self.num_craft_items - 1) / 1000.0 if self.num_craft_items > 0 else 29.0

        print(f"[ENV] Action Space: Box(1) [-1.0, 1.0] -> [0.0, {self.action_high:.3f}]")
        print(f"[ENV]   {self.num_craft_items} craftbare Items")

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(STATE_SIZE,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(1,), dtype=np.float32,
        )

        self.current_step = 0
        self.prev_health = 20.0
        self.episode_reward = 0.0
        self.max_steps = MAX_STEPS_PER_EPISODE

        # Lag-Monitoring
        self._last_game_tick = 0
        self._lag_ticks = 0

        # Bewegungs-State-Tracking (fuer clearControlStates-Optimierung)
        self._move_active = False

        # Explorations-Revard: Anker + Akkumulatoren (pro Episode)
        self._anchor_x = 0.0
        self._anchor_z = 0.0
        self._anchor_y = None
        self._last_anchor_dist = 0.0
        self._accum_explored = 0.0
        self._episode_exploration_reward = 0.0

        # Inventar-Sammel-Reward: High-Water pro Episode
        self._slot_high_water = 0

        # Grab-Push/Guide: akkumuliert ueber den GANZEN Lauf (nie resetet) -> Anneal
        self._guide_total_steps = 0

    def _reset_anchors(self):
        """Anker auf aktuelle Bot-Position setzen und Explorations-Budget zuruecksetzen."""
        try:
            if self.bot and self.bot.entity:
                p = self.bot.entity.position
                self._anchor_x = float(p.x)
                self._anchor_z = float(p.z)
                self._anchor_y = float(p.y)
            else:
                self._anchor_x = 0.0
                self._anchor_z = 0.0
                self._anchor_y = None
        except Exception:
            self._anchor_x = 0.0
            self._anchor_z = 0.0
            self._anchor_y = None
        self._last_anchor_dist = 0.0
        self._accum_explored = 0.0
        self._episode_exploration_reward = 0.0
        self._slot_high_water = 0

    def _movement_bonus(self, x, z):
        """Bonus fuer horizontale Netto-Abwanderung (x,z) vom Anker.
        +BONUS pro Block echter Entfernung. Kreis-Lauf (delta~0) und Buddeln nach
        unten (x,z-Delta~0) geben keinen Bonus. Anker wird weitergezogen, wenn der
        Bot ANCHOR_REFRESH_DIST erreicht. Deckel: MOVEMENT_MAX_ACCUM_DIST pro Episode.
        """
        if not MOVEMENT_REWARD_ENABLED or self._accum_explored >= MOVEMENT_MAX_ACCUM_DIST:
            return 0.0
        d = math.hypot(x - self._anchor_x, z - self._anchor_z)
        delta = max(0.0, d - self._last_anchor_dist)
        self._last_anchor_dist = d
        if delta <= 0.0:
            return 0.0
        take = min(delta, MOVEMENT_MAX_ACCUM_DIST - self._accum_explored)
        if take <= 0.0:
            return 0.0
        self._accum_explored += take
        if d >= MOVEMENT_ANCHOR_REFRESH_DIST:
            self._anchor_x, self._anchor_z = x, z
            self._last_anchor_dist = 0.0
        return take * MOVEMENT_BONUS_PER_BLOCK

    def _inventory_reward(self, obs):
        """Belohnt das Einsammeln von Items: +REWARD pro NEU gefuelltem Inventar-Slot.
        Zaehlt gefuellte Slots aus der Obs (Hotbar+Main-Inventar = obs[27..62]),
        Deckel auf INVENTORY_REWARD_MAX_SLOTS. High-Water-Logik (pro Episode):
        nur wenn die gefuellte-Slot-Anzahl einen neuen Rekord erreicht, gibt es Reward.
        Drop+Wiederaufnehmen (Anzahl wieder auf Alt-Niveau) erzeugt so KEINEN Reward.
        """
        if not INVENTORY_REWARD_ENABLED or obs is None:
            return 0.0
        try:
            count = int(np.count_nonzero(obs[27:63]))
        except Exception:
            return 0.0
        count = min(count, INVENTORY_REWARD_MAX_SLOTS)
        gain = count - self._slot_high_water
        if gain <= 0:
            return 0.0
        self._slot_high_water = count
        return gain * INVENTORY_REWARD_PER_SLOT

    def _guide_reward(self, y, moved_horizontal):
        """Grab-Push (initial): bestraft Netto-Abstieg nach unten weg vom y-Anker.
        guide_strength klingt ueber DIG_DOWN_GUIDE_STEPS auf 0 ab. Neutralisierung:
        sobald sich der Bot in diesem Step horizontal bewegt (moved_horizontal), wird
        der y-Anker mitgezogen, damit normales Gelände (Rampe/Treppe + Seitwärts) den
        Bot nicht faelschlich bestraft — nur reines/sinnloses Senkrecht-Buddeln.
        """
        if not DIG_DOWN_GUIDE_ENABLED:
            return 0.0
        strength = max(0.0, 1.0 - self._guide_total_steps / DIG_DOWN_GUIDE_STEPS)
        if strength <= 0.0 or self._anchor_y is None:
            return 0.0
        if moved_horizontal or y >= self._anchor_y - 1.0:
            # seitwaerts ODER auf/bis ~1 Block unter der Ankerhoehe (Oberflaechen-
            # bzw. Ausstiegs-Niveau): neutral, Anker folgt dem Bot (Terrain/Rampe,
            # Klettern zurueck nach oben bestraft nicht).
            self._anchor_y = float(y)
            return 0.0
        if y < self._anchor_y - 1.0:
            penalty = (self._anchor_y - y) * abs(DIG_DOWN_PENALTY_PER_BLOCK) * strength
            return -penalty
        return 0.0

    def _get_game_tick(self):
        """game-tick (lag monitoring). Nur fuer eat/craft-Pfad; im Bündel mitgeliefert."""
        try:
            return self.bot.time.age if self.bot and self.bot.time else 0
        except Exception:
            return 0

    def _build_craftable_list(self):
        if not self.mcData:
            return
        try:
            recipes = self.mcData.recipes
            items_array = self.mcData.itemsArray
            if recipes is None or items_array is None:
                return
            for i in range(items_array.length):
                item_obj = items_array[i]
                item_id = int(item_obj.id)
                item_name = str(item_obj.name)
                try:
                    if recipes[item_id] is not None:
                        self.craftable_items.append((item_id, item_name))
                except Exception:
                    continue
        except Exception as e:
            print(f"[ENV] Fehler beim Laden der Rezepte: {e}")
        self.craftable_items.sort(key=lambda x: x[1])

    def _interpret_action(self, action_val):
        mapped = (float(action_val) + 1.0) / 2.0 * self.action_high
        if mapped < 29.0:
            action_type = int(round(mapped))
            action_type = max(0, min(action_type, BASIC_ACTIONS - 1))
            return action_type, 0, mapped
        else:
            craft_idx = int(round((mapped - 29.0) * 1000))
            craft_idx = max(0, min(craft_idx, self.num_craft_items - 1))
            return 28, craft_idx, mapped

    def _action_name(self, action_type, craft_idx=0):
        if action_type == 28 and self.craftable_items:
            idx = min(max(craft_idx, 0), len(self.craftable_items) - 1)
            return f"craft_{self.craftable_items[idx][1]}"
        if 0 <= action_type < len(ACTION_NAMES):
            return ACTION_NAMES[action_type]
        return f"unknown_{action_type}"

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # DC/disonnected (nicht Tod) -> nur neu verbinden, keine Welt-Reset noetig
        if not self.bot_mod.is_connected():
            kick, disc = self.bot_mod.get_disconnect_info()
            print(f"[ENV] Bot nicht verbunden - Reconnect (kick={kick}, disconnect={disc})...")
            if not self.bot_mod.reconnect(port=self.game_port, timeout=60):
                raise RuntimeError("Bot reconnect fehlgeschlagen nach Disconnect")
            self.bot = self.bot_mod.bot
            print("[ENV] Reconnect erfolgreich.")

        # Hardcore-Tod -> Welt-Reset
        if self.bot_mod.is_dead():
            print("[ENV] Bot ist tot - Hardcore-Reset...")
            self.server.reset_world()
            if not self.bot_mod.reconnect(port=self.game_port, timeout=60):
                raise RuntimeError("Bot reconnect fehlgeschlagen nach Hardcore-Tod")
            self.bot = self.bot_mod.bot
            print("[ENV] Hardcore-Reset erfolgreich.")
        else:
            self.bot_mod.wait_for_spawn(timeout=30)

        # Warten bis Welt geladen ist
        time.sleep(1.5)

        if not self.bot or not self.bot.entity:
            raise RuntimeError("Bot-Entity nicht vorhanden nach Reset!")

        self.current_step = 0
        try:
            self.prev_health = self.bot.health if self.bot else 20.0
        except Exception:
            self.prev_health = 20.0
        self.episode_reward = 0.0

        # Lag-Monitoring initialisieren
        try:
            self._last_game_tick = self.bot.time.age if self.bot.time.age else 0
        except Exception:
            self._last_game_tick = 0
        self._lag_ticks = 0
        self._move_active = False

        # Neue Episode (= neue Welt bis Tod): Explorations-Anker + Budget zuruecksetzen
        self._reset_anchors()

        if self.debug_actions:
            pos = self.bot.entity.position
            print(f"  [RESET] Bot bei ({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f}) HP={self.bot.health}")

        obs = self._get_obs()
        return obs, {}

    def hygiene_restart(self):
        """Hygiene-Zyklus: Server sauber stoppen und neu starten (Welt bleibt),
        Bot neu verbinden. Wird am Ende jedes Trainings-Budgets aufgerufen.
        """
        print("[ENV] Hygiene-Restart...")
        self.server.stop()
        self.server.start()
        if not self.bot_mod.reconnect(port=self.game_port, timeout=60):
            raise RuntimeError("Bot reconnect fehlgeschlagen nach Hygiene-Restart")
        self.bot = self.bot_mod.bot
        time.sleep(3)
        if not self.bot or not self.bot.entity:
            raise RuntimeError("Bot-Entity nicht vorhanden nach Hygiene-Restart")
        self.current_step = 0
        try:
            self.prev_health = self.bot.health if self.bot else 20.0
        except Exception:
            self.prev_health = 20.0
        self.episode_reward = 0.0
        self._move_active = False
        print("[ENV] Hygiene-Restart abgeschlossen.")

    def step(self, action):
        self.current_step += 1
        t0 = time.time()

        # Float interpretieren
        action_type, craft_idx, mapped_value = self._interpret_action(action[0])

        if self.debug_actions:
            name = self._action_name(action_type, craft_idx)
            print(f"  [STEP {self.current_step}] raw={float(action[0]):+.3f} mapped={mapped_value:.3f} -> {action_type} ({name})")

        if action_type in (27, 28):
            # eat/craft: Python-Pfad (benoetigen Python-Helper), selten.
            self._apply_action(action_type, craft_idx, mapped_value)
            obs = self._get_obs()
            try:
                health = self.bot.health if self.bot and self.bot.entity else 0.0
            except Exception:
                health = 0.0
            current_tick = self._get_game_tick()
            self._move_active = False
            try:
                pos_x = float(obs[0])
                pos_z = float(obs[2])
                pos_y = float(obs[1])
            except Exception:
                pos_x = 0.0
                pos_z = 0.0
                pos_y = 0.0
        else:
            # Aktionen 0-26: EIN gebuendelter Bridge-Call (Aktion + Obs + health
            # + lagTick + alive + connected). Konvertierung via list()-Materialisieren.
            rotation = 0.0
            if action_type in (8, 9, 10, 11):
                rotation = (mapped_value - int(mapped_value)) * MAX_ROTATION
            res = HELPERS.step(self.bot, action_type, mapped_value,
                               rotation, self._move_active)
            # EIN IPC (serialize) statt ~103 Einzel-Calls: valueOf() liefert ein
            # echtes Python-list. Verifiziert am Live-Bot (kein None/NaN-Verlust,
            # np.array passt). Kein Slicing/Schleifen am Proxy.
            #
            # Laengen-Guard: Trotz JS-Hard-Safety (env_helpers _buildObs + Bundle
            # garantieren 103) kann valueOf() bei einer extremen Tod-Race theoretisch
            # ein zu kurzes Ergebnis liefern. Kein hartes vals[99] -> kein IndexError,
            # der sonst den ganzen VecEnv (alle Instanzen) crasht. Defensive Werte.
            vals = res.valueOf()
            try:
                n = len(vals)
            except Exception:
                n = 0
            if n >= 99:
                obs_part = list(vals[:99])
            else:
                obs_part = list(vals) + [0.0] * (99 - n)
            obs = np.array(obs_part, dtype=np.float32)
            if n > 99:
                try:
                    health = float(vals[99])
                except Exception:
                    health = 0.0
            else:
                health = 0.0
            if n > 100:
                try:
                    current_tick = int(vals[100])
                except Exception:
                    current_tick = self._last_game_tick
            else:
                current_tick = self._last_game_tick
            self._move_active = 1 <= action_type <= 7
            try:
                pos_x = float(vals[0])
                pos_y = float(vals[1])
                pos_z = float(vals[2])
            except Exception:
                pos_x = 0.0
                pos_y = 0.0
                pos_z = 0.0
            if self.debug_actions and n != 103:
                print(f"  [WARN] step(): valueOf() lieferte Laenge {n} (erwartet 103)")

        # Lag-Monitoring (nutzt den vom Bündel gelieferten Tick statt eigenem Read)
        tick_diff = current_tick - self._last_game_tick
        if tick_diff > 1:
            self._lag_ticks += (tick_diff - 1)
            if self.debug_actions:
                print(f"  [LAG] {tick_diff - 1} ticks skipped (total: {self._lag_ticks})")
        self._last_game_tick = current_tick

        reward = self._compute_reward(health)

        # Explorations-/Bewegungs-Bonus (Anker-basiert, pro Episode)
        mv = self._movement_bonus(pos_x, pos_z)
        reward += mv
        self._episode_exploration_reward += mv

        # Inventar-Sammel-Reward (dauerhaft): +REWARD pro neu gefuelltem Slot
        inv = self._inventory_reward(obs)
        reward += inv

        # Grab-Push/Guide (anneliert): bestraft senkrechtes Buddeln weg vom y-Anker.
        # moved_horizontal=True wenn sich der Bot in diesem Step horizontal bewegt
        # hat -> dann wird der y-Anker mitgezogen (kein Terrain-Fehlpositiv).
        guide = self._guide_reward(pos_y, moved_horizontal=(mv > 0.0))
        reward += guide

        # Done? is_dead()/is_connected() sind reine Python-Globals (kein IPC -> frei).
        # Konsistent mit dem urspruenglichen Pfad, um Verhaltens-Drift zu vermeiden.
        try:
            alive = not self.bot_mod.is_dead() and self.bot_mod.is_connected() and bool(self.bot) and bool(self.bot.entity)
        except Exception:
            alive = False
        terminated = not alive
        truncated = self.current_step >= self.max_steps

        self.prev_health = health

        info = {
            "episode_reward": self.episode_reward,
            "step": self.current_step,
            "health": health,
            "lag_ticks": self._lag_ticks,
            "exploration_reward": self._episode_exploration_reward,
            "explored_dist": self._accum_explored,
            "inventory_reward": inv,
            "slots_filled": self._slot_high_water if hasattr(self, "_slot_high_water") else 0,
            "guide_penalty": guide,
            "guide_strength": max(0.0, 1.0 - self._guide_total_steps / DIG_DOWN_GUIDE_STEPS),
        }

        # SB3-konformes Episode-Dict am Episodenende: befuellt den ep_info_buffer
        # (r, l) und traegt die gelaufene Distanz (explored) fuer den Fortschritts-Log.
        if terminated:
            info["episode"] = {
                "r": self.episode_reward,
                "l": self.current_step,
                "explored": self._accum_explored,
            }

        self._guide_total_steps += 1

        if self.debug_actions:
            t1 = time.time()
            print(f"    ~ total={t1-t0:.3f}s")

        return obs, reward, terminated, truncated, info

    def _apply_action(self, action_type, craft_idx, mapped_value):
        """Eine einzige Aktion ausfuehren. Direkte mineflayer-Calls."""
        b = self.bot
        if not b or not b.entity:
            return

        # Immer alle Controls zuruecksetzen (ausser bei Bewegungs-Aktionen)
        if action_type not in (1, 2, 3, 4, 5, 6, 7):
            try:
                b.clearControlStates()
            except Exception:
                pass

        # idle
        if action_type == 0:
            return

        # Bewegung (1-7)
        elif action_type == 1:
            b.setControlState("forward", True)
        elif action_type == 2:
            b.setControlState("back", True)
        elif action_type == 3:
            b.setControlState("left", True)
        elif action_type == 4:
            b.setControlState("right", True)
        elif action_type == 5:
            b.setControlState("jump", True)
        elif action_type == 6:
            b.setControlState("sneak", True)
        elif action_type == 7:
            b.setControlState("sprint", True)

        # Kamera (8-11) — force=True: Rotation sofort, kein Warten auf move-Event
        elif action_type in (8, 9, 10, 11):
            fraction = mapped_value - int(mapped_value)
            rotation = fraction * MAX_ROTATION
            try:
                yaw = b.entity.yaw
                pitch = b.entity.pitch
                if action_type == 8:
                    b.look(yaw + rotation, pitch, True)
                elif action_type == 9:
                    b.look(yaw - rotation, pitch, True)
                elif action_type == 10:
                    b.look(yaw, pitch + rotation, True)
                elif action_type == 11:
                    b.look(yaw, pitch - rotation, True)
            except Exception as e:
                print(f"[ENV] Kamera-Fehler: {e}")

        # attack (12) — via env_helpers.attackNearest: testet die Entity gegen die
        # Client-Registry, damit der Server nicht wegen "invalid entity" kickt
        # (das war die Root-Cause des DC).
        elif action_type == 12:
            try:
                HELPERS.attackNearest(b)
            except Exception:
                pass

        # use (13) — Rechtsklick ins Fadenkreuz (env_helpers.useBlock):
        # Block wird intern per Raycast geholt; nichts im Fadenkreuz = nichts.
        elif action_type == 13:
            try:
                HELPERS.useBlock(b)
            except Exception:
                pass

        # dig (14) — startet digging, mineflayer loest nativ auf
        elif action_type == 14:
            try:
                block = b.blockAtCursor(5)
                if block and b.canDigBlock(block):
                    b.dig(block, "ignore")
            except Exception:
                pass

        # place (15) — block_place mit forceLook:'ignore' (kein Look-Hang);
        # nichts im Fadenkreuz oder keine Hand-Item = nichts.
        elif action_type == 15:
            try:
                block = b.blockAtCursor(5)
                if block and b.heldItem:
                    b._placeBlockWithOptions(block, Vec3(0, 1, 0), {
                        "forceLook": "ignore",
                        "swingArm": "right",
                    })
            except Exception:
                pass

        # hotbar (16-24)
        elif 16 <= action_type <= 24:
            try:
                b.setQuickBarSlot(action_type - 16)
            except Exception:
                pass

        # swap_hand (25)
        elif action_type == 25:
            try:
                b.simpleClick.leftMouse(45)
            except Exception:
                pass

        # drop_item (26)
        elif action_type == 26:
            try:
                if b.heldItem:
                    b.tossStack(b.heldItem)
            except Exception:
                pass

        # eat (27) — Server verarbeitet ~40 Ticks
        elif action_type == 27:
            slot = find_food_slot(b)
            if slot is not None:
                try:
                    b.clickWindow(slot, 0, 0)
                    b.clickWindow(36, 0, 0)
                    b.setQuickBarSlot(0)
                    b.activateItem(False)
                except Exception:
                    pass

        # craft (28)
        elif action_type == 28:
            if self.craftable_items:
                idx = min(max(craft_idx, 0), len(self.craftable_items) - 1)
                item_id, _ = self.craftable_items[idx]
                craft_item(b, self.mcData, item_id)

    def _get_obs(self):
        """Obs via snapshot (1 RPC, ~1ms)."""
        return self.bot_mod.get_state()

    def _compute_reward(self, health):
        if not self.bot or not self.bot.entity:
            return 0.0
        if self.bot_mod.is_dead() or (health is not None and health <= 0):
            return DEATH_PENALTY
        reward = SURVIVE_REWARD
        if health < self.prev_health:
            reward -= DAMAGE_PENALTY * (self.prev_health - health)
        self.episode_reward += reward
        return reward

    def close(self):
        """Server-Instanz sauber beenden (kein Zombie-Prozess auf dem Port)."""
        try:
            self.server.stop()
        except Exception as e:
            print(f"[ENV] Fehler beim Stoppen des Servers: {e}")
