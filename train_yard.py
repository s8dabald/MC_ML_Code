"""
VecEnv Training fuer Minecraft-Bot.
Free-Running: K Server-Instanzen, ein geteiltes Modell (SB3 PPO + SubprocVecEnv).
Jede Instanz: eigener Server, eigener Bot, eigene Env.
"""
import os
import shutil
import time
import numpy as np

# Wichtiger Top-Level-Import: SB3 registriert BaseCallback automatisch im
# globalen Namespace des definierenden Moduls, WENN es bereits importiert ist.
# Ohne diesen Import wuerde `class ProgressCallback(BaseCallback)` mit NameError
# crashen (CallbackMeta-Injection setzt den Namen nur bei geladenem BaseCallback).
from stable_baselines3.common.callbacks import BaseCallback

from config import (
    SERVER_HOST, JAVA_CMD, RCON_PASSWORD,
    FINAL_MODEL_PATH, TB_LOG_DIR, DRIVE_MODEL_PATH,
    TOTAL_TIMESTEPS, STATE_SIZE,
    get_instance_config,
)


def make_env(instance_id, debug_actions=False):
    """Factory-Funktion fuer SubprocVecEnv."""
    def _init():
        import gymnasium as gym
        from gymnasium import spaces

        cfg = get_instance_config(instance_id)
        print(f"[ENV {instance_id}] Erstelle Env auf Port {cfg['game_port']}...")

        # Server starten
        from server_manager import InstanceManager
        server = InstanceManager(
            instance_id=instance_id,
            server_dir=cfg["server_dir"],
            game_port=cfg["game_port"],
            rcon_port=cfg["rcon_port"],
            rcon_password=RCON_PASSWORD,
            java_cmd=JAVA_CMD,
        )
        server.start()

        # Robustes Aufraeumen: Wenn dieser Worker endet (egal wodurch - normal,
        # Fehler, oder weil der Parent die Pipe schliesst bei Ctrl+C), wird der
        # Server ueber RCON gestoppt. Unabhaengig von der SubprocVecEnv-Pipe,
        # damit keine Zombie-Server/Ports zurueckbleiben.
        import atexit as _atexit

        def _worker_cleanup():
            try:
                server.stop()
            except Exception:
                pass
        _atexit.register(_worker_cleanup)

        # Bot verbinden (mit Retry: auf Colab-Frisch-VMs kann der Erst-Boot inkl.
        # Welt-Generierung länger dauern als ein einzelnes 60s-Fenster).
        import bot
        spawn_ok = False
        for attempt in range(3):
            if attempt == 0:
                bot.create_bot(SERVER_HOST, cfg["game_port"])
                spawn_ok = bot.wait_for_spawn(timeout=120)
            else:
                print(f"[ENV {instance_id}] Bot-Spawn Versuch {attempt + 1} fehlgeschlagen, reconnect...")
                spawn_ok = bot.reconnect(SERVER_HOST, cfg["game_port"], timeout=120)
            if spawn_ok:
                break
        if not spawn_ok:
            print(f"[ENV {instance_id}] Bot konnte nicht spawnen!")
            raise RuntimeError(f"Bot spawn fehlgeschlagen auf Instanz {instance_id}")

        # mcData laden
        from javascript import require
        mcData = require("minecraft-data")(bot.bot.version)
        print(f"[ENV {instance_id}] mcData geladen fuer Version {bot.bot.version}")

        # Env erstellen
        from env import MinecraftBotEnv
        env = MinecraftBotEnv(bot, server, mcData, debug_actions=debug_actions,
                              game_port=cfg["game_port"])

        return env

    return _init


class ProgressCallback(BaseCallback):
    """Druckt regelmaessig (~1x/Minute) einen ausfuehrlichen Fortschritts-Log des
    Trainings auf die Konsole. Rein lesend (kein Einfluss aufs Training).
    Liest die Metriken aus dem SB3-Logger (self.model.logger.name_to_value), der
    die train/*-Makros nach dem letzten Update haelt — robuster als self.locals,
    die zum _on_rollout_start-Zeitpunkt noch keinen train/*-Wert enthalten (sonst
    Standard-fill = nan). Beim allerersten Rollout (vor dem ersten Update) ist der
    Logger noch leer -> Metriken werden als '--' angezeigt.
    """

    def __init__(self, log_interval_sec=60.0, verbose=0):
        super().__init__(verbose)
        self.log_interval_sec = log_interval_sec
        self._last_log_time = 0.0

    def _on_step(self):
        # Pflicht (abstractmethod bei BaseCallback); Training niemals stoppen.
        return True

    def _on_training_start(self):
        self._last_log_time = time.time()
        return True

    def _on_rollout_start(self):
        # Ein Fortschritts-Log nur, wenn das Intervall abgelaufen ist (~1x/Minute
        # oder haeufiger nur bei langen Rollouts). Am Ende des Logs die Zeit pinnen,
        # damit es nicht pro Rollout spamt.
        now = time.time()
        if now - self._last_log_time >= self.log_interval_sec:
            self._print_progress()

        # WICHTIG: BaseCallback-Semantik — False stoppt das Training. Dieser Callback
        # darf das Training NIE beenden, also immer True.
        return True

    def _print_progress(self):
        model = self.model
        ts = int(getattr(model, "num_timesteps", 0))

        # Episoden-Return/Laenge/Distanz aus dem SB3-Puffer (Rolling).
        ep_rew = None
        ep_len = None
        explored = None
        buf = getattr(self, "ep_info_buffer", None)
        if buf is None:
            buf = getattr(model, "ep_info_buffer", None)
        if buf:
            try:
                ep_rew = float(sum(e.get("r", 0.0) for e in buf)) / len(buf)
                ep_len = float(sum(e.get("l", 0.0) for e in buf)) / len(buf)
                exp_vals = [e.get("explored", None) for e in buf]
                exp_vals = [v for v in exp_vals if v is not None]
                if exp_vals:
                    explored = float(sum(exp_vals)) / len(exp_vals)
            except Exception:
                ep_rew = ep_len = explored = None

        # train/*-Makros aus dem SB3-Logger lesen (zuverlaessig nach dem letzten
        # Update). Vor dem ersten Update ist der Logger leer -> nan.
        nv = {}
        try:
            logger = getattr(model, "logger", None)
            if logger is not None:
                nv = dict(logger.name_to_value or {})
        except Exception:
            nv = {}

        def fmt(key, spec="{:.3f}"):
            val = nv.get(key)
            if val is None:
                return "--"
            try:
                f = float(val)
            except Exception:
                return "--"
            if f != f:  # nan -> "--"
                return "--"
            return spec.format(f)

        ts_hr = f"{ts/1000:.0f}k" if ts >= 1000 else str(ts)
        line = (
            f"[PPO] timesteps={ts_hr}"
            + (f" | ep_len={ep_len:.0f} ep_rew={ep_rew:+.1f}" if ep_len is not None and ep_rew is not None else "")
            + (f" explored={explored:.1f}m" if explored is not None else "")
            + " | vf=" + fmt("train/value_loss", "{:.1f}")
            + " pol=" + fmt("train/policy_gradient_loss", "{:+.3f}")
            + " ent=" + fmt("train/entropy_loss", "{:.3f}")
            + " ev=" + fmt("train/explained_variance", "{:.3f}")
            + " clip=" + fmt("train/clip_fraction")
            + " kl=" + fmt("train/approx_kl")
            + " std=" + fmt("train/std")
        )
        print("[TRAIN] " + line, flush=True)

        # Einzelnes Modell-File laufend aktualisieren (+ Drive-Kopie, wenn gemountet).
        # Beides try/except-geschuetzt: ein Save-/Drive-Fehler darf Training nie killen.
        persist_model(model)

        self._last_log_time = time.time()


def _drive_push():
    """Kopiert das lokale Modell-File nach Google Drive, falls gemountet."""
    try:
        drive_dir = os.path.dirname(os.path.normpath(DRIVE_MODEL_PATH))
        if os.path.isdir(drive_dir):
            shutil.copy2(FINAL_MODEL_PATH, DRIVE_MODEL_PATH)
    except Exception as e:
        print(f"[TRAIN] Drive-Kopie fehlgeschlagen: {e}")


def persist_model(model):
    """Speichert das Modell auf den einen FINAL_MODEL_PATH und schiebt ggf. nach Drive."""
    try:
        model.save(FINAL_MODEL_PATH)
        _drive_push()
    except Exception as e:
        print(f"[TRAIN] Modell-Save fehlgeschlagen: {e}")


def train(num_instances=1, debug_actions=False):
    """Hauptfunktion: VecEnv Training."""
    ensure_dirs()

    print(f"\n{'='*60}")
    print(f"  TRAINING: {num_instances} Instanz(en), Free-Running 20 TPS")
    print(f"  Erwartete Steps/s: ~{num_instances * 20}")
    print(f"{'='*60}\n")

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv

    # VecEnv erstellen
    envs = SubprocVecEnv([make_env(i, debug_actions) for i in range(num_instances)])

    # atexit-Sicherheitsnetz: stoppt alle Server, falls das Skript unerwartet endet
    import atexit

    def _cleanup():
        try:
            envs.close()
        except Exception:
            pass
    atexit.register(_cleanup)

    # Modell laden oder neu erstellen
    model = load_or_create_model(envs)

    # Callbacks
    # Fortschritts-Log (~1x/Minute, ausfuehrlich, rein lesend). Speichert dabei
    # das Einzel-Modell-File (+ Drive-Kopie) — kein Checkpoint-System noetig.
    progress_cb = ProgressCallback(log_interval_sec=60.0)

    # Endlos-Weiterlauf: Nach jedem TOTAL_TIMESTEPS-Budget folgt ein Hygiene-Zyklus
    # (kurzer sauberer Stop + Restart des Servers, Welt bleibt erhalten), dann laeuft
    # das Training nahtlos weiter. Beendet wird nur bei KeyboardInterrupt.
    try:
        print("[TRAIN] Starte Training (Endlos-Weiterlauf)...")
        while True:
            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                callback=[progress_cb],
                reset_num_timesteps=False,
            )
            print(f"[TRAIN] Budget von {TOTAL_TIMESTEPS} Steps erreicht -> Speichern + Hygiene-Zyklus.")
            persist_model(model)

            # Hygiene: Server kurz sauber stoppen und neu starten (Welt bleibt).
            print("[TRAIN] Hygiene-Zyklus: stoppe und starte Server neu...")
            envs.env_method("hygiene_restart")
            print("[TRAIN] Hygiene-Zyklus abgeschlossen. Nahtlos weiter...")
    except KeyboardInterrupt:
        print("\n[TRAIN] Training unterbrochen (KeyboardInterrupt).")
        print("[TRAIN] Speichere Modell...")
        persist_model(model)
    except Exception as e:
        print(f"\n[TRAIN] Fehler: {e}")
        import traceback
        traceback.print_exc()
        # Auch bei hartem Fehler: aktuelles Modell sichern, falls es schon existiert.
        try:
            persist_model(model)
        except Exception:
            pass
    finally:
        try:
            envs.close()
        except Exception as e:
            print(f"[TRAIN] Warnung: envs.close() mit Fehler umgegangen ({e}).")
            print("[TRAIN] Worker-atexit stoppt die Server dennoch (RCON).")
        print("[TRAIN] Fertig.")


def load_or_create_model(env):
    """Modell laden (das eine existierende Modell-File) oder neu erstellen."""
    from stable_baselines3 import PPO

    # Das eine Modell-File (wird laufend ueberschrieben; von Drive zurueckgeholt)
    if os.path.exists(FINAL_MODEL_PATH):
        print(f"[TRAIN] Lade Modell: {FINAL_MODEL_PATH}")
        return PPO.load(FINAL_MODEL_PATH, env=env)

    # Neues Modell
    print("[TRAIN] Starte neues Training.")
    print(f"[TRAIN] Observation Space: Box({env.observation_space.shape})")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[256, 256],
        ),
        tensorboard_log=TB_LOG_DIR,
    )
    return model


def ensure_dirs():
    os.makedirs(TB_LOG_DIR, exist_ok=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Minecraft-Bot VecEnv Training")
    parser.add_argument("--instances", "-n", type=int, default=3,
                        help="Anzahl paralleler Server-Instanzen (default: 4)")
    parser.add_argument("--debug", action="store_true",
                        help="Action-/Detal-Logs pro Step aktivieren (default: AUS)")
    parser.add_argument("--no-actions", action="store_true",
                        help="Action-Logging auch mit --debug unterdruecken (Kompatibilitaet)")
    args = parser.parse_args()

    # Default: Action-/Detal-Logs AUS (nur mit --debug und ohne --no-actions an).
    # Server-/Env-/Trainings-Logs sind unabhaengig davon und laufen immer.
    train(num_instances=args.instances,
          debug_actions=args.debug and not args.no_actions)
