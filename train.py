"""
Training-Script fuer den Minecraft-Bot.
Box(1) Action Space mit PPO.
"""
import os
import sys
import glob
import time

from config import (
    SERVER_HOST, SERVER_PORT, RCON_PORT, RCON_PASSWORD,
    SERVER_DIR, JAVA_CMD,
    CHECKPOINT_DIR, FINAL_MODEL_PATH, TB_LOG_DIR,
    TOTAL_TIMESTEPS, CHECKPOINT_FREQ, BASIC_ACTIONS,
)


def ensure_dirs():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(TB_LOG_DIR, exist_ok=True)


def load_or_create_model(env):
    """Modell laden oder neu erstellen."""
    from stable_baselines3 import PPO

    # Checkpoints durchsuchen
    checkpoints = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "ppo_mc_*.zip")))
    if checkpoints:
        latest = checkpoints[-1]
        print(f"[TRAIN] Lade Checkpoint: {latest}")
        model = PPO.load(latest, env=env)
        return model

    # Finales Modell
    if os.path.exists(FINAL_MODEL_PATH):
        print(f"[TRAIN] Lade finales Modell: {FINAL_MODEL_PATH}")
        model = PPO.load(FINAL_MODEL_PATH, env=env)
        return model

    # Neues Modell
    print("[TRAIN] Starte neues Training.")
    print(f"[TRAIN] Action Space: Box(1) [-1.0, 1.0] → [0.0, {env.action_high:.3f}]")
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


def train():
    ensure_dirs()

    # Server starten
    from server import ServerManager
    server = ServerManager(
        server_dir=SERVER_DIR,
        java_cmd=JAVA_CMD,
        rcon_host=SERVER_HOST,
        rcon_port=RCON_PORT,
        rcon_password=RCON_PASSWORD,
    )
    server.start()

    # Bot verbinden
    import bot
    bot.create_bot(SERVER_HOST, SERVER_PORT)
    if not bot.wait_for_spawn(timeout=60):
        print("[TRAIN] Bot konnte nicht spawnen. Breche ab.")
        server.stop()
        return

    # mcData laden
    from javascript import require
    mcData = require("minecraft-data")(bot.bot.version)
    print(f"[TRAIN] mcData geladen fuer Version {bot.bot.version}")

    DEBUG_ACTIONS = True

    # Environment erstellen
    from env import MinecraftBotEnv
    env = MinecraftBotEnv(bot, server, mcData, debug_actions=DEBUG_ACTIONS)

    # Modell laden/erstellen
    model = load_or_create_model(env)

    # Callbacks
    from stable_baselines3.common.callbacks import CheckpointCallback
    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_mc",
        verbose=1,
    )

    # Training
    try:
        print("[TRAIN] Starte Training...")
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_cb,
            reset_num_timesteps=False,
        )
        # Nur bei Erfolg speichern
        print("[TRAIN] Speichere Modell...")
        model.save(FINAL_MODEL_PATH)
    except KeyboardInterrupt:
        print("\n[TRAIN] Training unterbrochen (KeyboardInterrupt).")
        print("[TRAIN] Speichere Modell...")
        model.save(FINAL_MODEL_PATH)
    except Exception as e:
        print(f"\n[TRAIN] Fehler: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        server.stop()
        print("[TRAIN] Fertig.")


if __name__ == "__main__":
    train()
