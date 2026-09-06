import time
import numpy as np
from javascript import require
from config import (
    SERVER_HOST, SERVER_PORT, MAX_ENTITIES, STATE_SIZE, BLOCK_RADIUS,
)

mineflayer = require("mineflayer")
Vec3 = require("vec3")
HELPERS = require("./env_helpers.js")

# --- GLOBAL STATE ---
bot = None
mcData = None
bot_ready = False
dead = False
disconnected = False
kick_reason = None
disconnect_reason = None


def create_bot(host=SERVER_HOST, port=SERVER_PORT):
    """Bot erstellen und Event-Handler registrieren."""
    global bot, mcData, bot_ready, dead, disconnected, kick_reason, disconnect_reason

    print("Erstelle Mineflayer-Bot...")
    bot = mineflayer.createBot({
        "host": host,
        "port": port,
        "username": "bananagod",
        "auth": "offline",
        "version": False,
        # Client-Keepalive-Timeout erhoehen: Default 30s beendet die Verbindung
        # ("socketClosed"/"keepAliveError"), wenn der Server mal kurz rueckelt.
        # Bei Free-Running 20TPS sollte der Server normal antworten, aber bei
        # einem Spike (z.B. Weltgen, Mobs) darf der Client nicht sofort gezwungen
        # werden. 5 Minuten Puffer.
        "checkTimeoutInterval": 5 * 60 * 1000,
    })

    def handle_spawn(*args):
        global mcData, bot_ready, disconnected, kick_reason, disconnect_reason
        print("-> Bot ist gespannt!")
        # Neuer Spawn = neue Verbindung -> DC-Flags zuruecksetzen
        disconnected = False
        kick_reason = None
        disconnect_reason = None
        if mcData is None:
            mcData = require("minecraft-data")(bot.version)
            print(f"-> minecraft-data geladen fuer Version: {bot.version}")
        bot_ready = True

    def handle_death(*args):
        global bot_ready, dead
        bot_ready = False
        dead = True
        print("-> BOT IST GESTORBEN!")

    def handle_kicked(reason, loggedIn):
        global bot_ready, disconnected, kick_reason, disconnect_reason
        bot_ready = False
        disconnected = True
        kick_reason = repr(reason) if reason else "(kein Grund angegeben)"
        disconnect_reason = "kicked"
        print(f"-> BOT WURDE GEKICKT! loggedIn={loggedIn} reason={kick_reason}")

    def handle_end(reason):
        global bot_ready, disconnected, disconnect_reason
        bot_ready = False
        disconnected = True
        disconnect_reason = repr(reason) if reason else "(kein Grund angegeben)"
        print(f"-> BOT VERBINDUNG BEENDET! reason={disconnect_reason}")

    def handle_error(err):
        print(f"-> BOT-FEHLER: {repr(err) if err else '(kein Grund)'}")

    bot.on("spawn", handle_spawn)
    bot.on("death", handle_death)
    bot.on("kicked", handle_kicked)
    bot.on("end", handle_end)
    bot.on("error", handle_error)
    return bot


def is_connected():
    """Prueft ob der Bot aktuell verbunden und ready ist."""
    global disconnected, bot_ready
    if disconnected:
        return False
    if not bot_ready:
        return False
    try:
        if bot is not None and not getattr(bot, "_client", None):
            return False
    except Exception:
        pass
    return True


def get_disconnect_info():
    """Liefert Tuempel (kick_reason, disconnect_reason) fuer Debug-Ausgabe."""
    global kick_reason, disconnect_reason
    return kick_reason, disconnect_reason


def is_dead():
    """Prueft ob der Bot gestorben ist."""
    global dead
    if dead:
        return True
    try:
        if bot is not None and getattr(bot, "isAlive", True) is False:
            return True
        if bot is not None and bot.game is not None:
            gm = bot.game.gameMode
            if gm == "spectator" or gm == 3:
                return True
    except Exception:
        pass
    return False


def reconnect(host=SERVER_HOST, port=SERVER_PORT, timeout=60):
    """Bot nach Hardcore-Tod neu verbinden."""
    global bot, bot_ready, dead, disconnected, kick_reason, disconnect_reason

    print("[BOT] Reconnecting...")
    bot_ready = False
    dead = False
    disconnected = False
    kick_reason = None
    disconnect_reason = None

    if bot is not None:
        try:
            bot.quit()
        except Exception:
            pass
        bot = None

    time.sleep(2)
    create_bot(host, port)
    return wait_for_spawn(timeout=timeout)


def wait_for_spawn(timeout=120):
    """Warten bis Bot gespawnt ist."""
    global bot_ready
    print("Warte auf Bot-Spawn...")
    start = time.time()
    while not bot_ready and time.time() - start < timeout:
        time.sleep(0.2)
    if not bot_ready:
        print(" TIMEOUT: Bot hat nicht gespawnt!")
    return bot_ready


def is_bot_ready():
    return bot_ready


def get_state():
    """Gibt das vollstaendige Observation-Array zurueck.
    Nutzt snapshot() aus env_helpers.js — ein einziger Bridge-Call (~1ms).
    WICHTIG: Die Bridge liefert ein `Proxy`-Objekt, kein Python-list. Direktes
    `np.array(proxy)` failt mit "invalid __array_struct__" (daher lieferte diese
    Funktion frueher still np.zeros -- silent-zeros-Bug). Erst `list(proxy)`
    materialisieren, dann numpy. Bei einem Fehler wird NULLEN + Kurzmeldung
    zurueckgegeben (nicht stillschweigend nur Nullen verdrängt).
    """
    try:
        if not bot or not bot.entity:
            return np.zeros(STATE_SIZE, dtype=np.float32)
        arr = HELPERS.snapshot(bot)
        return np.array(list(arr), dtype=np.float32)
    except Exception as e:
        print(f"[STATE-ERROR] Konnte Obs nicht lesen: {type(e).__name__}: {e}")
        return np.zeros(STATE_SIZE, dtype=np.float32)


if __name__ == "__main__":
    from config import RCON_PORT, RCON_PASSWORD, SERVER_DIR, JAVA_CMD
    from server import ServerManager

    server = ServerManager(
        server_dir=SERVER_DIR,
        java_cmd=JAVA_CMD,
        rcon_host=SERVER_HOST,
        rcon_port=RCON_PORT,
        rcon_password=RCON_PASSWORD,
    )
    server.start()
    create_bot()
    wait_for_spawn()

    state = get_state()
    print(f"\n=== STATE ({len(state)} Features) ===")
    nonzero = np.count_nonzero(state)
    print(f"  Non-zero features: {nonzero}/{len(state)}")
    print(f"  First 10: {state[:10]}")

    server.stop()
    print("\nTest beendet.")
