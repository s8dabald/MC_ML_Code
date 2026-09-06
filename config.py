# --- SERVER ---
import sys

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 25565
RCON_PORT = 25575
RCON_PASSWORD = "banana1"
# Lokal (Windows) bleibt der Live-Template-Ordner; auf Linux/Colab liegen Code und
# Server-Assets unter /content. Alle Instanz-Dirs (Server_0/1/2) haengen daran.
SERVER_DIR = (
    r"C:\Users\Balda\Desktop\MC_ML\Server"
    if sys.platform == "win32"
    else "/content/mcml/Server"
)
# JVM-Heap pro Server-Instanz: Start 1G, Deckel 2G (RAM-Sparen).
JAVA_CMD = "java -Xms1G -Xmx2G -jar paper.jar --nogui"

# --- ML ---
MAX_ENTITIES = 1
BLOCK_RADIUS = 1
BLOCK_GRID_SIZE = (2 * BLOCK_RADIUS + 1) ** 3  # 27
STATE_SIZE = 5 + 4 + 3 + 5 + 3 + 3 + 45 + BLOCK_GRID_SIZE + (MAX_ENTITIES * 4)  # 99
MAX_ROTATION = 1.0  # Maximale Drehung in Radians pro Step (~57°)

# --- ACTION NAMES ---
BASIC_ACTIONS = 29
ACTION_NAMES = [
    "idle",                                          # 0
    "forward", "back", "left", "right",              # 1-4
    "jump", "sneak", "sprint",                       # 5-7
    "yaw+", "yaw-", "pitch+", "pitch-",              # 8-11
    "attack", "use", "dig", "place",                  # 12-15
    "hotbar_0", "hotbar_1", "hotbar_2",             # 16-18
    "hotbar_3", "hotbar_4", "hotbar_5",             # 19-21
    "hotbar_6", "hotbar_7", "hotbar_8",             # 22-24
    "swap_hand",                                     # 25
    "drop_item",                                     # 26
    "eat",                                           # 27
    "craft",                                         # 28
]

# --- INSTANCE CONFIG ---
# Fuer VecEnv: Instance-Offset auf Ports und Server-Dir.
# Jede Instanz bekommt einen eigenen Dir Server_<id>; Server/ bleibt reine Vorlage.
# Instanz 0: Port 25565, RCON 25575, Dir = Server_0
# Instanz 1: Port 25566, RCON 25576, Dir = Server_1
# Instanz 2: Port 25567, RCON 25577, Dir = Server_2
INSTANCE_BASE_PORT = 25565
INSTANCE_BASE_RCON = 25575

# --- TRAINING ---
# Kein Checkpoint-System: EIN Modell-File wird laufend ueberschrieben (~1x/Min) und
# - falls gemountet - nach Google Drive kopiert (Colab-Persistenz, kein nohup).
FINAL_MODEL_PATH = "./ppo_minecraft_final.zip"
TB_LOG_DIR = "./tb_logs/"
TOTAL_TIMESTEPS = 1_000_000
MAX_STEPS_PER_EPISODE = 1_000_000
# Colab: Pfad auf dem gemounteten Drive. Existiert der Zielordner nicht (Windows),
# wird die Drive-Kopie einfach uebersprungen.
DRIVE_MODEL_PATH = "/content/drive/MyDrive/MC_ML/ppo_minecraft_final.zip"

# --- REWARDS ---
SURVIVE_REWARD = 0.005     # pro Tick (~+0.1/Sekunde bei 20 TPS)
DEATH_PENALTY = -100.0     # Tod dominiert gegenueber Ueberlebens-Einkommen
DAMAGE_PENALTY = -0.2      # pro Schadenspunkt (halbes Heart)

# --- EXPLORATIONS-/BEWEGUNGS-REWARD (Anker-basiert, pro Episode) ---
# Belohnt horizontale Netto-Abwanderung (x,z) vom Anker. Nur echte Bewegung weg
# vom Anker gibt Bonus: im Kreis laufen und Buddeln nach unten (x,z-Delta~0) zaehlt
# nicht. Der Anker wird alle ANCHOR_REFRESH_DIST weitergezogen, damit der Bot
# kontinuierlich strebt. Deckel pro Episode (= Welt bis Tod).
MOVEMENT_REWARD_ENABLED = True
MOVEMENT_BONUS_PER_BLOCK = 0.6    # +0.6 pro Block Netto-Entfernung vom Anker
MOVEMENT_MAX_ACCUM_DIST = 500.0    # Deckel pro Episode (Blöcke)
MOVEMENT_ANCHOR_REFRESH_DIST = 25.0  # Anker alle 25 Blöcke weiterziehen

# --- INVENTAR-SAMMEL-REWARD (dauerhaft) ---
# +REWARD pro neu gefuelltem Item-Slot (Hotbar+Main-Inventar aus der Obs), Deckel
# auf INVENTORY_REWARD_MAX_SLOTS gefuellte Slots. High-Water-Logik: nur NEUE
# Rekord-Befuellung der Episode wird belohnt, damit Drop+Wiederaufnehmen keinen
# Reward gibt (wieder voll = nicht > Rekord). Fördert Gegenst�nde einsammeln.
INVENTORY_REWARD_ENABLED = True
INVENTORY_REWARD_PER_SLOT = 0.5       # +0.5 pro neu gefuelltem Slot
INVENTORY_REWARD_MAX_SLOTS = 18       # Deckel an gefuellten Slots pro Episode

# --- GRAB-PUSH/GUIDE (nur zu Beginn, anneliert) ---
# Bestraft Netto-Abstieg nach unten (senkrechtes Buddeln weg vom y-Anker), solange
# der Bot nicht horizontal weiterlaeuft. Klingt ueber DIG_DOWN_GUIDE_STEPS auf 0 ab
# (fruehe Fuehrung gegen sinnloses Runtergraben, kein dauerhafter Eingriff).
DIG_DOWN_GUIDE_ENABLED = True
DIG_DOWN_PENALTY_PER_BLOCK = -0.5     # pro Block Netto-Abstieg
DIG_DOWN_GUIDE_STEPS = 10_000         # Anneal: klingt hier auf 0 ab


def get_instance_config(instance_id):
    """Gibt (server_dir, game_port, rcon_port) fuer eine Instanz zurueck.
    Jede Instanz bekommt einen eigenen Server-Dir (Server_<id>), damit der
    Vorlage-Ordner SERVER_DIR (Server/) als unangetastete Template-Quelle fuer
    paper.jar/eula.txt erhalten bleibt. Keine Kopie der Welt wird uebernommen.
    """
    suffix = f"_{instance_id}"
    return {
        "server_dir": SERVER_DIR + suffix,
        "game_port": INSTANCE_BASE_PORT + instance_id,
        "rcon_port": INSTANCE_BASE_RCON + instance_id,
    }
