import time
from javascript import require
from mcrcon import MCRcon
import numpy as np

mineflayer = require('mineflayer')
Vec3 = require('vec3')

# --- CONFIG ---
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 25565
RCON_PORT = 25575
RCON_PASSWORD = "banana1"

bot_ready = False
mcData = None

# --- PERSISTENT RCON CLIENT ---

class PersistentRCON:
    def __init__(self, host, password, port):
        self.host = host
        self.password = password
        self.port = port
        self.mcr = None

    def connect(self):
        try:
            self.mcr = MCRcon(self.host, self.password, port=self.port)
            self.mcr.connect()
            print("-> RCON Verbindung dauerhaft aufgebaut.")
        except Exception as e:
            print(f"[RCON Fehler beim Connect]: {e}")

    def send(self, command):
        if self.mcr:
            try:
                return self.mcr.command(command)
            except Exception as e:
                print(f"[RCON Sende-Fehler]: {e}")
        return None

    def close(self):
        if self.mcr:
            try:
                self.mcr.disconnect()
                print("-> RCON Verbindung sauber geschlossen.")
            except Exception as e:
                print(f"[RCON Fehler beim Schließen]: {e}")

rcon = PersistentRCON(SERVER_HOST, RCON_PASSWORD, RCON_PORT)

# --- BOT CREATION ---

print("Erstelle Mineflayer-Bot...")
bot = mineflayer.createBot({
    'host': SERVER_HOST,
    'port': SERVER_PORT,
    'username': 'bananagod',
    'auth': 'offline',
    'version': False
})

# --- EVENT HANDLER ---

def handle_spawn(*args):
    global bot_ready, mcData
    print("-> Bot ist gespannt!")
    
    # mcData ERST HIER laden, sobald bot.version bekannt ist!
    if mcData is None:
        mcData = require('minecraft-data')(bot.version)
        print(f"-> minecraft-data geladen für Version: {bot.version}")
        
    rcon.send("tick freeze")
    bot_ready = True

def handle_death(*args):
    global bot_ready
    bot_ready = False
    print("-> BOT IST GESTORBEN!")

bot.on('spawn', handle_spawn)
bot.on('death', handle_death)

# --- STATE EXTRACTION LOGIC ---

def get_block_id(block):
    if block is None:
        return -1.0  # -1 signalisiert: Chunk noch nicht vom Server geladen!
    
    """Liest die offizielle Block-ID über minecraft-data aus."""
    if not block or block.name == 'air' or block.name == 'cave_air':
        return 0.0
    
    # Verwendet die offizielle numeric state ID aus minecraft-data
    if mcData and block.name in mcData.blocksByName:
        return float(mcData.blocksByName[block.name].id)
    
    return 1.0  # Fallback für unbekannte Blöcke

def get_surrounding_blocks(radius=1, debug=False):
    """
    Liest ein (2*radius + 1)^3 Raster um den Bot aus (Standard: 3x3x3 = 27 Blöcke).
    """
    grid_size = (2 * radius + 1) ** 3
    if not bot or not bot.entity or mcData is None:
        return np.zeros(grid_size, dtype=np.float32)

    pos = bot.entity.position
    base_x = int(np.floor(pos.x))
    base_y = int(np.floor(pos.y))
    base_z = int(np.floor(pos.z))

    block_list = []
    debug_names = []

    for dy in range(-radius, radius + 1):
        for dz in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                target_pos = Vec3(base_x + dx, base_y + dy, base_z + dz)
                block = bot.blockAt(target_pos)
                
                b_id = get_block_id(block)
                block_list.append(b_id)
                
                if debug:
                    name = block.name if block else "unknown"
                    debug_names.append(f"({dx},{dy},{dz}): {name} [ID: {b_id}]")

    if debug:
        print("--- Umgebende Blöcke ---")
        for entry in debug_names:
            print(entry)

    return np.array(block_list, dtype=np.float32)

def get_inventory_state():
    """
    Liest den aktuellen Inventar- & Hotbar-Status aus.
    Gibt ein Vektor-Array für das ML-Modell zurück.
    """
    if not bot or not bot.inventory:
        return np.zeros(12, dtype=np.float32)

    # 1. Welches Hotbar-Item hält der Bot gerade in der Hand?
    held_item = bot.heldItem
    held_item_id = float(held_item.type) if held_item else 0.0
    held_item_count = float(held_item.count) if held_item else 0.0

    # 2. Aktuell ausgewählter Hotbar-Index (0 bis 8)
    selected_slot = float(bot.quickBarSlot) if hasattr(bot, 'quickBarSlot') else 0.0

    # 3. Hotbar-Belegung (Slots 36 bis 44) - wie viele Items liegen in den 9 Slots?
    hotbar_counts = []
    for slot_idx in range(36, 45):
        item = bot.inventory.slots[slot_idx]
        hotbar_counts.append(float(item.count) if item else 0.0)

    # Vektor: [Held_ID, Held_Count, Selected_Slot_Idx, Hotbar_Slot_0_Count ... Hotbar_Slot_8_Count]
    # Gesamt = 1 + 1 + 1 + 9 = 12 Features
    inv_vector = [held_item_id, held_item_count, selected_slot] + hotbar_counts
    return np.array(inv_vector, dtype=np.float32)

def get_state():
    """Gibt das vollständige Observation-Array (44 Features) zurück."""
    if not bot or not bot.entity:
        return np.zeros(5 + 27 + 12, dtype=np.float32)
        
    pos = bot.entity.position
    # 1. Player Features (5 Werte)
    player_features = [pos.x, pos.y, pos.z, bot.entity.yaw, bot.entity.pitch]
    
    # 2. Block Features (27 Werte)
    block_features = get_surrounding_blocks(radius=1)
    
    # 3. Inventory Features (12 Werte)
    inv_features = get_inventory_state()
    
    # Combined Observation: 5 + 27 + 12 = 44 Features
    return np.concatenate([player_features, block_features, inv_features]).astype(np.float32)

# --- TRAININGS-SCHLEIFE ---

def run_training_loop():
    global bot_ready
    rcon.connect()
    
    print("Warte auf Bot-Spawn...")
    while not bot_ready:
        time.sleep(0.2)
        
    print("Bot ist bereit!")
    
    
    print(get_surrounding_blocks(radius=1, debug=True))
    
    rcon.close()
    print("Test-Schleife beendet.")

if __name__ == "__main__":
    run_training_loop()