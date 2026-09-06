"""
Server-Verwaltung: Starten, Stoppen, Welt-Reset.
Free-Running: Kein Freeze, kein Tick-Step — das Game laeuft normal.
"""
import os
import shutil
import subprocess
import time
from mcrcon import MCRcon


class ServerManager:
    def __init__(self, server_dir, java_cmd, rcon_host, rcon_port, rcon_password):
        self.server_dir = server_dir
        self.java_cmd = java_cmd
        self.rcon_host = rcon_host
        self.rcon_port = rcon_port
        self.rcon_password = rcon_password
        self.process = None
        self.mcr = None

    def start(self):
        """Server starten."""
        print("[SERVER] Starte Server...")
        self.process = subprocess.Popen(
            self.java_cmd,
            cwd=self.server_dir,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._wait_for_server(timeout=120)
        self._connect_rcon()
        print("[SERVER] Server laeuft.")

    def stop(self):
        """Server sauber beenden."""
        print("[SERVER] Stoppe Server...")
        self._send_rcon("stop")
        if self.process:
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self._disconnect_rcon()
        self.process = None
        print("[SERVER] Server gestoppt.")

    def reset_world(self):
        """Server stoppen, Welt loeschen, neu starten."""
        self.stop()

        world_dirs = ["world", "world_nether", "world_the_end"]
        for world_name in world_dirs:
            world_path = os.path.join(self.server_dir, world_name)
            if os.path.exists(world_path):
                print(f"[SERVER] Loesche {world_name}...")
                shutil.rmtree(world_path)

        logs_dir = os.path.join(self.server_dir, "logs")
        if os.path.exists(logs_dir):
            shutil.rmtree(logs_dir)

        self.start()

    def is_running(self):
        if self.process is None:
            return False
        return self.process.poll() is None

    def send_command(self, command):
        """RCON-Kommando senden."""
        return self._send_rcon(command)

    def tp_spawn(self):
        self._send_rcon("tp @p ~ ~ ~")

    def gamemode_survival(self):
        self._send_rcon("gamemode survival @p")

    def clear_effects(self):
        self._send_rcon("effect clear @p")

    # --- Intern ---

    def _wait_for_server(self, timeout=120):
        print("[SERVER] Warte auf Server-Start...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with MCRcon(self.rcon_host, self.rcon_password, port=self.rcon_port) as mcr:
                    mcr.command("list")
                    return True
            except Exception:
                time.sleep(2)
        print("[SERVER] TIMEOUT beim Server-Start!")
        return False

    def _connect_rcon(self):
        try:
            self.mcr = MCRcon(self.rcon_host, self.rcon_password, port=self.rcon_port)
            self.mcr.connect()
        except Exception as e:
            print(f"[SERVER] RCON-Fehler: {e}")

    def _disconnect_rcon(self):
        if self.mcr:
            try:
                self.mcr.disconnect()
            except Exception:
                pass
            self.mcr = None

    def _send_rcon(self, command):
        if self.mcr:
            try:
                resp = self.mcr.command(command)
                resp = (resp or "").strip()
                if resp:
                    print(f"  [RCON] {command} -> {resp}")
                return resp
            except Exception as e:
                print(f"[SERVER] RCON-Fehler bei '{command}': {e}")
        return None
