"""
Multi-Instance Server-Manager.
Startet und verwaltet K Paper-Server unabhängig voneinander.
Jede Instanz: eigener Port, eigener RCON-Port, eigener Server-Dir.
"""
import os
import shutil
import subprocess
import time
from mcrcon import MCRcon

from config import (
    SERVER_HOST, JAVA_CMD, INSTANCE_BASE_PORT, INSTANCE_BASE_RCON,
    SERVER_DIR, RCON_PASSWORD,
)

# Vorlage-Quelle: der unangetastete Live-Server-Ordner (Server/).
# Hier kommen paper.jar + eula.txt her; niemals wird dessen Welt kopiert.
TEMPLATE_DIR = SERVER_DIR


class InstanceManager:
    """Verwaltet EINE Server-Instanz."""

    def __init__(self, instance_id, server_dir, game_port, rcon_port,
                 rcon_password="banana1", java_cmd=JAVA_CMD):
        self.instance_id = instance_id
        self.server_dir = server_dir
        self.game_port = game_port
        self.rcon_port = rcon_port
        self.rcon_password = rcon_password
        self.java_cmd = java_cmd
        self.process = None
        self.mcr = None
        self.log_file = None

    def start(self):
        """Server-Instanz starten."""
        self.prepare()
        self._check_java()
        print(f"[INST {self.instance_id}] Starte Server auf Port {self.game_port}...")
        # Server-Output nicht schlucken: laeuft in <instanz>/server.log, damit
        # Boot-Fehler (z.B. Paperclip-Netzwerk, Java-Probleme) sichtbar werden.
        log_path = os.path.join(self.server_dir, "server.log")
        self.log_file = open(log_path, "a")
        self.process = subprocess.Popen(
            self.java_cmd,
            cwd=self.server_dir,
            shell=True,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
        )
        ready = self._wait_for_server(timeout=300)
        self._connect_rcon()
        self._set_difficulty_hard()
        if ready:
            print(f"[INST {self.instance_id}] Server laeuft (RCON {self.rcon_port}).")
        else:
            print(f"[INST {self.instance_id}] WARNUNG: RCON in 300s nicht bereit. "
                  f"Boot-Log: {log_path} — Bot-Retry greift trotzdem.")

    def prepare(self):
        """Instanz-Verzeichnis vorbereiten: leer anlegen (falls neu), paper.jar und
        eula.txt aus der Vorlage (TEMPLATE_DIR) uebernehmen, server.properties mit
        eigenen Ports/Passwort erzeugen. Es wird NIE eine Welt aus der Vorlage
        kopiert — frische Instanzen generieren beim Erststart eine neue Vanilla-Welt
        (zufaelliger Seed).
        """
        os.makedirs(self.server_dir, exist_ok=True)

        # paper.jar aus der Vorlage kopieren (JAVA_CMD referenziert paper.jar relativ)
        src_jar = os.path.join(TEMPLATE_DIR, "paper.jar")
        if not os.path.exists(src_jar):
            raise FileNotFoundError(
                f"Vorlage paper.jar fehlt unter {src_jar}. "
                f"Bitte paper.jar in {TEMPLATE_DIR} ablegen."
            )
        dst_jar = os.path.join(self.server_dir, "paper.jar")
        if not os.path.exists(dst_jar):
            shutil.copy2(src_jar, dst_jar)

        # eula.txt uebernehmen (identischer Inhalt: eula=true)
        src_eula = os.path.join(TEMPLATE_DIR, "eula.txt")
        dst_eula = os.path.join(self.server_dir, "eula.txt")
        if os.path.exists(src_eula):
            if not os.path.exists(dst_eula):
                shutil.copy2(src_eula, dst_eula)
        else:
            with open(dst_eula, "w") as f:
                f.write("eula=true\n")

        # server.properties frisch erzeugen (eigene Ports / Passwort)
        self._write_server_properties()

        # Paperclip-Offline-Bootstrap: versions/ + cache/ aus der Vorlage uebernehmen.
        # Lokal bereits materialisiert; auf Colab wird die Vorlage (SERVER_DIR) im
        # Setup per Google Drive geseedet. Damit bootet Paperclip ohne Netzwerk-/
        # Mojang-Download (sonst haengt der Erststart auf Colab ueber Minuten).
        for sub in ("versions", "cache"):
            src_dir = os.path.join(TEMPLATE_DIR, sub)
            dst_dir = os.path.join(self.server_dir, sub)
            if os.path.isdir(src_dir) and not os.path.exists(dst_dir):
                shutil.copytree(src_dir, dst_dir)
                print(f"[INST {self.instance_id}] {sub}/ aus Vorlage uebernommen.")

    def _write_server_properties(self):
        """server.properties mit dieser Instanz' Ports erzeugen. Fehlende Felder
        fuellt Paper mit Standard-Defaults, daher hier nur die relevanten setzen.
        """
        props = {
            "server-port": str(self.game_port),
            "enable-rcon": "true",
            "rcon.port": str(self.rcon_port),
            "rcon.password": self.rcon_password,
            "online-mode": "false",
            "gamemode": "survival",
            "difficulty": "hard",
            "spawn-protection": "0",
            "level-type": "default",
            "view-distance": "8",
            "simulation-distance": "5",
            "max-players": "2",
            "motd": f"MC-ML Instance {self.instance_id}",
            "player-idle-timeout": "0",
            "allow-nether": "true",
        }
        path = os.path.join(self.server_dir, "server.properties")
        with open(path, "w") as f:
            for k, v in props.items():
                f.write(f"{k}={v}\n")
        print(f"[INST {self.instance_id}] server.properties erzeugt (game {self.game_port}, rcon {self.rcon_port}).")

    def _set_difficulty_hard(self):
        """Versucht nach jedem Start via RCON /difficulty hard zu setzen.
        Greift auch auf bestehende Welten, deren level.dat easy/normal hat.
        """
        try:
            self._send_rcon("difficulty hard")
        except Exception as e:
            print(f"[INST {self.instance_id}] Warnung: difficulty hard fehlgeschlagen: {e}")

    def stop(self):
        """Server-Instanz sauber beenden."""
        print(f"[INST {self.instance_id}] Stoppe Server...")
        self._send_rcon("stop")
        if self.process:
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self._disconnect_rcon()
        self.process = None
        if self.log_file is not None:
            try:
                self.log_file.close()
            except Exception:
                pass
            self.log_file = None
        print(f"[INST {self.instance_id}] Server gestoppt.")

    def reset_world(self):
        """Server stoppen, Welt loeschen, neu starten."""
        self.stop()

        world_dirs = ["world", "world_nether", "world_the_end"]
        for world_name in world_dirs:
            world_path = os.path.join(self.server_dir, world_name)
            if os.path.exists(world_path):
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
        return self._send_rcon(command)

    # --- Intern ---

    def _check_java(self):
        """java im PATH erzwingen und Version ausgeben — bevor der Prozess startet."""
        java = shutil.which("java")
        if java is None:
            raise RuntimeError(
                f"[INST {self.instance_id}] 'java' nicht im PATH gefunden — "
                f"colab/setup.sh muss Java 21 installieren."
            )
        try:
            out = subprocess.run([java, "-version"], capture_output=True,
                                 text=True, timeout=10)
            first = (out.stderr or out.stdout).strip().splitlines()[0]
        except Exception:
            first = "java"
        print(f"[INST {self.instance_id}] Java: {first}")

    def _wait_for_server(self, timeout=300):
        print(f"[INST {self.instance_id}] Warte auf Server-Start...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with MCRcon(SERVER_HOST, self.rcon_password, port=self.rcon_port) as mcr:
                    mcr.command("list")
                    return True
            except Exception:
                time.sleep(1)
        print(f"[INST {self.instance_id}] TIMEOUT!")
        return False

    def _connect_rcon(self):
        try:
            self.mcr = MCRcon(SERVER_HOST, self.rcon_password, port=self.rcon_port)
            self.mcr.connect()
        except Exception as e:
            print(f"[INST {self.instance_id}] RCON-Fehler: {e}")

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
                    print(f"  [INST {self.instance_id} RCON] {command} -> {resp}")
                return resp
            except Exception as e:
                print(f"[INST {self.instance_id}] RCON-Fehler: {e}")
        return None


class MultiInstanceManager:
    """Verwaltet K Server-Instanzen."""

    def __init__(self, num_instances, server_base_dir=None, rcon_password="banana1"):
        self.num_instances = num_instances
        self.instances = []

        for i in range(num_instances):
            cfg = self._get_config(i, server_base_dir)
            inst = InstanceManager(
                instance_id=i,
                server_dir=cfg["server_dir"],
                game_port=cfg["game_port"],
                rcon_port=cfg["rcon_port"],
                rcon_password=rcon_password,
            )
            self.instances.append(inst)

    def _get_config(self, instance_id, server_base_dir=None):
        """Config fuer eine Instanz: eigener Port, eigener Dir (immer _<id>).
        """
        if server_base_dir is None:
            from config import SERVER_DIR
            server_base_dir = SERVER_DIR

        suffix = f"_{instance_id}"
        return {
            "server_dir": server_base_dir + suffix,
            "game_port": INSTANCE_BASE_PORT + instance_id,
            "rcon_port": INSTANCE_BASE_RCON + instance_id,
        }

    def start_all(self):
        """Alle Instanzen starten (sequentiell, wegen Port-Konflikte)."""
        for inst in self.instances:
            inst.start()

    def stop_all(self):
        """Alle Instanzen stoppen."""
        for inst in self.instances:
            try:
                inst.stop()
            except Exception as e:
                print(f"[MULTI] Fehler beim Stoppen von Instanz {inst.instance_id}: {e}")

    def reset_world(self, instance_id):
        """Welt einer Instanz zuruecksetzen."""
        self.instances[instance_id].reset_world()

    def get_instance(self, instance_id):
        return self.instances[instance_id]
