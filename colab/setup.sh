#!/bin/bash
# Colab-Setup: Idempotent. Java 21, Node 22 (nvm), Python-Deps, Server-Assets,
# Paperclip-Offline-Cache (Drive), JS-Bridge-Warmup. Laueft im Colab-VM als root
# (kein sudo noetig). Eingebunden aus colab/colab_train.sh; alternativ manuell:
#   echo '!bash /content/MC_ML_Code/colab/setup.sh' | colab exec -s <session>
set -e

REPO_DIR="/content/MC_ML_Code"
SERVER_DIR="/content/mcml/Server"
DRIVE_ROOT="/content/drive/MyDrive/MC_ML"
DRIVE_CACHE="$DRIVE_ROOT/server_cache"

echo "=== [SETUP] Start ==="

# --- Java 21 (Paper braucht Java 21) ---
if java -version 2>&1 | grep -q 'version "21'; then
  echo "Java 21 bereits vorhanden."
else
  echo "Installiere Java 21 (openjdk-21-jre-headless)..."
  apt-get update -qq
  apt-get install -y -qq openjdk-21-jre-headless || {
    echo "apt-Java fehlgeschlagen -> Temurin-21-Fallback (Tarball)..."
    curl -sL -o /tmp/jdk21.tar.gz \
      "https://api.adoptium.net/v3/binary/latest/21/ga/linux/x64/jre/hotspot/normal/eclipse"
    mkdir -p /opt/jdk21
    tar xzf /tmp/jdk21.tar.gz -C /opt/jdk21 --strip-components=1
    ln -sf /opt/jdk21/bin/java /usr/local/bin/java
  }
fi
java -version 2>&1 | head -1

# --- Node 22 LTS via nvm (Colab-apt hat nur Node 12 -> zu alt fuer mineflayer;
# mineflayer>=4.39 verlangt node>=22 -> nicht Node 20 verwenden) ---
export NVM_DIR="$HOME/.nvm"
if [ ! -s "$NVM_DIR/nvm.sh" ]; then
  echo "Installiere nvm..."
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
fi
. "$NVM_DIR/nvm.sh"
nvm install 22 >/dev/null 2>&1 || true
nvm use 22 >/dev/null 2>&1 || true
# Node/npm systemweit verfuegbar machen (jede colab-exec-Shell hat sonst frisches PATH).
NODE_BIN_DIR="$(dirname "$(command -v node)")"
ln -sf "$NODE_BIN_DIR/node" /usr/local/bin/node
ln -sf "$NODE_BIN_DIR/npm" /usr/local/bin/npm
echo "Node: $(node --version) | npm: $(npm --version)"

# --- Python-Deps ---
# Fallback 1: Colab-vorinstalliertes torch/gymnasium/numpy/tensorboard wiederverwenden
# (sb3 ohne deps -> kein 2-3GB torch-Wheel-Download). Fallback 2: volle requirements.
echo "Installiere Python-Deps..."
python -m pip install -q --upgrade pip 2>&1 | tail -1 || true
if python -c "import gymnasium, torch, numpy, tensorboard" 2>/dev/null; then
  python -m pip install -q stable-baselines3 --no-deps
else
  python -m pip install -q -r "$REPO_DIR/requirements.txt"
fi
python -m pip install -q mcrcon javascript

# --- Server-Assets: Template-Ordner fuer InstanceManager ---
mkdir -p "$SERVER_DIR"
cp -n "$REPO_DIR/server_assets/paper.jar" "$SERVER_DIR/paper.jar" || true
cp -n "$REPO_DIR/server_assets/eula.txt" "$SERVER_DIR/eula.txt" || true
ls -la "$SERVER_DIR" | grep -E "paper.jar|eula.txt" || exit 1

# --- Paperclip-Offline-Cache von Google Drive ---
# Fehlt mojang_26.1.2.jar, versucht Paperclip beim 1. Boot die Server-Jar von
# Mojang herunterzuladen -> haengt/stallt auf Colab. Der Cache wird per find im
# gesamten gemounteten Drive gesucht — egal wo abgelegt (empfohlen
# MyDrive/MC_ML/server_cache/, funktioniert auch direkt irgendwo unter MyDrive).
echo "Suche Paperclip-Bootstrap-Cache auf Google Drive..."
HAS_CACHE=0
if [ -d /content/drive ]; then
  CACHE_JAR=""
  if [ -f "$DRIVE_CACHE/cache/mojang_26.1.2.jar" ]; then
    CACHE_JAR="$DRIVE_CACHE/cache/mojang_26.1.2.jar"
  fi
  if [ -z "$CACHE_JAR" ]; then
    CACHE_JAR="$(find /content/drive -maxdepth 6 -type f -name 'mojang_26.1.2.jar' 2>/dev/null | head -1)"
  fi
  if [ -n "$CACHE_JAR" ]; then
    mkdir -p "$SERVER_DIR/cache"
    cp -f "$CACHE_JAR" "$SERVER_DIR/cache/mojang_26.1.2.jar"
    echo "OK: cache/mojang_26.1.2.jar von $CACHE_JAR -> $SERVER_DIR/cache/"
    HAS_CACHE=1
  fi

  # Optional versions/-Mirror (offline-materialisierte Server-Jar) irgendwo auf Drive
  if [ ! -d "$SERVER_DIR/versions" ]; then
    VER_DIR_SRC="$(find /content/drive -maxdepth 7 -type d -path '*/versions/26.1.2' 2>/dev/null | head -1)"
    if [ -n "$VER_DIR_SRC" ]; then
      mkdir -p "$SERVER_DIR/versions"
      cp -rn "$VER_DIR_SRC/." "$SERVER_DIR/versions/"
      echo "OK: versions/-Mirror von $VER_DIR_SRC -> $SERVER_DIR/versions/"
      HAS_CACHE=1
    fi
  fi
fi
if [ "$HAS_CACHE" = "0" ]; then
  echo "WARNUNG: Kein mojang_26.1.2.jar auf Google Drive gefunden!"
  echo "         -> Paperclip wird die Server-Jar von Mojang laden (kann auf Colab"
  echo "            fehlschlagen/laengere Erstboot-Zeit). Fix: die Datei aus"
  echo "            Server/cache/ irgendwo unter MyDrive ablegen."
fi
ls -la "$SERVER_DIR" 2>/dev/null | grep -E "cache|versions" || echo "(keine Bootstrap-Caches materialisiert)"

# --- JS-Bridge-Warmup: npm-Module EINMAL installieren, BEVOR SubprocVecEnv fork-t ---
# (Sonst installieren 3 Worker gleichzeitig in dasselbe node_modules -> Race/Corruption).
echo "Warme JS-Bridge (npm-Module installieren)..."
python - <<'PY'
from javascript import require
require("mineflayer")
require("vec3")
require("minecraft-data")
print("Bridge-Warmup OK")
PY

echo "=== [SETUP] Fertig ==="