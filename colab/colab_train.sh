#!/bin/bash
# Colab-Run: Session aufsetzen -> Google Drive mounten -> setup.sh -> train_yard.py
# SYNCHRON ausfuehren. Synchron = die CLI bleibt am Prozess dran und streamt die
# Live-Ausgabe (kein nohup/setsid: detached Prozesse werden vom Colab-Runtime beim
# Session-Teardown gekillt). Persistenz: train_yard.py kopiert das eine Modell-File
# 1x/Minute UND am Ende zusaetzlich nach Google Drive.
# Vor dem Run den aktuellen Code committen und pushen, sonst laeuft der Clone den
# alten Stand. Resume: bei nachfolgendem Start wird das letzte Drive-Modell wieder
# in den frischen Clone gelegt -> load_or_create_model setzt nahtlos an.

S=mcml
# Runtime-Wahl: leer = CPU. GPU bringt hier nichts (MLP 256x256, Bottleneck ist die
# 20-TPS-Env-Schleife, nichts davon laeuft auf der GPU).
ACCEL=""
INSTANCES=2
REPO="https://github.com/s8dabald/MC_ML_Code.git"
DRIVE_MODEL="/content/drive/MyDrive/MC_ML/ppo_minecraft_final.zip"

# Stale Session vom letzten Mal aufraeumen (falls vorhanden)
colab stop -s $S 2>/dev/null || true

colab new -s $S $ACCEL

# Google Drive mounten, damit das Modell-File das VM-Clearing ueberlebt
colab drivemount -s $S

# Erinnerung: Paperclip-Bootstrap-Cache erwartet (Einmal-Upload von lokal):
#   mojang_26.1.2.jar (aus Server/cache/) — irgendwo unter MyDrive reicht,
#   setup.sh finder sie automatisch per find (siehe setup.sh).
echo "!echo 'Cache-Pruefung: suche mojang_26.1.2.jar auf dem gemounteten Drive...' && find /content/drive -maxdepth 6 -name 'mojang_26.1.2.jar' 2>/dev/null | head -1 || echo '  (kein Drive-Cache -> siehe setup.sh-Warnung)'" | colab exec -s $S

# Frischer Clone, damit gepushte Aenderungen sicher ankommen
echo "!cd /content && rm -rf MC_ML_Code && git clone $REPO" | colab exec -s $S

# Letztes Drive-Modell zurueckholen (Resume) — fehlt, startet frisches Training
echo "!mkdir -p /content/MC_ML_Code && if [ -f $DRIVE_MODEL ]; then cp $DRIVE_MODEL /content/MC_ML_Code/ppo_minecraft_final.zip && echo 'Drive-Modell geladen'; else echo 'Kein Drive-Modell -> frisches Training'; fi" | colab exec -s $S

# Einrichtung (idempotent): deps, java 21, node 22, assets, Drive-Bootstrap-Cache, bridge-warmup
echo "Setup läuft auf $S..."
echo "!bash /content/MC_ML_Code/colab/setup.sh" | colab exec -s $S

echo "Run gestartet (synchron auf $S). Streamt Live-Ausgabe..."
echo "!cd /content/MC_ML_Code && python -u train_yard.py -n $INSTANCES" | colab exec -s $S

# Falls train_yard.py es beim Exit nicht geschafft hat, letzter Sicherheits-Copy
echo "!if [ -f /content/MC_ML_Code/ppo_minecraft_final.zip ]; then cp -f /content/MC_ML_Code/ppo_minecraft_final.zip $DRIVE_MODEL; fi" | colab exec -s $S

colab stop -s $S