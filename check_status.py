import subprocess, os

# Java (Minecraft Server)
java = subprocess.run(["tasklist", "/FI", "IMAGENAME eq java.exe"], capture_output=True, text=True)
server_up = "java.exe" in java.stdout

# Node (JS Bridge)
node = subprocess.run(["tasklist", "/FI", "IMAGENAME eq node.exe"], capture_output=True, text=True)
bridge_up = "node.exe" in node.stdout

# Python (Training)
py = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe"], capture_output=True, text=True)
py_lines = [l for l in py.stdout.splitlines() if "python.exe" in l.lower()]

# Server log
log = "C:/Users/Balda/Desktop/MC_ML/Server/logs/latest.log"
log_age = ""
if os.path.exists(log):
    mtime = os.path.getmtime(log)
    import time
    age_s = time.time() - mtime
    log_age = f"{age_s:.0f}s ago" if age_s < 60 else f"{age_s/60:.0f}min ago"

print(f"Server (Java):  {'UP' if server_up else 'DOWN'}")
print(f"JS Bridge:      {'UP' if bridge_up else 'DOWN'}")
print(f"Python procs:   {len(py_lines)}")
print(f"Server log:     {log_age}")

# Check server log for player
if os.path.exists(log):
    with open(log, "r", errors="ignore") as f:
        lines = f.readlines()
    for line in lines[-50:]:
        if "joined" in line.lower() or "left" in line.lower() or "death" in line.lower():
            print(f"  {line.strip()}")
