import socket
import struct
import json
import threading
from flask import Flask, request, jsonify
import random
import atexit
import control  # 🧠 Drone komutları buradan çağrılır

# ---------- AYARLAR ----------
SERVER_IP = '10.245.198.73'  # PC/server IP
SERVER_PORT = 8000       # El komut server portu

# ---------- GLOBAL DEĞİŞKENLER ----------
drone_state = "land"
emergency_flag = False
current_altitude = 1.0  # Varsayılan başlangıç yüksekliği

# ---------- LOG FONKSİYONU ----------
def log(msg, level="info"):
    renk = {
        "info": "\033[94m", "warning": "\033[93m",
        "danger": "\033[91m", "success": "\033[92m", "end": "\033[0m"
    }
    print(f"{renk.get(level, '')}{msg}{renk['end']}")

# ---------- KALKIŞ & İNİŞ ----------
def takeoff_and_hover(target_altitude=1.0):
    if target_altitude < 1.0: target_altitude = 1.0
    if target_altitude > 3.0: target_altitude = 3.0
    log(f"🚁 Kalkış! {target_altitude:.1f} metreye yükseliyor...", "success")
    global current_altitude
    current_altitude = target_altitude
    control.arm_and_takeoff(target_altitude)

def land_drone():
    log("🛬 Drone iniş yapıyor...", "danger")
    global current_altitude
    current_altitude = 0.0
    control.land()

atexit.register(land_drone)

# ---------- FLASK APP ----------
app = Flask(__name__)

@app.route("/ping")
def ping():
    return "pong", 200

@app.route("/command", methods=["POST"])
def command():
    global drone_state, current_altitude
    data = request.json
    mode = data.get("mode")
    altitude = data.get("altitude")
    log(f"📥 Komut alındı | Mod: {mode}, İrtifa: {altitude}", "info")

    try:
        alt = float(altitude)
    except (ValueError, TypeError):
        alt = 1.0

    alt = max(1.0, min(3.0, alt))
    if current_altitude != alt:
        takeoff_and_hover(alt)

    drone_state = "track"

    if mode not in ["el"]:
        return jsonify({"status": "geçersiz mod"}), 400

    return jsonify({
        "status": f"{mode} modu başlatıldı (irtifa: {alt:.1f} m)",
        "id": random.randint(10000, 99999)
    })

@app.route("/hand_command", methods=["POST"])
def hand_command():
    file = request.files.get("photo")
    if file is None:
        return jsonify({"result": "Fotoğraf yok"}), 400

    img_bytes = file.read()

    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((SERVER_IP, SERVER_PORT))
        client_socket.sendall(b'E')
        client_socket.sendall(struct.pack(">L", len(img_bytes)) + img_bytes)

        # Tahmin sonucunu al
        cmd_len_bytes = client_socket.recv(4)
        cmd_len = struct.unpack(">L", cmd_len_bytes)[0]
        data = b""
        while len(data) < cmd_len:
            part = client_socket.recv(cmd_len - len(data))
            if not part:
                break
            data += part
        command_json = data.decode('utf-8')
        log(f"📨 PC'den gelen komut tahmini: {command_json}", "info")

        result_data = json.loads(command_json)
        client_socket.close()
    except Exception as e:
        log(f"EL KOMUTU gönderilemedi: {str(e)}", "danger")
        return jsonify({"result": f"PC'ye iletilemedi: {str(e)}"}), 500

    return jsonify(result_data or {"result": "Bilinmeyen hata"})

@app.route("/confirm_command", methods=["POST"])
def confirm_command():
    global drone_state
    data = request.json
    command = data.get("command")
    confirmation = data.get("confirmation")

    log(f"📩 Komut onayı alındı → Komut: {command}, Cevap: {confirmation}", "info")

    valid_commands = ["sag", "sol", "ileri", "geri", "dur"]

    if confirmation == "okey" and not emergency_flag:
        if command in valid_commands:
            drone_state = command
            log(f"✅ Komut ONAYLANDI → Drone hareket: {command}", "success")

            # 🔥 El komutuna göre hareket başlat
            control.apply_hand_command(command)
        else:
            log(f"⚠️ Geçersiz komut onaylandı (yok sayıldı): {command}", "warning")
    else:
        drone_state = "hover"
        log("❌ Komut REDDEDİLDİ → Drone hover modda bekliyor", "warning")

    return "", 204  # No Content

@app.route("/emergency", methods=["POST"])
def emergency():
    global emergency_flag, drone_state
    emergency_flag = True
    drone_state = "emergency"
    log("‼️ ACİL DURDURMA KOMUTU ALINDI", "danger")
    control.stop_drone()
    return jsonify({"status": "acil durdurma tamamlandı"})

@app.route("/resume", methods=["POST"])
def resume():
    global emergency_flag, drone_state
    emergency_flag = False
    drone_state = "land"
    log("✅ Emergency bitti, sistem tekrar aktif", "success")
    return jsonify({"status": "emergency sıfırlandı, devam edebilirsiniz"})

if __name__ == "__main__":
    control.configure_PID()
    control.connect_drone("/dev/serial0")  # ← Bağlantı noktan buysa
    takeoff_and_hover(1.0)
    log("🤖 El Komut Sunucusu başlatılıyor...", "info")
    app.run(host="0.0.0.0", port=5000)
