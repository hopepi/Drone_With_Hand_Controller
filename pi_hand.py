import socket
import struct
import json
import threading
import time
from flask import Flask, request, jsonify
import random
import atexit
import control

# ---------- AYARLAR ----------
SERVER_IP = '10.245.198.73'  # 📡 PC'deki tahmin sunucusunun IP'si
SERVER_PORT = 8000           # 📡 El komut sunucusu portu

# ---------- GLOBAL DURUMLAR ----------
drone_state = "land"
emergency_flag = False
current_altitude = 1.0

# ---------- LOG FONKSİYONU ----------
def log(msg, level="info"):
    renk = {
        "info": "\033[94m", "warning": "\033[93m",
        "danger": "\033[91m", "success": "\033[92m", "end": "\033[0m"
    }
    print(f"{renk.get(level, '')}{msg}{renk['end']}")

# ---------- DRONE TEMEL FONKSİYONLARI ----------
def takeoff_and_hover(target_altitude=1.0):
    global current_altitude, drone_state

    # Zaten havadaysa tekrar kalkmaya gerek yok
    if drone_state != "land":
        log("Drone zaten havada, kalkış iptal.", "warning")
        return

    target_altitude = max(1.0, min(3.0, float(target_altitude)))
    current_altitude = target_altitude
    log(f"Kalkış! {target_altitude:.1f} metreye yükseliyor...", "success")
    control.arm_and_takeoff(target_altitude)
    drone_state = "guide"


def land_drone():
    global current_altitude,drone_state
    log("Drone iniş yapıyor...", "danger")
    current_altitude = 0.0
    drone_state = "land"
    control.land()

atexit.register(land_drone)

# ---------- SETUP ----------
def setup():
    while True:
        try:
            log("Drone bağlantısı deneniyor...", "info")
            control.connect_drone("/dev/serial0")
            log("Drone bağlantısı başarılı", "success")
            break
        except Exception as e:
            log(f"Drone bağlantı hatası: {e}", "danger")
            log("3 saniye sonra tekrar deneniyor...", "warning")
            time.sleep(3)

    try:
        control.configure_PID()
        log("PID konfigürasyonu tamamlandı", "success")
    except Exception as e:
        log(f"PID konfigürasyon hatası: {e}", "danger")
        exit(1)

    takeoff_and_hover(1.0)
    log("🟢 Sistem hazır. El komutu bekleniyor...", "success")

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

    # Hatalı veya eksik parametre kontrolü
    if mode is None or altitude is None:
        log("Eksik parametre: mode veya altitude yok", "warning")
        return jsonify({"status": "eksik parametre"}), 400

    log(f"Komut alındı | Mod: {mode}, İrtifa: {altitude}", "info")

    try:
        alt = float(altitude)
    except (ValueError, TypeError):
        alt = 1.0

    alt = max(1.0, min(3.0, alt))

    # Sadece "land" durumundayken kalkış yapılabilir
    if current_altitude != alt and drone_state == "land":
        takeoff_and_hover(alt)
    elif current_altitude != alt:
        log("İrtifa isteği geldi ama drone havada, kalkış yapılmadı.", "warning")

    if mode != "el":
        return jsonify({"status": "geçersiz mod"}), 400

    drone_state = "guide"

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
        log(f"PC'den gelen komut tahmini: {command_json}", "info")
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

    # Eksik veri kontrolü
    if not command or not confirmation:
        log("Eksik onay parametreleri alındı", "warning")
        return jsonify({"status": "eksik veri"}), 400

    log(f"Komut onayı alındı → Komut: {command}, Cevap: {confirmation}", "info")

    valid_commands = ["sag", "sol", "ileri", "geri", "dur"]

    if confirmation == "okey" and not emergency_flag:
        if command in valid_commands:
            drone_state = command
            log(f"Komut ONAYLANDI → Drone hareket: {command}", "success")
            control.apply_hand_command(command)
        else:
            log(f"Geçersiz komut onaylandı (yok sayıldı): {command}", "warning")
    else:
        drone_state = "hover"
        log("Komut REDDEDİLDİ → Drone hover modda bekliyor", "warning")

    return "", 204


@app.route("/emergency", methods=["POST"])
def emergency():
    global emergency_flag, drone_state
    emergency_flag = True
    drone_state = "emergency"
    log("‼ACİL DURDURMA KOMUTU ALINDI", "danger")
    control.stop_drone()
    drone_state = "hover"
    return jsonify({"status": "acil durdurma tamamlandı"})

@app.route("/resume", methods=["POST"])
def resume():
    global emergency_flag, drone_state
    emergency_flag = False
    drone_state = "hover"
    log("Emergency bitti, sistem tekrar aktif", "success")
    return jsonify({"status": "emergency sıfırlandı, devam edebilirsiniz"})

# ---------- MAIN ----------
if __name__ == "__main__":
    setup()
    log("El Komut Sunucusu başlatılıyor...", "info")
    app.run(host="0.0.0.0", port=5000)
