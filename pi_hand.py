import socket
import struct
import cv2
import time
import json
import threading
from flask import Flask, request, jsonify
import random
import atexit

# ---------- AYARLAR ----------
SERVER_IP = '127.0.0.1'  # PC/server'ın IP'si
SERVER_PORT = 8000       # PC tarafı port
FPS = 15
FRAME_DELAY = 1 / FPS

# ---------- GLOBAL DEĞİŞKENLER ----------
drone_state = "land"
emergency_flag = False
current_altitude = 1.0  # Varsayılan 1m

def log(msg, level="info"):
    renk = {"info": "\033[94m", "warning": "\033[93m", "danger": "\033[91m", "success": "\033[92m", "end": "\033[0m"}
    print(f"{renk.get(level, '')}{msg}{renk['end']}")

def takeoff_and_hover(target_altitude=1.0):
    if target_altitude < 1.0: target_altitude = 1.0
    if target_altitude > 5.0: target_altitude = 5.0
    log(f"🚁 Kalkış! {target_altitude:.1f} metreye yükseliyor...", "success")
    global current_altitude
    current_altitude = target_altitude

def land_drone():
    log("🛬 Drone iniş yapıyor...", "danger")
    global current_altitude
    current_altitude = 0.0

atexit.register(land_drone)

# ---------- FLASK API ----------
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
    if alt < 1.0: alt = 1.0
    if alt > 5.0: alt = 5.0
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
    """
    Mobil uygulamadan gelen el komut fotoğrafını PC'ye iletir ve sonucu döner.
    """
    file = request.files.get("photo")
    if file is None:
        return jsonify({"result": "Fotoğraf yok"}), 400

    img_bytes = file.read()

    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((SERVER_IP, SERVER_PORT))
        client_socket.sendall(b'E')
        client_socket.sendall(struct.pack(">L", len(img_bytes)) + img_bytes)

        # 3. PC'den sonucu al - Tüm cevabı tam oku!
        cmd_len_bytes = client_socket.recv(4)
        cmd_len = struct.unpack(">L", cmd_len_bytes)[0]
        data = b""
        while len(data) < cmd_len:
            part = client_socket.recv(cmd_len - len(data))
            if not part:
                break
            data += part
        command_json = data.decode('utf-8')
        print("PC'den gelen cevap:", command_json) 

        result_data = json.loads(command_json)
        client_socket.close()
    except Exception as e:
        print("EL KOMUTU PC'ye gönderilemedi:", e)
        return jsonify({"result": f"PC'ye iletilemedi: {str(e)}"}), 500

    return jsonify(result_data or {"result": "Bilinmeyen hata"})


@app.route("/emergency", methods=["POST"])
def emergency():
    global emergency_flag, drone_state
    emergency_flag = True
    drone_state = "emergency"
    log("‼️ ACİL DURDURMA KOMUTU ALINDI", "danger")
    return jsonify({"status": "acil durdurma tamamlandı"})

@app.route("/resume", methods=["POST"])
def resume():
    global emergency_flag, drone_state
    emergency_flag = False
    drone_state = "land"
    log("✅ Emergency bitti, sistem tekrar aktif", "success")
    return jsonify({"status": "emergency sıfırlandı, devam edebilirsiniz"})

if __name__ == "__main__":
    takeoff_and_hover(1.0)  # Başlangıç kalkışı
    log("🤖 Hand Command Pi sunucu başlatılıyor...", "info")
    app.run(host="0.0.0.0", port=5000)
