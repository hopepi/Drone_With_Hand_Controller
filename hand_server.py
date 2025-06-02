import socket
import struct
import json

HOST = '0.0.0.0'
PORT = 8000

def run_hand_server():
    while True:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"✋ EL KOMUTU SERVER başladı: {HOST}:{PORT}, bağlantı bekleniyor...")

        conn, addr = server_socket.accept()
        print(f"📡 Bağlantı geldi: {addr}")
        buffer = b""

        try:
            # 1. MODE BYTE (b'E' olmalı)
            while len(buffer) < 1:
                recv_data = conn.recv(4096)
                if not recv_data:
                    raise ConnectionError("Bağlantı koptu!")
                buffer += recv_data
            mode_byte = buffer[:1]
            mode = mode_byte.decode()
            buffer = buffer[1:]

            if mode != "E":
                print("Yanlış mod! Sadece E (el) kabul ediliyor.")
                conn.close()
                server_socket.close()
                continue

            # 2. FOTOĞRAF BOYUTU (4 byte)
            while len(buffer) < 4:
                recv_data = conn.recv(4096)
                if not recv_data:
                    raise ConnectionError("Bağlantı koptu!")
                buffer += recv_data
            img_size = struct.unpack(">L", buffer[:4])[0]
            buffer = buffer[4:]

            # 3. FOTOĞRAF VERİSİ
            while len(buffer) < img_size:
                recv_data = conn.recv(4096)
                if not recv_data:
                    raise ConnectionError("Bağlantı koptu!")
                buffer += recv_data
            img_data = buffer[:img_size]
            buffer = buffer[img_size:]

            # ARTIK img_data = jpg byte'ı! (İster modele, ister dosyaya)
            # Örnek olarak dosyaya kaydedelim:
            with open("last_hand.jpg", "wb") as f:
                f.write(img_data)
            print("Gelen fotoğraf kaydedildi: last_hand.jpg")

            # Modelden çıkan sonucu burada yaz (dummy örnek):
            dummy_result = {
                "status": "el_komutu",
                "komut": "saga_don",
                "confidence": 0.95
            }
            result_json = json.dumps(dummy_result).encode('utf-8')

            # Cevabı gönder (uzunluk + json)
            conn.sendall(struct.pack(">L", len(result_json)) + result_json)

        except Exception as e:
            print("⛔ Hata:", e)
        finally:
            try: conn.close()
            except: pass
            try: server_socket.close()
            except: pass
            print("⏳ Yeniden bağlantı bekleniyor...\n")

if __name__ == "__main__":
    run_hand_server()
