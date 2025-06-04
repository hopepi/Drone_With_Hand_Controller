import socket
import struct
import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

# ========== MODEL TANIMI ==========
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# ========== MODEL YÜKLE ==========
model = CNNModel()
model.load_state_dict(torch.load("model/hand_model_Son.pt", map_location=torch.device('cpu')))
model.eval()

# ========== DÖNÜŞÜM ==========
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# ========== SINIFLAR ==========
label_map = ['DUR', 'ILERI', 'SAG', 'SOL', 'GERI']

# ========== SOCKET SERVER ==========
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
            # MODE BYTE
            while len(buffer) < 1:
                buffer += conn.recv(4096)
            mode = buffer[:1].decode()
            buffer = buffer[1:]
            if mode != "E":
                print("⛔ Yanlış mod geldi:", mode)
                conn.close()
                server_socket.close()
                continue

            # BOYUT
            while len(buffer) < 4:
                buffer += conn.recv(4096)
            img_size = struct.unpack(">L", buffer[:4])[0]
            buffer = buffer[4:]

            # GÖRÜNTÜ
            while len(buffer) < img_size:
                buffer += conn.recv(4096)
            img_data = buffer[:img_size]
            buffer = buffer[img_size:]

            # ========== TAHMİN BAŞLA ==========
            image = Image.open(io.BytesIO(img_data)).convert("RGB")
            image = image.rotate(-90, expand=True)  # 🔁 90° saat yönü

            img_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(img_tensor)
                prob = torch.softmax(output, dim=1)
                confidence, pred_idx = torch.max(prob, 1)

            predicted_label = label_map[pred_idx.item()]
            conf_score = confidence.item()

            print(f"✅ Tahmin: {predicted_label} | Güven: {conf_score:.2f}")

            result_json = json.dumps({
                "status": "el_komutu",
                "komut": predicted_label.lower(),
                "confidence": round(conf_score, 2)
            }).encode("utf-8")

            conn.sendall(struct.pack(">L", len(result_json)) + result_json)

        except Exception as e:
            print("⛔ Hata:", e)

        finally:
            try: conn.close()
            except: pass
            try: server_socket.close()
            except: pass
            print("🔁 Yeni bağlantı için hazır...\n")

if __name__ == "__main__":
    run_hand_server()
