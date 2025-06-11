from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import json

# ========== MODEL VE TRANSFORM ==========
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

model = CNNModel()
model.load_state_dict(torch.load("model/hand_model_Son.pt", map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

label_map = ['DUR', 'ILERI', 'SAG', 'SOL', 'GERI']

# ========== FLASK ==========

app = Flask(__name__)

PI_IP = "10.245.61.138"  # Bu IP'yi sabit tut, sen girersin zaten uygulamadan

@app.route("/confirm_command", methods=["POST"])
def confirm_command():
    try:
        data = request.get_json()
        command = data.get("command")
        confirmation = data.get("confirmation")

        if not command or confirmation not in ["okey", "no"]:
            return jsonify({"error": "Eksik veri"}), 400

        print(f"📥 Onay durumu: {confirmation}, Komut: {command}")

        if confirmation == "okey":
            # 👇 PI’ye komutu gönderiyoruz
            response = requests.post(
                f"http://{PI_IP}:5000/execute_command",
                json={"command": command},
                timeout=3
            )

            if response.status_code == 200:
                print("📤 Komut Pi'ye gönderildi.")
                return jsonify({"status": "command sent to pi"}), 200
            else:
                print("⚠️ Pi'den kötü yanıt geldi.")
                return jsonify({"error": "Pi yanıt vermedi"}), 500

        else:
            print("❌ Komut reddedildi, işlem yok.")
            return jsonify({"status": "reddedildi"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/tahmin", methods=["POST"])
def tahmin():
    file = request.files.get("photo")
    if file is None:
        return jsonify({"hata": "foto eksik"}), 400

    try:
        img = Image.open(file.stream).convert("RGB")
        img = img.rotate(-90, expand=True)
        tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)
            prob = torch.softmax(output, dim=1)
            confidence, pred_idx = torch.max(prob, 1)

        result = {
            "komut": label_map[pred_idx.item()].lower(),
            "confidence": round(confidence.item(), 2)
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"hata": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
