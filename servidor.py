from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)

print("Intentando cargar modelo...")
try:
    from ultralytics import YOLO
    model = YOLO("best.pt")
    print("Modelo YOLO cargado.")
except Exception as e:
    print(f"Error cargando YOLO: {e}")
    print("El servidor seguira funcionando pero sin detecciones.")
    model = None

try:
    with open("model/labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
except:
    class_names = ["cpu", "mesa", "mouse", "pantalla", "silla", "teclado"]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Modelo no cargado"}), 500
        
        data = request.json
        image_base64 = data.get("image")
        
        if not image_base64:
            return jsonify({"error": "No image"}), 400
        
        # Decodificar imagen
        image_data = base64.b64decode(image_base64.split(",")[1])
        img = Image.open(io.BytesIO(image_data))
        img_array = np.array(img)
        
        # Deteccion
        results = model(img_array, conf=0.45)
        
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                
                detections.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": conf,
                    "class": cls,
                    "class_name": class_names[cls] if cls < len(class_names) else f"id:{cls}"
                })
        
        return jsonify({"detections": detections})
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "OK"})

if __name__ == "__main__":
    print("Servidor corriendo en http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)