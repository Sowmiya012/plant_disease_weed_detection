# YOLO detection utilities
import torch
import os
from PIL import Image

def load_yolo_model():
    model_path = os.path.join("app", "models", "best.pt")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    return model

def predict_weeds_yolo(image_path):
    model = load_yolo_model()
    results = model(image_path)
    labels, cords = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()

    # Annotated image
    annotated_img = results.render()[0]

    predictions = []
    for i, box in enumerate(cords):
        if box[4] >= 0.3:  # Confidence threshold
            predictions.append({
                "class": int(labels[i]),
                "confidence": float(box[4]),
                "bbox": [float(b) for b in box[:4]]
            })

    return annotated_img, predictions

