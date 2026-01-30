import onnxruntime as ort
import imageio.v2 as imageio
import numpy as np

VIDEO_PATH = "video/sport-union-1190-20260112-fussball-00.11.10.680-00.12.56.790.mkv"

# Load model
session = ort.InferenceSession("models/yolov8n.onnx", providers=["CPUExecutionProvider"])

# Read one frame
reader = imageio.get_reader(VIDEO_PATH)
frame = reader.get_data(0)
reader.close()

# Preprocess
img = frame.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))[None, ...]

# Inference
outputs = session.run(None, {"images": img})

print("Inference OK, outputs:", len(outputs))
