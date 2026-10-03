import argparse, sys, time
from pathlib import Path
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common.config import load_config
from src.pi.classifier import TFLiteClassifier, load_labels
ap=argparse.ArgumentParser(); ap.add_argument("--image",required=True); a=ap.parse_args(); cfg=load_config(); model=Path(cfg["model"]["path"])
if not model.exists(): raise SystemExit(f"Model missing: {model}")
c=TFLiteClassifier(model,load_labels("models/labels.txt"),cfg["model"]["image_size"],cfg["model"]["normalization"]); image=cv2.imread(a.image)
if image is None: raise SystemExit(f"Could not read image: {a.image}")
label,confidence,lat=c.predict(image); print(f"Predicted: {label}\nConfidence: {confidence:.3f}\nInference time: {lat:.0f} ms")
