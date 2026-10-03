"""Measure local TFLite latency; run on the Pi for meaningful hardware numbers."""
import argparse, time, json
from pathlib import Path
import numpy as np
from PIL import Image
from src.pi.classifier import TFLiteClassifier, load_labels
ap=argparse.ArgumentParser(); ap.add_argument("--model",default="models/model_float32.tflite"); ap.add_argument("--image",required=True); ap.add_argument("--runs",type=int,default=50); a=ap.parse_args(); c=TFLiteClassifier(a.model,load_labels("models/labels.txt")); im=Image.open(a.image); times=[]
for _ in range(a.runs):
 t=time.perf_counter(); c.predict(im); times.append((time.perf_counter()-t)*1000)
r={"model":a.model,"file_size_bytes":Path(a.model).stat().st_size,"runs":a.runs,"mean_ms":float(np.mean(times)),"median_ms":float(np.median(times)),"p95_ms":float(np.percentile(times,95)),"fps":float(1000/np.mean(times))}
print(json.dumps(r,indent=2))
