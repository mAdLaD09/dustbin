from __future__ import annotations
import time
from pathlib import Path
import numpy as np
from PIL import Image

class TFLiteClassifier:
    def __init__(self, model_path, labels, image_size=224, normalization="zero_one"):
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite import Interpreter
        self.interpreter=Interpreter(model_path=str(model_path)); self.interpreter.allocate_tensors(); self.input=self.interpreter.get_input_details()[0]; self.output=self.interpreter.get_output_details()[0]; self.labels=labels; self.image_size=image_size; self.normalization=normalization
    def preprocess(self, image):
        if not isinstance(image, Image.Image): image=Image.fromarray(image[:, :, ::-1])
        x=np.asarray(image.convert("RGB").resize((self.image_size,self.image_size)),dtype=np.float32)
        if self.input["dtype"] in (np.uint8,np.int8):
            x=x/255.0 if self.normalization=="zero_one" else (x/127.5)-1.0
            scale,zero=self.input["quantization"]
            x=np.round(x/scale+zero).clip(np.iinfo(self.input["dtype"]).min,np.iinfo(self.input["dtype"]).max).astype(self.input["dtype"])
        else: x=x/255.0 if self.normalization=="zero_one" else (x/127.5)-1.0
        return np.expand_dims(x,0)
    def predict(self, image):
        start=time.perf_counter(); x=self.preprocess(image); self.interpreter.set_tensor(self.input["index"],x); self.interpreter.invoke(); y=self.interpreter.get_tensor(self.output["index"])[0]
        if self.output["dtype"] in (np.uint8,np.int8):
            scale,zero=self.output["quantization"]; y=(y.astype(np.float32)-zero)*scale
        y=np.exp(y-np.max(y)); y=y/y.sum(); index=int(np.argmax(y)); return self.labels[index],float(y[index]),(time.perf_counter()-start)*1000

def load_labels(path): return [x.strip() for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
