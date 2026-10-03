"""Evaluate a Keras model on a held-out directory and write metrics."""
import argparse, json
from pathlib import Path
import numpy as np, tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
ap=argparse.ArgumentParser(); ap.add_argument("--model",default="models/best_model.keras"); ap.add_argument("--data",default="datasets/processed"); a=ap.parse_args(); ds=tf.keras.utils.image_dataset_from_directory(a.data,validation_split=.3,subset="validation",seed=42,image_size=(224,224),batch_size=32); m=tf.keras.models.load_model(a.model); y=np.concatenate([y for _,y in ds]); p=np.argmax(m.predict(ds),1); result=classification_report(y,p,target_names=ds.class_names,output_dict=True); Path("outputs").mkdir(exist_ok=True); Path("outputs/evaluation.json").write_text(json.dumps({"classification_report":result,"confusion_matrix":confusion_matrix(y,p).tolist()},indent=2)); print(json.dumps(result,indent=2))
