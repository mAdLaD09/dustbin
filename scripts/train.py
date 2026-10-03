"""MobileNetV3-Small transfer-learning training entry point."""
import argparse, json
from pathlib import Path
import tensorflow as tf
from src.common.config import load_config

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--config",default="config.yaml"); ap.add_argument("--epochs",type=int,default=20); a=ap.parse_args(); cfg=load_config(a.config); size=cfg["model"]["image_size"]; root=Path("datasets/processed")
 train=tf.keras.utils.image_dataset_from_directory(root,validation_split=.3,subset="training",seed=42,image_size=(size,size),batch_size=32); valtest=tf.keras.utils.image_dataset_from_directory(root,validation_split=.3,subset="validation",seed=42,image_size=(size,size),batch_size=32); names=train.class_names
 aug=tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"),tf.keras.layers.RandomRotation(.04),tf.keras.layers.RandomZoom(.1),tf.keras.layers.RandomContrast(.1)])
 base=tf.keras.applications.MobileNetV3Small(input_shape=(size,size,3),include_top=False,weights="imagenet",include_preprocessing=False); base.trainable=False; inp=tf.keras.Input((size,size,3)); x=aug(inp); x=tf.keras.layers.Rescaling(1/127.5,offset=-1)(x); x=base(x,training=False); x=tf.keras.layers.GlobalAveragePooling2D()(x); out=tf.keras.layers.Dense(len(names),activation="softmax")(x); model=tf.keras.Model(inp,out); model.compile("adam","sparse_categorical_crossentropy",metrics=["accuracy"])
 Path("models").mkdir(exist_ok=True); model.fit(train,validation_data=valtest,epochs=a.epochs,callbacks=[tf.keras.callbacks.EarlyStopping(patience=4,restore_best_weights=True),tf.keras.callbacks.ModelCheckpoint("models/best_model.keras",save_best_only=True)]); Path("models/labels.txt").write_text("\n".join(names),encoding="utf-8")
if __name__=="__main__": main()
