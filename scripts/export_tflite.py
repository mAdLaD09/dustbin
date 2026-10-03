import argparse, tensorflow as tf
ap=argparse.ArgumentParser(); ap.add_argument("--model",default="models/best_model.keras"); a=ap.parse_args(); model=tf.keras.models.load_model(a.model)
for name,optim in [("model_float32.tflite",None),("model_float16.tflite",[tf.lite.Optimize.DEFAULT])]:
 c=tf.lite.TFLiteConverter.from_keras_model(model)
 if optim: c.optimizations=optim; c.target_spec.supported_types=[tf.float16]
 open("models/"+name,"wb").write(c.convert())
print("Exported float32 and float16 models. INT8 requires a representative dataset; add it only after validating calibration images.")
