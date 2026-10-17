"""MobileNetV3-Small transfer-learning training entry point."""
import argparse
import json
from pathlib import Path
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.common.config import load_config


def make_dataset(paths, labels, size, batch_size=32, shuffle=False):
    path_ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, (size, size))
        return image, label

    dataset = path_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(len(paths), seed=42)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--epochs", type=int, default=20)
    a = ap.parse_args()
    cfg = load_config(a.config)
    size = cfg["model"]["image_size"]
    root = Path("datasets/processed")
    names = sorted(path.name for path in root.iterdir() if path.is_dir())
    name_to_index = {name: index for index, name in enumerate(names)}
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    all_paths = []
    all_labels = []
    for class_name in names:
        for path in sorted((root / class_name).rglob("*")):
            if path.is_file() and path.suffix.lower() in image_extensions:
                all_paths.append(str(path))
                all_labels.append(name_to_index[class_name])

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths,
        all_labels,
        test_size=0.3,
        random_state=42,
        stratify=all_labels,
    )
    train = make_dataset(train_paths, train_labels, size, shuffle=True)
    valtest = make_dataset(val_paths, val_labels, size)
    aug = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(
        .04), tf.keras.layers.RandomZoom(.1), tf.keras.layers.RandomContrast(.1)])
    base = tf.keras.applications.MobileNetV3Small(input_shape=(
        size, size, 3), include_top=False, weights="imagenet", include_preprocessing=False)
    base.trainable = False
    inp = tf.keras.Input((size, size, 3))
    x = aug(inp)
    x = tf.keras.layers.Rescaling(1/127.5, offset=-1)(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(len(names), activation="softmax")(x)
    model = tf.keras.Model(inp, out)
    model.compile("adam", "sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    Path("models").mkdir(exist_ok=True)
    model.fit(train, validation_data=valtest, epochs=a.epochs, callbacks=[tf.keras.callbacks.EarlyStopping(
        patience=4, restore_best_weights=True), tf.keras.callbacks.ModelCheckpoint("models/best_model.keras", save_best_only=True)])
    Path("models/labels.txt").write_text("\n".join(names), encoding="utf-8")
    test_loss, test_accuracy = model.evaluate(valtest)
    print("Test loss:", test_loss)
    print("Test accuracy:", test_accuracy)
if __name__ == "__main__":
    main()
