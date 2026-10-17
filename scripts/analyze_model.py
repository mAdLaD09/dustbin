from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.common.config import load_config


def main():
    cfg = load_config("config.yaml")
    size = cfg["model"]["image_size"]
    root = Path("datasets/processed")

    class_names = sorted(path.name for path in root.iterdir() if path.is_dir())
    name_to_index = {name: index for index, name in enumerate(class_names)}
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    paths, labels = [], []
    for name in class_names:
        for path in sorted((root / name).rglob("*")):
            if path.is_file() and path.suffix.lower() in extensions:
                paths.append(str(path))
                labels.append(name_to_index[name])
    _, validation_paths, _, validation_labels = train_test_split(
        paths, labels, test_size=0.3, random_state=42, stratify=labels
    )

    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, (size, size))
        return image, label

    validation = tf.data.Dataset.from_tensor_slices(
        (validation_paths, validation_labels)
    ).map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(128).prefetch(tf.data.AUTOTUNE)
    model = tf.keras.models.load_model("models/best_model.keras")

    print("Generating predictions...")
    y_true = []
    y_pred = []
    for images, labels in validation:
        probabilities = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(probabilities, axis=1))

    print("\nClassification report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    matrix = confusion_matrix(y_true, y_pred)
    print("Confusion matrix (rows = true class, columns = predicted class):")
    print("Classes:", class_names)
    print(matrix)
    output = Path("models/confusion_matrix.csv")
    np.savetxt(output, matrix, delimiter=",", fmt="%d")
    print(f"Saved confusion matrix to {output}")


if __name__ == "__main__":
    main()
