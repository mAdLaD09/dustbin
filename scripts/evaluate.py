from pathlib import Path

import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.common.config import load_config


def main():
    print("Indexing validation images...")
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

    print("Loading saved model...")
    model = tf.keras.models.load_model("models/best_model.keras")
    print("Evaluating model...")
    loss, accuracy = model.evaluate(validation, verbose=1)

    print(f"Validation loss: {loss:.4f}")
    print(f"Validation accuracy: {accuracy:.4%}")


if __name__ == "__main__":
    main()
