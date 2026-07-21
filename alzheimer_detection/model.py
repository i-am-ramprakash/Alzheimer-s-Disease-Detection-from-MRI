"""Small CNN suitable for an educational four-class MRI experiment."""

from typing import Tuple

from .constants import CLASS_KEYS


def build_augmentation():
    import tensorflow as tf

    # Vertical flips are intentionally excluded because they are anatomically implausible.
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.03),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )


def build_model(
    image_size: Tuple[int, int] = (128, 128), learning_rate: float = 1e-3
):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is not installed. Run: python -m pip install -r requirements.txt"
        ) from exc

    inputs = tf.keras.Input((*image_size, 1), name="mri_image")
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = build_augmentation()(x)

    for filters, dropout in ((32, 0.10), (64, 0.15), (128, 0.20)):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.SeparableConv2D(192, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(
        len(CLASS_KEYS), activation="softmax", name="dementia_stage"
    )(x)

    model = tf.keras.Model(inputs, outputs, name="educational_alzheimer_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def build_transfer_model(
    image_size: Tuple[int, int] = (96, 96), learning_rate: float = 3e-4
):
    """Build a frozen ImageNet MobileNetV2 feature extractor and classifier."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is not installed. Run: python -m pip install -r requirements.txt"
        ) from exc

    if min(image_size) < 96:
        raise ValueError("MobileNetV2 training requires an image size of at least 96.")

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*image_size, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.Input((*image_size, 3), name="mri_image")
    x = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.02),
            tf.keras.layers.RandomZoom(0.05),
            tf.keras.layers.RandomContrast(0.08),
        ],
        name="transfer_augmentation",
    )(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(
        len(CLASS_KEYS), activation="softmax", name="dementia_stage"
    )(x)
    model = tf.keras.Model(inputs, outputs, name="alzheimer_mobilenetv2")
    compile_transfer_model(model, learning_rate)
    return model, base_model


def compile_transfer_model(model, learning_rate: float) -> None:
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def enable_transfer_fine_tuning(base_model, trainable_layers: int = 30) -> None:
    """Unfreeze only the top feature layers while keeping batch norm fixed."""
    import tensorflow as tf

    base_model.trainable = True
    cutoff = max(0, len(base_model.layers) - trainable_layers)
    for index, layer in enumerate(base_model.layers):
        layer.trainable = index >= cutoff and not isinstance(
            layer, tf.keras.layers.BatchNormalization
        )
