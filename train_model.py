"""
AI Learning Support Platform — train_model.py
Complete training pipeline:
  Phase 1: CustomCNN on MNIST
  Phase 2: ResNet on MNIST + augmentation
  Phase 3: MobileNet transfer learning + fine-tuning
  Exports: models/ directory with .h5 files
"""

import os
import logging
import numpy as np
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TF_AVAILABLE = True

# Check TensorFlow
try:
    import tensorflow as tf
    print("TensorFlow OK:", tf.__version__)
except Exception as e:
    TF_AVAILABLE = False
    logger.error(f"TensorFlow import failed: {e}")

# Check Keras
try:
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.callbacks import (
        ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
    )
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except Exception as e:
    logger.error(f"Keras import failed: {e}")
    TF_AVAILABLE = False

# Check OpenCV
try:
    import cv2
except Exception as e:
    logger.error(f"OpenCV import failed: {e}")

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)


# ─── Data Loading & Preprocessing ────────────────────────────────────────────

def load_mnist():
    """Load and preprocess MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    logger.info(f"MNIST loaded: train={x_train.shape}, test={x_test.shape}")

    # Normalize
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    # Add channel dim for CNN
    x_train_cnn = x_train.reshape(-1, 28, 28, 1)
    x_test_cnn = x_test.reshape(-1, 28, 28, 1)

    return (x_train_cnn, y_train), (x_test_cnn, y_test)


def load_mnist_rgb_upscaled(target_size=96):
    """Load MNIST upscaled to RGB for MobileNet transfer learning."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    def upscale_to_rgb(images, size):
        out = []
        for img in images:
            resized = cv2.resize(img, (size, size))
            rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            out.append(rgb)
        return np.array(out, dtype=np.float32) / 255.0

    logger.info("Upscaling MNIST to RGB 96x96 for MobileNet...")
    x_train_rgb = upscale_to_rgb(x_train, target_size)
    x_test_rgb = upscale_to_rgb(x_test, target_size)
    logger.info("Upscaling complete.")
    return (x_train_rgb, y_train), (x_test_rgb, y_test)


def get_augmentation_generator():
    """Strong augmentation for robust handwriting recognition."""
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        shear_range=0.1,
        fill_mode="nearest"
    )


def get_callbacks(model_name: str, patience: int = 10):
    return [
        ModelCheckpoint(
            f"models/{model_name}.h5",
            save_best_only=True,
            monitor="val_accuracy",
            verbose=1
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(log_dir=f"logs/{model_name}", histogram_freq=1)
    ]


# ─── Phase 1: CustomCNN ───────────────────────────────────────────────────────

def train_custom_cnn(epochs=30, batch_size=128):
    from model import build_custom_cnn
    logger.info("=" * 60)
    logger.info("PHASE 1: Training CustomCNN")
    logger.info("=" * 60)

    (x_train, y_train), (x_test, y_test) = load_mnist()
    model = build_custom_cnn()
    model.summary()

    datagen = get_augmentation_generator()
    datagen.fit(x_train)

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=get_callbacks("cnn_digit"),
        steps_per_epoch=len(x_train) // batch_size
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    logger.info(f"CustomCNN Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    return history, acc


# ─── Phase 2: ResNet ──────────────────────────────────────────────────────────

def train_resnet(epochs=40, batch_size=128):
    from model import build_resnet_digit
    logger.info("=" * 60)
    logger.info("PHASE 2: Training ResNetDigit")
    logger.info("=" * 60)

    (x_train, y_train), (x_test, y_test) = load_mnist()
    model = build_resnet_digit()
    model.summary()

    datagen = get_augmentation_generator()
    datagen.fit(x_train)

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=get_callbacks("resnet_digit", patience=15),
        steps_per_epoch=len(x_train) // batch_size
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    logger.info(f"ResNetDigit Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    return history, acc


# ─── Phase 3: MobileNet Transfer Learning ────────────────────────────────────

def train_mobilenet(epochs_frozen=15, epochs_finetune=20, batch_size=64):
    from model import build_mobilenet_digit, unfreeze_mobilenet
    logger.info("=" * 60)
    logger.info("PHASE 3: Training MobileNetV2 (Transfer Learning)")
    logger.info("=" * 60)

    (x_train, y_train), (x_test, y_test) = load_mnist_rgb_upscaled(96)
    model = build_mobilenet_digit()
    model.summary()

    # Stage 1: Train with frozen base
    logger.info("Stage 1: Training with frozen MobileNetV2 base...")
    history1 = model.fit(
        x_train, y_train,
        epochs=epochs_frozen,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        callbacks=get_callbacks("mobilenet_digit_stage1"),
    )

    # Stage 2: Fine-tune top layers
    logger.info("Stage 2: Fine-tuning unfrozen top layers...")
    model = unfreeze_mobilenet(model, fine_tune_from=100)
    history2 = model.fit(
        x_train, y_train,
        epochs=epochs_finetune,
        batch_size=batch_size // 2,
        validation_data=(x_test, y_test),
        callbacks=get_callbacks("mobilenet_digit"),
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    logger.info(f"MobileNetDigit Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    return history2, acc


# ─── Evaluation & Report ──────────────────────────────────────────────────────

def evaluate_all():
    """Run inference on all saved models and compare."""
    (_, _), (x_test, y_test) = load_mnist()

    results = {}
    model_files = {
        "cnn": "models/cnn_digit.h5",
        "resnet": "models/resnet_digit.h5",
    }

    for name, path in model_files.items():
        if os.path.exists(path):
            model = tf.keras.models.load_model(path)
            loss, acc = model.evaluate(x_test, y_test, verbose=0)
            results[name] = {"accuracy": acc, "loss": loss}
            logger.info(f"{name}: Accuracy={acc:.4f}, Loss={loss:.4f}")
        else:
            logger.warning(f"Model not found: {path}")

    # MobileNet uses different input
    if os.path.exists("models/mobilenet_digit.h5"):
        (_, _), (x_test_rgb, y_test_rgb) = load_mnist_rgb_upscaled(96)
        model = tf.keras.models.load_model("models/mobilenet_digit.h5")
        loss, acc = model.evaluate(x_test_rgb, y_test_rgb, verbose=0)
        results["mobilenet"] = {"accuracy": acc, "loss": loss}
        logger.info(f"mobilenet: Accuracy={acc:.4f}, Loss={loss:.4f}")

    return results


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train digit recognition models")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["cnn", "resnet", "mobilenet", "all", "quick", "eval"],
                        help="Which phase to run")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="Fast training mode (20 epochs, ~98% accuracy)")
    args = parser.parse_args()

    if not TF_AVAILABLE:
        logger.error("TensorFlow not installed. Cannot train.")
        return

    # Quick mode: fewer epochs, still >98% accuracy
    if args.quick or args.phase == "quick":
        logger.info("🚀 QUICK MODE: Fast training (20 epochs CNN/ResNet, 10+10 MobileNet)")
        train_custom_cnn(epochs=args.epochs or 20)
        train_resnet(epochs=args.epochs or 20)
        train_mobilenet(
            epochs_frozen=args.epochs or 10,
            epochs_finetune=args.epochs or 10
        )
        evaluate_all()
        return

    # Full mode: more epochs, 99%+ accuracy
    if args.phase in ("cnn", "all"):
        train_custom_cnn(epochs=args.epochs or 30)

    if args.phase in ("resnet", "all"):
        train_resnet(epochs=args.epochs or 40)

    if args.phase in ("mobilenet", "all"):
        train_mobilenet(
            epochs_frozen=args.epochs or 15,
            epochs_finetune=args.epochs or 20
        )

    if args.phase == "eval" or args.phase == "all":
        evaluate_all()


if __name__ == "__main__":
    main()