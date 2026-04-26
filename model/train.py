"""
CropGuard NE — Plant Disease Detection Model Training
======================================================
Uses MobileNetV2 fine-tuned on PlantVillage dataset (~87K images, 38 classes).

Setup:
    1. Download dataset from:
       kaggle datasets download abdallahalidev/plantvillage-dataset
       unzip plantvillage-dataset.zip -d data/
    2. pip install tensorflow pillow numpy scikit-learn matplotlib seaborn
    3. python model/train.py

Expected output:
    - models/plantdisease_mobilenetv2.h5   (full model)
    - models/class_indices.json            (label mapping)
    - models/training_history.json         (metrics)
    - plots/                               (accuracy/loss curves)
"""

import os, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR       = "data/plantvillage dataset/color"   # path after unzip
MODEL_DIR      = "models"
PLOT_DIR       = "plots"
IMG_SIZE       = (224, 224)
BATCH_SIZE     = 32
EPOCHS_FROZEN  = 10    # train only head (fast convergence)
EPOCHS_FINE    = 15    # unfreeze top layers (refinement)
LEARNING_RATE  = 1e-4
FINE_LR        = 1e-5
VALIDATION_SPLIT = 0.2
SEED           = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ─── DATA GENERATORS ─────────────────────────────────────────────────────────
print("\n[1/6] Setting up data generators...")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest",
    validation_split=VALIDATION_SPLIT,
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=VALIDATION_SPLIT,
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    seed=SEED,
    shuffle=True,
)

val_gen = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    seed=SEED,
    shuffle=False,
)

NUM_CLASSES = len(train_gen.class_indices)
print(f"    Classes   : {NUM_CLASSES}")
print(f"    Train imgs: {train_gen.samples}")
print(f"    Val imgs  : {val_gen.samples}")

# Save class index mapping
class_indices = {v: k for k, v in train_gen.class_indices.items()}
with open(f"{MODEL_DIR}/class_indices.json", "w") as f:
    json.dump(class_indices, f, indent=2)
print(f"    Saved class_indices.json ({NUM_CLASSES} classes)")


# ─── MODEL ARCHITECTURE ───────────────────────────────────────────────────────
print("\n[2/6] Building MobileNetV2 model...")

base_model = MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False   # freeze base initially

inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation="relu",
                 kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation="relu",
                 kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs)
model.summary()


# ─── PHASE 1: HEAD TRAINING ───────────────────────────────────────────────────
print("\n[3/6] Phase 1 — training classification head (base frozen)...")

model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")],
)

cb_phase1 = [
    callbacks.EarlyStopping(monitor="val_accuracy", patience=5,
                            restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                patience=3, min_lr=1e-7, verbose=1),
    callbacks.ModelCheckpoint(f"{MODEL_DIR}/best_phase1.h5",
                              monitor="val_accuracy", save_best_only=True, verbose=1),
]

t0 = time.time()
hist1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FROZEN,
    callbacks=cb_phase1,
    verbose=1,
)
print(f"    Phase 1 done in {(time.time()-t0)/60:.1f} min")


# ─── PHASE 2: FINE-TUNING ─────────────────────────────────────────────────────
print("\n[4/6] Phase 2 — fine-tuning top layers of MobileNetV2...")

# Unfreeze top 50 layers of the base model
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

trainable_count = sum(1 for l in model.layers if l.trainable)
print(f"    Trainable layers: {trainable_count}")

model.compile(
    optimizer=optimizers.Adam(learning_rate=FINE_LR),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")],
)

cb_phase2 = [
    callbacks.EarlyStopping(monitor="val_accuracy", patience=7,
                            restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                patience=4, min_lr=1e-8, verbose=1),
    callbacks.ModelCheckpoint(f"{MODEL_DIR}/best_phase2.h5",
                              monitor="val_accuracy", save_best_only=True, verbose=1),
    callbacks.CSVLogger(f"{MODEL_DIR}/training_log.csv"),
]

t0 = time.time()
hist2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINE,
    callbacks=cb_phase2,
    verbose=1,
)
print(f"    Phase 2 done in {(time.time()-t0)/60:.1f} min")


# ─── SAVE FINAL MODEL ─────────────────────────────────────────────────────────
print("\n[5/6] Saving final model...")

model.save(f"{MODEL_DIR}/plantdisease_mobilenetv2.h5")

# Merge history from both phases
history = {
    "accuracy":     hist1.history["accuracy"]     + hist2.history["accuracy"],
    "val_accuracy": hist1.history["val_accuracy"] + hist2.history["val_accuracy"],
    "loss":         hist1.history["loss"]         + hist2.history["loss"],
    "val_loss":     hist1.history["val_loss"]     + hist2.history["val_loss"],
}
with open(f"{MODEL_DIR}/training_history.json", "w") as f:
    json.dump(history, f)

best_val_acc = max(history["val_accuracy"])
print(f"    Best val accuracy: {best_val_acc*100:.2f}%")
print(f"    Model saved  → {MODEL_DIR}/plantdisease_mobilenetv2.h5")


# ─── PLOTS ────────────────────────────────────────────────────────────────────
print("\n[6/6] Generating training plots...")

epochs_total = len(history["accuracy"])
phase1_end = len(hist1.history["accuracy"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("CropGuard NE — MobileNetV2 Training", fontsize=14, fontweight="bold")

for ax, metric, title in zip(axes,
    [("accuracy", "val_accuracy"), ("loss", "val_loss")],
    ["Model Accuracy", "Model Loss"]):
    train_key, val_key = metric
    ax.plot(history[train_key], label="Train", color="#1D9E75", linewidth=2)
    ax.plot(history[val_key],   label="Validation", color="#D85A30", linewidth=2)
    ax.axvline(x=phase1_end - 0.5, color="gray", linestyle="--", alpha=0.6, label="Fine-tune start")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/training_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# Confusion matrix on validation set
print("    Generating confusion matrix (this may take a minute)...")
val_gen.reset()
y_pred_probs = model.predict(val_gen, verbose=1)
y_pred  = np.argmax(y_pred_probs, axis=1)
y_true  = val_gen.classes
labels  = [class_indices[i] for i in range(NUM_CLASSES)]

# Short label version for readability
short_labels = [l.split("___")[-1].replace("_", " ")[:20] for l in labels]

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(20, 18))
sns.heatmap(cm, annot=False, fmt="d", cmap="Greens",
            xticklabels=short_labels, yticklabels=short_labels, ax=ax)
ax.set_title("Confusion Matrix — Validation Set", fontsize=14, fontweight="bold")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
plt.xticks(rotation=90, fontsize=7)
plt.yticks(rotation=0,  fontsize=7)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.close()

# Classification report
report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
with open(f"{MODEL_DIR}/classification_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("\n" + "="*60)
print("  TRAINING COMPLETE")
print("="*60)
print(f"  Best Val Accuracy : {best_val_acc*100:.2f}%")
print(f"  Model saved       : {MODEL_DIR}/plantdisease_mobilenetv2.h5")
print(f"  Class map         : {MODEL_DIR}/class_indices.json")
print(f"  Plots             : {PLOT_DIR}/")
print("="*60)
