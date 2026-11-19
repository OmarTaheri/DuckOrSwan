"""Train the Duck vs Swan CNNs (Keras 3, PyTorch backend on the GPU).

Model A = exact lecture architecture, 150x150 (baseline)
Model B = lecture CNN + geometric augmentation + dropout, 150x150
Model C = improved CNN, 224x224: 5 VGG-style blocks (2x Conv3x3 + BatchNorm + ReLU, MaxPool),
          GlobalAveragePooling instead of Flatten, dropout, augmentation with colour jitter /
          random grayscale, learning-rate decay. Model C is the final model on the website.

Run from the project root:  python -I scripts/train.py
"""
import os

os.environ["KERAS_BACKEND"] = "torch"

import gc
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import keras
import matplotlib
import numpy as np
import torch
from keras import layers

torch.backends.cudnn.benchmark = True  # autotune conv kernels for the fixed input sizes
from PIL import Image

from app.augment import GPUAugment

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

DATA = ROOT / "data" / "processed"
FIG = ROOT / "report" / "figures"
MODEL_DIR = ROOT / "model"
FIG.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 32
CLASS_NAMES = ["duck", "swan"]  # 0 = Duck, 1 = Swan
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 12})


# --------------------------------------------------------------------------- #
# Data: images kept as uint8 in RAM, resized + normalised /255 per batch
# --------------------------------------------------------------------------- #
def load_split(split, size):
    xs, ys, paths = [], [], []
    for label, cls in enumerate(CLASS_NAMES):
        for p in sorted((DATA / split / cls).glob("*.jpg")):
            img = Image.open(p).convert("RGB").resize(size)
            xs.append(np.asarray(img, dtype=np.uint8))
            ys.append(label)
            paths.append(p)
    return np.stack(xs), np.array(ys, dtype="float32"), paths


class Batches(keras.utils.PyDataset):
    """prepare_dataset() from the lecture: batch, shuffle (train only), rescale 0-255 -> 0.0-1.0."""

    def __init__(self, x, y, shuffle=False):
        super().__init__()
        self.x, self.y, self.shuffle = x, y, shuffle
        self.idx = np.arange(len(x))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.x) / BATCH_SIZE))

    def __getitem__(self, i):
        j = self.idx[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        return self.x[j].astype("float32") / 255.0, self.y[j]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx)


def predict_probs(model, x):
    return np.concatenate([model.predict_on_batch(x[i:i + 64].astype("float32") / 255.0).ravel()
                           for i in range(0, len(x), 64)])


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def lecture_body():
    return [
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
    ]


def build_model_a():
    model = keras.Sequential(
        [keras.Input(shape=(150, 150, 3))] + lecture_body() + [
            layers.Dense(512, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ], name="model_A_lecture")
    model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
    return model


def build_model_b():
    model = keras.Sequential(
        [keras.Input(shape=(150, 150, 3)),
         GPUAugment(rotation_deg=36, zoom=0.1, translate=0.0, color=False, name="augmentation")]
        + lecture_body() + [
            layers.Dropout(0.5),
            layers.Dense(512, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ], name="model_B_augment_dropout")
    model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(1e-4), metrics=["accuracy"])
    return model


def conv_block(filters):
    return [
        layers.Conv2D(filters, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(filters, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(2, 2),
    ]


def build_model_c():
    blocks = []
    for f in [32, 64, 128, 256, 256]:
        blocks += conv_block(f)
    model = keras.Sequential(
        [keras.Input(shape=(224, 224, 3)), GPUAugment(name="augmentation")] + blocks + [
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.4),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ], name="model_C_improved")
    model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
    return model


def train(model, data, epochs, tag, patience=10, reduce_lr=False):
    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = data
    callbacks = [keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=patience, restore_best_weights=True, verbose=1)]
    if reduce_lr:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1))
    print(f"\nStarting Training {tag} with Early Stopping...", flush=True)
    t0 = time.time()
    history = model.fit(Batches(x_tr, y_tr, shuffle=True), epochs=epochs,
                        validation_data=Batches(x_va, y_va), callbacks=callbacks, verbose=2)
    minutes = (time.time() - t0) / 60
    test_loss, test_acc = model.evaluate(Batches(x_te, y_te), verbose=0)
    print(f"{tag}: test accuracy {test_acc:.4f}, test loss {test_loss:.4f}, {minutes:.1f} min", flush=True)
    h = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    return h, {"test_accuracy": float(test_acc), "test_loss": float(test_loss),
               "epochs_run": len(h["loss"]), "best_epoch": int(np.argmax(h["val_accuracy"]) + 1),
               "best_val_accuracy": float(max(h["val_accuracy"])),
               "train_minutes": round(minutes, 2), "params": int(model.count_params())}


# --------------------------------------------------------------------------- #
# Figures that do not depend on training
# --------------------------------------------------------------------------- #
stats = json.loads((ROOT / "data" / "stats.json").read_text())
counts = {s: stats["splits"][s] for s in ["train", "val", "test"]}


def fig_dataset():
    total = {c: sum(counts[s][c] for s in counts) for c in CLASS_NAMES}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].pie(total.values(), labels=[c.capitalize() for c in total], autopct="%1.1f%%",
                colors=["#636efa", "#ef553b"], startangle=90, textprops={"color": "w", "weight": "bold"})
    axes[0].legend(loc="upper right", bbox_to_anchor=(1.25, 1))
    axes[0].set_title("Dataset Distribution (by class)")
    splits = list(counts)
    bottom = np.zeros(len(splits))
    for c, col in zip(CLASS_NAMES, ["#636efa", "#ef553b"]):
        vals = np.array([counts[s][c] for s in splits])
        axes[1].bar(splits, vals, bottom=bottom, color=col, label=c.capitalize())
        for i, v in enumerate(vals):
            axes[1].text(i, bottom[i] + v / 2, str(v), ha="center", va="center", color="w", weight="bold")
        bottom += vals
    axes[1].set_title("Images per split")
    axes[1].set_ylabel("Number of images")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG / "dataset_distribution.png", dpi=200)
    plt.close(fig)


def fig_taxa():
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, cls, col in zip(axes, CLASS_NAMES, ["#636efa", "#ef553b"]):
        taxa = stats["taxa"][cls]
        names = list(taxa)[::-1]
        ax.barh(names, [taxa[n] for n in names], color=col)
        ax.set_title(f"{cls.capitalize()} images by genus / species")
        ax.set_xlabel("Images")
        for lbl in ax.get_yticklabels():
            lbl.set_fontstyle("italic")
    fig.tight_layout()
    fig.savefig(FIG / "dataset_taxa.png", dpi=200)
    plt.close(fig)


def fig_preprocessing():
    p = DATA / "test" / "swan" / "inat_760381227.jpg"
    original = Image.open(p).convert("RGB")
    resized = original.resize((28, 28))
    gray = resized.convert("L")
    normalised = np.array(gray).astype("float32") / 255.0
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.8))
    ax[0].imshow(original); ax[0].set_title(f"1: Original {original.size[0]}x{original.size[1]}")
    ax[1].imshow(resized); ax[1].set_title("2: Resized 28x28")
    ax[2].imshow(gray, cmap="gray"); ax[2].set_title("3: Grayscale (L)")
    im = ax[3].imshow(normalised, cmap="gray", vmin=0, vmax=1); ax[3].set_title("4: Normalised 0.0-1.0")
    fig.colorbar(im, ax=ax[3], fraction=0.046)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "preprocessing_steps.png", dpi=150)
    plt.close(fig)


def fig_augmentation():
    p = DATA / "test" / "swan" / "inat_760381227.jpg"
    img = np.asarray(Image.open(p).convert("RGB").resize((224, 224)), dtype="float32") / 255.0
    aug = GPUAugment(seed=7)
    out = keras.ops.convert_to_numpy(aug(np.repeat(img[None], 7, 0), training=True))
    fig, ax = plt.subplots(2, 4, figsize=(12, 6.2))
    for i, a in enumerate(ax.ravel()):
        a.imshow(img if i == 0 else out[i - 1])
        a.set_title("Original" if i == 0 else f"Augmented {i}", fontsize=11)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "augmentation_examples.png", dpi=130)
    plt.close(fig)


fig_dataset()
fig_taxa()
fig_preprocessing()
fig_augmentation()

# --------------------------------------------------------------------------- #
# Models A and B at 150x150
# --------------------------------------------------------------------------- #
print("Loading images at 150x150 ...", flush=True)
x_tr, y_tr, _ = load_split("train", (150, 150))
x_va, y_va, _ = load_split("val", (150, 150))
x_te, y_te, _ = load_split("test", (150, 150))
print(f"Training images: {len(x_tr)}\nValidation images: {len(x_va)}\nTest images: {len(x_te)}", flush=True)


def fig_samples():
    idx = np.random.choice(len(x_tr), 9, replace=False)
    fig = plt.figure(figsize=(9, 9))
    for i, j in enumerate(idx):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x_tr[j])
        name = "Swan" if int(y_tr[j]) == 1 else "Duck"
        plt.title(f"Label: {name} ({int(y_tr[j])})")
        plt.axis("off")
    plt.suptitle("Sample Images from Training Set (Resized to 150x150)")
    plt.tight_layout()
    fig.savefig(FIG / "sample_images.png", dpi=110)
    plt.close(fig)


fig_samples()
data150 = ((x_tr, y_tr), (x_va, y_va), (x_te, y_te))

model_a = build_model_a()
model_a.summary()
hist_a, res_a = train(model_a, data150, epochs=30, tag="Model A (lecture)")
model_a.save(MODEL_DIR / "model_A_lecture.keras")

model_b = build_model_b()
model_b.summary()
hist_b, res_b = train(model_b, data150, epochs=60, tag="Model B (augment + dropout)")
model_b.save(MODEL_DIR / "model_B_augment_dropout.keras")

del x_tr, x_va, x_te, data150, model_a, model_b
gc.collect()

# --------------------------------------------------------------------------- #
# Model C at 224x224 (final)
# --------------------------------------------------------------------------- #
print("Loading images at 224x224 ...", flush=True)
x_tr, y_tr, _ = load_split("train", (224, 224))
x_va, y_va, _ = load_split("val", (224, 224))
x_te, y_te, test_paths = load_split("test", (224, 224))
data224 = ((x_tr, y_tr), (x_va, y_va), (x_te, y_te))

model_c = build_model_c()
model_c.summary()
with open(FIG / "model_summary.txt", "w", encoding="utf-8") as f:
    model_c.summary(print_fn=lambda s, **k: f.write(s + "\n"))
hist_c, res_c = train(model_c, data224, epochs=100, tag="Model C (improved, 224px)",
                      patience=15, reduce_lr=True)
model_c.save(MODEL_DIR / "duck_swan_model.keras")
print("Model saved", flush=True)

# --------------------------------------------------------------------------- #
# Evaluation of the final model
# --------------------------------------------------------------------------- #
probs = predict_probs(model_c, x_te)
pred = (probs > 0.5).astype(int)
y_int = y_te.astype(int)
cm = confusion_matrix(y_int, pred)
report = classification_report(y_int, pred, target_names=["Duck", "Swan"], output_dict=True)
print(classification_report(y_int, pred, target_names=["Duck", "Swan"], digits=4))

# per-taxon accuracy on the test set (which species are hardest?)
manifest_path = ROOT / "data" / "raw" / "inat_manifest.json"
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
per_taxon = {}
for p, yt, yp in zip(test_paths, y_int, pred):
    t = manifest.get(p.name, "Cygnus spp." if yt == 1 else "Anatidae spp.")
    d = per_taxon.setdefault(t, {"n": 0, "correct": 0, "class": CLASS_NAMES[yt]})
    d["n"] += 1
    d["correct"] += int(yt == yp)
for d in per_taxon.values():
    d["accuracy"] = d["correct"] / d["n"]


def fig_curves():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for h, name, col in [(hist_a, "A (lecture)", "tab:orange"), (hist_b, "B (+aug, dropout)", "tab:blue"),
                         (hist_c, "C (improved, 224px)", "tab:green")]:
        ax[0].plot(h["accuracy"], "--", color=col, alpha=0.6, label=f"{name} train")
        ax[0].plot(h["val_accuracy"], "-", color=col, label=f"{name} val")
        ax[1].plot(h["loss"], "--", color=col, alpha=0.6, label=f"{name} train")
        ax[1].plot(h["val_loss"], "-", color=col, label=f"{name} val")
    ax[0].set_title("Accuracy"); ax[1].set_title("Loss")
    ax[1].set_ylim(0, 1.5)
    for a in ax:
        a.set_xlabel("Epoch"); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "training_curves.png", dpi=200)
    plt.close(fig)


def fig_confusion():
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Duck", "Swan"]); ax.set_yticks([0, 1], ["Duck", "Swan"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix - Model C (test set)")
    fig.tight_layout()
    fig.savefig(FIG / "confusion_matrix.png", dpi=200)
    plt.close(fig)


def fig_mistakes():
    rng = np.random.default_rng(3)
    panels = []
    for true_label in [0, 1]:
        wrong = np.where((pred != y_int) & (y_int == true_label))[0]
        for j in rng.choice(wrong, min(4, len(wrong)), replace=False):
            conf = probs[j] if pred[j] == 1 else 1 - probs[j]
            panels.append((x_te[j], f"True: {CLASS_NAMES[true_label].capitalize()} | "
                                    f"Pred: {CLASS_NAMES[pred[j]].capitalize()} ({conf:.0%})"))
    fig = plt.figure(figsize=(12, 6.6))
    for i, (img, title) in enumerate(panels):
        plt.subplot(2, 4, i + 1)
        plt.imshow(img)
        plt.title(title, fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    fig.savefig(FIG / "misclassified.png", dpi=110)
    plt.close(fig)


fig_curves()
fig_confusion()
fig_mistakes()

results = {
    "counts": counts,
    "model_A": res_a, "model_B": res_b, "model_C": res_c,
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "per_taxon_test": per_taxon,
    "history_A": hist_a, "history_B": hist_b, "history_C": hist_c,
}
(MODEL_DIR / "results.json").write_text(json.dumps(results, indent=2))
print(json.dumps({"A": res_a, "B": res_b, "C": res_c}, indent=2))
