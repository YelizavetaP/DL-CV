#------------------- Identification of 5 binary digits with a raw NumPy neural network --------------------------

'''
Homework 6 - Variant 5 (Level I).

Identify binary images of 5 digits (0..4) given by a raster matrix, using the
neural network from the lecture (Neural_Networks_numpy_2.py) built "from scratch"
- raw matrix operations only, no ML frameworks. The architecture and training
algorithm are justified in README.md; workability + efficiency are proven by the
train/test curves, confusion matrix, sample predictions and a gallery of the
misclassified examples.

Data:
    MNIST (the classic handwritten-digit dataset). Each 28x28 grayscale image is
    binarized by thresholding into a strict {0, 1} raster matrix. We keep all
    available samples of digits 0..4 and use the dataset's own train/test split,
    so high TEST accuracy proves the network learned to recognize the digits
    rather than memorize fixed pictures.

Architecture (lecture baseline):
    1st level: input  layer (1, 784)   - one neuron per pixel of the 28x28 raster
    2nd layer: hidden layer (1, 5)     - sigmoid
    3rd layer: output layer (1, 5)     - sigmoid, one neuron per digit class
    no bias terms.

Training (lecture baseline):
    loss        - mean square error (MSE)
    algorithm   - per-sample gradient descent (one weight update per image), rate alpha
    init        - plain random normal (unscaled)

Package                      Version
---------------------------- -----------
numpy                        2.x
matplotlib                   3.x
'''

import urllib.request
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ----- configuration (generic over any digit subset) -----
DIGITS = [0, 1, 2, 3, 4]      # the 5 classes to identify
THRESHOLD = 50               # grayscale -> binary cutoff (0..255)

# ----- hyperparameters (lecture baseline) -----
N_HIDDEN = 5      # hidden-layer size, as in the lecture
ALPHA = 0.1       # learning rate, as in the lecture
EPOCHS = 15

RNG = np.random.default_rng(42)  # single source of randomness -> reproducible run

# every plot is also saved here as a PNG
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


#----------------------------------- input data: MNIST DataSet ------------------------------------

# TensorFlow has no wheel for Python 3.14, so `from tensorflow.keras.datasets import
# mnist` cannot run here. But keras's mnist.load_data() only downloads this official
# npz and np.load()s it - so we replicate it exactly and get the identical
# (x_train, y_train), (x_test, y_test) tuples, with no heavy dependency.
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MNIST_CACHE = Path.home() / ".keras" / "datasets" / "mnist.npz"


def load_data():
    '''
    Drop-in equivalent of keras.datasets.mnist.load_data().
    Downloads the ~11 MB MNIST npz once into the cache, then loads from disk.

    :return: (x_train, y_train), (x_test, y_test) - uint8 images (N, 28, 28), labels 0..9
    '''
    if not MNIST_CACHE.exists():
        MNIST_CACHE.parent.mkdir(parents=True, exist_ok=True)
        print("downloading mnist.npz (~11 MB) ...")
        urllib.request.urlretrieve(MNIST_URL, MNIST_CACHE)
    with np.load(MNIST_CACHE) as f:
        return (f["x_train"], f["y_train"]), (f["x_test"], f["y_test"])


def make_split(images, labels):
    '''
    Keep only DIGITS, binarize, and flatten 28x28 -> 784 (uses all available samples).

    :param images: np.array (N, 28, 28) uint8
    :param labels: np.array (N,) digit labels 0..9
    :return: X - (N, 784) float {0,1} matrix; y - (N,) class indices 0..K-1
    '''
    binary = (images > THRESHOLD).astype(np.float64)
    X_parts, y_parts = [], []
    for class_idx, digit in enumerate(DIGITS):
        cls_images = binary[labels == digit]
        X_parts.append(cls_images.reshape(len(cls_images), -1))
        y_parts.append(np.full(len(cls_images), class_idx))
    return np.vstack(X_parts), np.concatenate(y_parts)


#----------------------------------- network construction (lecture baseline) ------------------------------------

def one_hot(y, n_classes):
    '''Integer labels -> one-hot rows.'''
    out = np.zeros((y.size, n_classes))
    out[np.arange(y.size), y] = 1.0
    return out


def sigmoid(x):
    '''Activation used in both layers (lecture baseline).'''
    return 1.0 / (1.0 + np.exp(-x))


def generate_wt(n_in, n_out):
    '''Random weight initialization (standard normal, unscaled) - as in the lecture.'''
    return RNG.standard_normal((n_in, n_out))


def f_forward(x, w1, w2):
    '''Forward pass: input -> sigmoid hidden -> sigmoid output. No bias terms.'''
    a1 = sigmoid(x.dot(w1))     # hidden layer
    a2 = sigmoid(a1.dot(w2))    # output layer
    return a2


def loss(out, Y):
    '''Mean-square error (lecture baseline loss).'''
    return np.sum(np.square(out - Y)) / len(Y)


def back_prop(x, y, w1, w2, alpha):
    '''Back-propagation for ONE sample (MSE + sigmoid), exactly as the lecture.'''
    a1 = sigmoid(x.dot(w1))
    a2 = sigmoid(a1.dot(w2))
    d2 = (a2 - y)
    d1 = np.multiply((w2.dot(d2.T)).T, np.multiply(a1, 1 - a1))
    w1_adj = x.T.dot(d1)
    w2_adj = a1.T.dot(d2)
    w1 = w1 - alpha * w1_adj
    w2 = w2 - alpha * w2_adj
    return w1, w2


def accuracy(probs, y_int):
    '''Fraction of correctly identified samples (arg-max of the output).'''
    return float(np.mean(probs.argmax(axis=1) == y_int))


#------- training of the network with per-sample gradient descent -----------------------------

def train(X_train, y_train, X_test, y_test, w1, w2, alpha=ALPHA, epochs=EPOCHS):
    '''
    Train per-sample (one weight update per image), recording loss and
    train/test accuracy per epoch. The sample order is shuffled each epoch
    because make_split stores the data class-by-class.

    :return: history dict {loss, train_acc, test_acc}; trained w1, w2
    '''
    n_classes = w2.shape[1]
    Y_train = one_hot(y_train, n_classes)
    Y_test = one_hot(y_test, n_classes)
    history = {"loss": [], "train_acc": [], "test_acc": []}

    n = X_train.shape[0]
    for epoch in range(epochs):
        order = RNG.permutation(n)               # shuffle sample order each epoch
        for i in order:
            xi = X_train[i:i + 1]                 # (1, 784) one image
            yi = Y_train[i:i + 1]                 # (1, 5) one label
            w1, w2 = back_prop(xi, yi, w1, w2, alpha)

        train_probs = f_forward(X_train, w1, w2)
        test_probs = f_forward(X_test, w1, w2)
        history["loss"].append(loss(train_probs, Y_train))
        history["train_acc"].append(accuracy(train_probs, y_train))
        history["test_acc"].append(accuracy(test_probs, y_test))
        print("epoch %3d  loss %.4f  train_acc %.3f  test_acc %.3f" % (
            epoch + 1, history["loss"][-1],
            history["train_acc"][-1], history["test_acc"][-1]))
    return history, w1, w2


#------- visualization / analysis of results --------------------------------------------------

def show_samples(X, y, n_show=8):
    '''Show a few binarized examples per class.'''
    fig, axes = plt.subplots(len(DIGITS), n_show, figsize=(n_show, len(DIGITS)))
    for row, digit in enumerate(DIGITS):
        idxs = np.where(y == row)[0][:n_show]
        for col, idx in enumerate(idxs):
            ax = axes[row, col]
            ax.imshow(X[idx].reshape(28, 28), cmap="gray_r")
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(str(digit), rotation=0, labelpad=12, fontsize=14, va="center")
    fig.suptitle("Binarized MNIST samples (28x28 raster), one row per class")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_samples.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_history(history):
    '''Training loss (MSE) + train/test accuracy curves.'''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["loss"], color="crimson")
    ax1.set_title("Training loss (MSE)")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.grid(alpha=0.3)
    ax2.plot(history["train_acc"], label="train")
    ax2.plot(history["test_acc"], label="test")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy")
    ax2.set_ylim(0, 1.02); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_training_curves.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion(X_test, y_test, w1, w2):
    '''Confusion matrix + per-class accuracy on the held-out test set.'''
    probs = f_forward(X_test, w1, w2)
    pred = probs.argmax(axis=1)
    n_classes = len(DIGITS)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true_c, pred_c in zip(y_test, pred):
        cm[true_c, pred_c] += 1

    labels = [str(d) for d in DIGITS]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n_classes)); ax.set_xticklabels(labels)
    ax.set_yticks(range(n_classes)); ax.set_yticklabels(labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Confusion matrix (test set)")
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()

    for c, digit in enumerate(DIGITS):
        print("%d: %.3f" % (digit, cm[c, c] / cm[c].sum()))
    return pred


def plot_predictions(X_test, y_test, pred):
    '''Predictions on random unseen test images (green = correct, red = wrong).'''
    sample = RNG.choice(len(X_test), size=15, replace=False)
    fig, axes = plt.subplots(3, 5, figsize=(9, 5.5))
    for ax, idx in zip(axes.ravel(), sample):
        ax.imshow(X_test[idx].reshape(28, 28), cmap="gray_r")
        ok = pred[idx] == y_test[idx]
        ax.set_title("pred %d / true %d" % (DIGITS[pred[idx]], DIGITS[y_test[idx]]),
                     color="green" if ok else "red", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_sample_predictions.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_misclassified(X_test, y_test, pred, max_show=15):
    '''Show the test images the network got wrong (pred != true).'''
    wrong = np.where(pred != y_test)[0]
    print("misclassified: %d of %d test images" % (len(wrong), len(y_test)))
    if len(wrong) == 0:
        print("No misclassified test images.")
        return
    sample = wrong[:max_show]
    cols = 5
    rows = int(np.ceil(len(sample) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(9, 1.9 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, idx in zip(axes, sample):
        ax.imshow(X_test[idx].reshape(28, 28), cmap="gray_r")
        ax.set_title("pred %d / true %d" % (DIGITS[pred[idx]], DIGITS[y_test[idx]]),
                     color="red", fontsize=10)
    fig.suptitle("Misclassified test images (%d of %d wrong)" % (len(wrong), len(y_test)))
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_misclassified.png", dpi=150, bbox_inches="tight")
    plt.show()


# ------------------------------- MAIN CALLS BLOCK -------------------------------------------------------
if __name__ == '__main__':

    # ------------------- input data -----------------------------------
    (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = load_data()
    X_train, y_train = make_split(x_train_raw, y_train_raw)
    X_test, y_test = make_split(x_test_raw, y_test_raw)
    print("train: X", X_train.shape, "y", y_train.shape)
    print("test:  X", X_test.shape, "y", y_test.shape)
    print("pixel values present:", np.unique(X_train), "(strictly binary)")
    show_samples(X_train, y_train)

    # ------- initialization of the weights for the 2 layers -----------
    w1 = generate_wt(X_train.shape[1], N_HIDDEN)   # (784, 5)
    w2 = generate_wt(N_HIDDEN, len(DIGITS))        # (5, 5)

    # ------- training of the network ----------------------------------
    print("training the network ...")
    history, w1, w2 = train(X_train, y_train, X_test, y_test, w1, w2)
    print("\nFinal test accuracy: %.3f" % history["test_acc"][-1])

    # ------- proof of workability and efficiency ----------------------
    plot_history(history)
    pred = plot_confusion(X_test, y_test, w1, w2)
    plot_predictions(X_test, y_test, pred)
    plot_misclassified(X_test, y_test, pred)
