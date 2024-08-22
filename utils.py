import numpy as np
import pandas as pd
import tensorflow
from matplotlib import pyplot as plt
import seaborn as sns

bs = 32


def norm_data(x, stats: pd.DataFrame):
    return (x - stats['mean']) / stats['std']


def format_output(data):
    fan = np.array(data.pop('Kipas'))
    humid = np.array(data.pop('Humidifier'))
    led = np.array(data.pop('LED'))
    return fan, humid, led


def print_model(model, tf: tensorflow):
    return tf.keras.utils.plot_model(model, show_shapes=True)


def show_image_samples(generated_data, classes):
    class_indices = generated_data.class_indices
    class_ = list(class_indices.keys())
    imgs, labels = generated_data[0][0], generated_data[0][1]
    plt.figure(figsize=(20, 20))
    length = len(labels)
    r = length if length < 25 else 25

    for i in range(r):
        plt.subplot(5, 5, i + 1)
        plt.imshow(imgs[i], cmap='gray')
        idx = int(labels[i])
        c_name = classes[idx]
        plt.title(c_name, color='Blue', fontsize=16)
        plt.axis('off')
    plt.show()


def plot_confusion_matrix(cm, title, labels=[0, 1], ax=None):
    if ax is None:
        ax = plt.gca()
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', ax=ax)
    ax.set_title(title, pad=20, fontsize=20, fontweight="bold")
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
