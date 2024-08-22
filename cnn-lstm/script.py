import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

HEIGHT = 128
WIDTH = 128
CHANNEL = 3
TIME_STEPS = 10
NUM_SEQ = 1_000

data = np.zeros((NUM_SEQ, TIME_STEPS, HEIGHT, WIDTH, CHANNEL))

dataset_path = '/dataset/'

for i in range(NUM_SEQ):
    seq_path = os.path.join(dataset_path, f"seq_{i + 1}")
    for j in range(TIME_STEPS):
        img_path = os.path.join(seq_path, f'img_{j + 1}.jpg')
        img = img_to_array(load_img(img_path))
        data[i, j] = img


data /= 255.

np.save('cnn-lstm/paddy_time_distributed_data.npy', data)

