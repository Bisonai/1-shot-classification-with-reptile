from PIL import Image
from pathlib import Path
import re

from tensorflow.keras.utils import to_categorical
import numpy as np

def parse_train_data(path,
                normalization=True):

    path_train = Path(path + "/train").glob('**/*')

    X = []
    y = []
    for path_file in path_train:
        if path_file.is_file() and str(path_file).endswith(".png"):

            img = Image.open(str(path_file)).resize((28, 28)).convert('L')
            img = np.array(img).astype('float32').reshape(28, 28, 1)
            X.append(img)

            y.append(int(re.search('class-(.*)-num', str(path_file)).group(1)))

    X = np.array(X)
    y = to_categorical(np.array(y),num_classes=3)

    if  normalization == True:
        min = -1
        max = 1
        scale = (max - min) / (X.max() - X.min())
        X = scale * X + min - X.min() * scale

    return X, y

def parse_predict_data(path,
                    normalization=True):

    paths = Path(path + "/predict").glob('**/*')

    X = []
    for path_file in paths:
        if path_file.is_file() and str(path_file).endswith(".png"):

            img = Image.open(str(path_file)).resize((28, 28)).convert('L')
            img = np.array(img).astype('float32').reshape(28, 28, 1)
            X.append(img)

    X = np.array(X)

    if  normalization == True:
        min = -1
        max = 1
        scale = (max - min) / (X.max() - X.min())
        X = scale * X + min - X.min() * scale

    return X