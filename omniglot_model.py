from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import Flatten, Dense, Conv2D, BatchNormalization, ReLU

def omniglot(input_shape, num_classes):
    model = Sequential([
      Conv2D(24, (3,3), strides=2, padding='same', input_shape=input_shape, use_bias=False),
      BatchNormalization(axis=-1,),
      ReLU(),
      Conv2D(24, (3,3), strides=2, padding='same', use_bias=False),
      BatchNormalization(axis=-1),
      ReLU(),
      Conv2D(24, (3,3),  strides=2, padding='same', use_bias=False),
      BatchNormalization(axis=-1),
      ReLU(),
      Conv2D(24, (3,3), strides=2, padding='same', use_bias=False),
      BatchNormalization(axis=-1),
      ReLU(),
      Flatten(),
      Dense(num_classes,activation='softmax')
    ])
    return model
