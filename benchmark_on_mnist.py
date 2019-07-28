from fine_tuning.pretrained_models.bisonai.models import OmniglotModelBisonai
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

_checkpoint_path = "fine_tuning/pretrained_models/bisonai/1shot_3way_bisonai_ckpt_o15t/model.ckpt-99999"

def training_data(num_classes=3):
    sess = tf.Session()

    mnist_train = tfds.load(name="mnist", split=tfds.Split.TRAIN)
    mnist_train = mnist_train.shuffle(1024)
    mnist_example = mnist_train.take(1000)
    mnist_example_iter = mnist_example.make_initializable_iterator()
    sess.run(mnist_example_iter.initializer)

    X_train = []
    y_train = []

    while len(y_train) != num_classes:
        data = mnist_example_iter.get_next()
        image = data['image']
        label = data['label']
        x, y = sess.run([image,label])
        if y not in set(y_train):
            X_train.append(x)
            y_train.append(y)
        if len(y_train) == num_classes:
            break

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    min = -1
    max = 1
    scale = (max - min) / (X_train.max() - X_train.min())
    X_train = scale * X_train + min - X_train.min() * scale

    return  X_train, y_train

def predict_data(y_train):
    sess = tf.Session()

    mnist_test = tfds.load(name="mnist", split=tfds.Split.TEST)
    mnist_test = mnist_test.shuffle(1024)
    mnist_example = mnist_test.take(1000)
    mnist_example_iter = mnist_example.make_initializable_iterator()
    sess.run(mnist_example_iter.initializer)

    while True:
        data = mnist_example_iter.get_next()
        image = data['image']
        label = data['label']
        x, y = sess.run([image,label])
        if y in set(y_train):
            X_predict = x
            y_predict = y
            break

    X_predict = np.array(X_predict).reshape(1, 28, 28, 1)
    y_predict = np.array(y_predict)

    min = -1
    max = 1
    scale = (max - min) / (X_predict.max() - X_predict.min())
    X_predict = scale * X_predict + min - X_predict.min() * scale

    return X_predict, y_predict

def predict(X_train,
            y_train,
            X_predict,
            epochs = 40,
            num_classes=3):

    sess = tf.Session()

    model = OmniglotModelBisonai(num_classes=num_classes)
    saver = tf.train.Saver()
    saver.restore(sess, _checkpoint_path)

    y_train_label = np.array([i for i in range(num_classes)])

    for e in range(epochs):
        sess.run(model.minimize_op, feed_dict={model.input_ph: X_train.reshape(X_train.shape[:3]), model.label_ph: y_train_label})
    result = sess.run(model.predictions, feed_dict={model.input_ph: X_predict.reshape(X_predict.shape[:3])})

    return y_train[result[0]]
