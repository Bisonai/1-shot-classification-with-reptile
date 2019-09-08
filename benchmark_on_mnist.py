import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import tensorflow_datasets as tfds

import numpy as np
from skimage.transform import resize
from sklearn import preprocessing
import random

from fine_tuning.pretrained_models.bisonai.models import OmniglotModelBisonai

def benchmark_mnist(num_classes,
                num_data_per_class,
                model_path,
                epochs,
                lr_range = [0.01, 0.001, 0.0001, 0.00001],
                num_data_test = 1):

    sess = tf.Session()

    classes_names = random.sample(range(1, 10), num_classes)

    mnist_train = tfds.load(name="mnist", split=tfds.Split.TRAIN).batch(num_classes*num_data_per_class*num_data_test+20000)
    mnist_example = mnist_train.take(1)
    mnist_example_iter = mnist_example.make_initializable_iterator()
    sess.run(mnist_example_iter.initializer)

    data = mnist_example_iter.get_next()
    image = data['image']
    label = data['label']
    x, y = sess.run([image,label])

    X = []
    Y = []
    train_data_index = []
    for i, item in enumerate(x):
        if y[i] in classes_names and Y.count(y[i])<num_data_per_class:
            Y.append(y[i])
            X.append(item)
            train_data_index.append(i)
        if len(Y) == num_classes*num_data_per_class:
            break

    X_train = np.array(1-np.array(X)/255.0).reshape(int(num_data_per_class*num_classes), 28, 28, 1)
    y_train = np.array(Y)

    res = sum([np.where(y == c)[0].tolist() for c in y_train], [])
    [res.remove(i) for i in train_data_index]

    X = []
    Y = []
    for i in range(num_data_test):
        ind = random.choice(res)
        res.remove(ind)
        Y.append(y[ind])
        X.append(x[ind])

    X_predict = np.array(1-np.array(X)/255.0).reshape(int(num_data_test), 28, 28, 1)
    y_predict = np.array(Y)

    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    y_train_label = le.transform(y_train)
    y_predict = le.transform(y_predict)

    sess.close()

    learning_rate = tf.placeholder(tf.float32)
    model = OmniglotModelBisonai(num_classes=num_classes, **{"learning_rate": learning_rate})
    saver = tf.train.Saver()

    loss_test = []
    loss_train = []

    for lr in lr_range:
        with tf.Session() as sess:
            saver.restore(sess, model_path)
            #sess.run(tf.global_variables_initializer())
            loss_test_temp = []
            loss_train_temp = []
            for e in range(epochs):
                loss = sess.run(model.loss, feed_dict={model.input_ph: X_train.reshape(X_train.shape[:3]), model.label_ph: y_train_label})
                loss_train_temp.append(loss)
                loss = sess.run(model.loss, feed_dict={model.input_ph: X_predict.reshape(X_predict.shape[:3]), model.label_ph: y_predict})
                loss_test_temp.append(loss)
                sess.run(model.minimize_op, feed_dict={model.input_ph: X_train.reshape(X_train.shape[:3]), model.label_ph:y_train_label, learning_rate: lr})
            loss_test.append(loss_test_temp)
            loss_train.append(loss_train_temp)

    return loss_train, loss_test
