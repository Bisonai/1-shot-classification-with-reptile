import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import tensorflow_datasets as tfds

import numpy as np
from skimage.transform import resize
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import random
from tqdm import tqdm

from fine_tuning.pretrained_models.bisonai.models import OmniglotModelBisonai

def parse_mnist_data(num_classes,
            num_data_per_class,
            num_data_test = 1):

    classes_names = random.sample(range(1, 10), num_classes)

    sess = tf.Session()
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
    return X_train, y_train_label, X_predict, y_predict

def parse_omniglot_data(num_classes,
            num_data_per_class,
            num_data_test = 1):

    classes_names = random.sample(range(1, 800), num_classes)

    sess = tf.Session()
    mnist_train = tfds.load(name="omniglot", split=tfds.Split.TRAIN).batch(num_classes*num_data_per_class*num_data_test+20000)
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
    X_train = resize(np.array(X)/255.0, (int(num_data_per_class*num_classes), 28, 28, 1))
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

    X_predict = resize(np.array(X)/255.0, (int(num_data_test), 28, 28, 1))
    y_predict = np.array(Y)

    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    y_train_label = le.transform(y_train)
    y_predict = le.transform(y_predict)

    sess.close()
    return X_train, y_train_label, X_predict, y_predict

def benchmark(num_classes,
                num_data_per_class,
                model_path,
                dataset_name = "mnist",
                epochs = 5,
                lr_range = [0.01, 0.001, 0.0001],
                num_data_train = 1,
                num_data_test = 1):

    loss_test = []
    loss_train = []
    acc = []

    if dataset_name == "mnist":
        data = [parse_mnist_data(num_classes,
                num_data_per_class,
                num_data_test)
                for _ in range(num_data_train)]
    elif dataset_name == "omniglot":
        data = [parse_omniglot_data(num_classes,
                    num_data_per_class,
                    num_data_test)
                    for _ in range(num_data_train)]

    sess = tf.Session()
    learning_rate = tf.placeholder(tf.float32)
    model = OmniglotModelBisonai(num_classes=num_classes, **{"learning_rate": learning_rate})
    saver = tf.train.Saver()

    for lr in tqdm(lr_range):
        acc_temp_lr = []
        loss_test_temp_lr = []
        loss_train_temp_lr = []
        for d in data:
            X_train, y_train_label, X_predict, y_predict = d[0], d[1], d[2], d[3]
            with tf.Session() as sess:
                saver.restore(sess, model_path)
                loss_test_temp = []
                loss_train_temp = []
                for e in range(epochs):
                    loss = sess.run(model.loss, feed_dict={model.input_ph: X_train.reshape(X_train.shape[:3]), model.label_ph: y_train_label})
                    loss_train_temp.append(sum(loss)/(num_classes*num_data_per_class))
                    loss = sess.run(model.loss, feed_dict={model.input_ph: X_predict.reshape(X_predict.shape[:3]), model.label_ph: y_predict})
                    loss_test_temp.append(sum(loss)/num_data_test)
                    sess.run(model.minimize_op, feed_dict={model.input_ph: X_train.reshape(X_train.shape[:3]), model.label_ph:y_train_label, learning_rate: lr})
                loss_test_temp_lr.append(loss_test_temp)
                loss_train_temp_lr.append(loss_train_temp)
                pred = sess.run(model.predictions,
                            feed_dict={model.input_ph: X_predict.reshape(X_predict.shape[:3])})
            acc_temp = accuracy_score(
                            pred,
                            y_predict)
            acc_temp_lr.append(acc_temp)
        loss_test.append(np.sum(loss_test_temp_lr, axis=0)/num_data_train)
        loss_train.append(np.sum(loss_train_temp_lr, axis=0)/num_data_train)
        acc.append(sum(acc_temp_lr)/num_data_train)

    sess.close()
    return acc, loss_train, loss_test
