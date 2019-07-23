import tensorflow as tf
import numpy as np

from argparse import ArgumentParser

from data_factory import parse_train_data, parse_predict_data
from omniglot_model import omniglot
from utils import omnoiglot_get_weights

def main(args):

    X_train, y_train = parse_train_data(args.data_path)

    model = omniglot(input_shape=X_train.shape[1:], num_classes=y_train.shape[1])

    if args.pretrained_model == True:
        X = omnoiglot_get_weights()
        model.layers[0].set_weights([X[0]])
        model.layers[3].set_weights([X[6]])
        model.layers[6].set_weights([X[12]])
        model.layers[9].set_weights([X[18]])

        model.layers[1].set_weights([X[2], X[4], np.array([0]*24), np.array([1]*24)])
        model.layers[4].set_weights([X[8], X[10], np.array([0]*24), np.array([1]*24)])
        model.layers[7].set_weights([X[14], X[16], np.array([0]*24), np.array([1]*24)])
        model.layers[10].set_weights([X[20], X[22], np.array([0]*24), np.array([1]*24)])

        model.layers[13].set_weights([X[24], X[26]])

    model.compile(loss=tf.compat.v1.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.compat.v1.train.AdamOptimizer(args.learning_rate),
                  metrics=[tf.compat.v1.keras.metrics.CategoricalAccuracy()])

    model.fit(X_train, y_train,
              steps_per_epoch=(X_train.shape[0]//args.batch_size),
              epochs=args.epochs,)

    tf.keras.models.save_model(model, "fine_tuned_model", include_optimizer=True)

    print(model.predict(X_train))
    if args.predict == True:
        X_predict = parse_predict_data(args.data_path)
        result = model.predict(X_predict)[0]
        return result

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_path", type=str, default='data')
    parser.add_argument("--pretrained_model", type=str, default=True)
    parser.add_argument("--predict", type=str, default=True)

    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)

    args = parser.parse_args()
    main(args)
