import tensorflow as tf
import numpy as np

from argparse import ArgumentParser

from data_factory import parse_train_data, parse_predict_data
from pretrained_models.model_factory import build_model

def main(args):

    X_train, y_train = parse_train_data(args.data_path)
    X_predict = parse_predict_data(args.data_path)

    sess = tf.Session()
    model = build_model(args.model_name, num_classes=y_train.shape[0])
    saver = tf.train.Saver()
    saver.restore(sess, args.checkpoint_path)

    for e in range(args.epochs):
        sess.run(model.minimize_op, feed_dict={model.input_ph: X_train.reshape(X_train.shape[:3]), model.label_ph: y_train})

    if args.predict== True:
        result = sess.run(model.predictions, feed_dict={model.input_ph: X_predict.reshape(X_predict.shape[:3])})
        print("The predicted class is " + str(result[0])+".")

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_path", type=str, default='example/openai')
    parser.add_argument("--model_name", type=str, default="bisonai")
    parser.add_argument("--checkpoint_path", type=str, default="fine_tuning/pretrained_models/bisonai/1shot_3way_bisonai_ckpt_o15t/model.ckpt-99999")

    parser.add_argument("--predict", type=str, default=True)

    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=40)

    args = parser.parse_args()
    main(args)
