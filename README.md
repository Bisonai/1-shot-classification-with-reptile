# 1-shot classification with Reptile

Unofficial implementation of the meta-learning algorithm called [Reptile](https://openai.com/blog/reptile/) for 1-shot classification.

This repository contains a pre-trained model with the Omniglot dataset using Reptile for three classes. We provide a test this pre-trained model on three different datasets (Omniglot, MNIST, Openai) with a 1-shot classification.

We characterized the performances of our pre-trained model by proceeding in multiple 1-shot classification on both Omniglot and MNIST datasets. The results are in the following table (under construction).

 | Dataset | learning rate | Average logloss train | Average logloss test |
 | ------- | ------------- | --------------------- | -------------------- |
 | MNIST | 0.01 |  |  |
 | MNIST | 0.005 |  |  |
 | MNIST | 0.001 |  |  |
 | Omniglot | 0.01 |  |  |
 | Omniglot | 0.005 |  |  |
 | Omniglot | 0.001 |  |  |

## Requirements
 * Python 3.6+
 * CVXPY 1.0

 ```shell
 pip install -r requirements.txt
 ```

## Demos

In the following examples, there are three classes with one example for each class. Moreover, we are interested to predict the class of an additional image that belongs to one of the classes. We provide the jupyter notebooks version of our demos. The learning rate and the number of epochs for the fine tuning can be changed within the jupyter notebooks. We used the three following datasets:

 - Omniglot: dataset that was used to pre-train our model. [Jupyter notebook](https://github.com/adelshb/1-shot-classification-with-Reptile/blob/master/demo_omniglot.ipynb).
 - MNIST: this tests how the pre-trained model performed on unseen data. [Jupyter notebook](https://github.com/adelshb/1-shot-classification-with-Reptile/blob/master/demo_mnist.ipynb).
 - Openai: dataset that was used in the original [blogpost](https://openai.com/blog/reptile/) from [OpenAI](https://openai.com). [Jupyter notebook](https://github.com/adelshb/1-shot-classification-with-Reptile/blob/master/demo_openai.ipynb).

## License
[MIT](https://github.com/adelshb/1-shot-classification-with-Reptile/blob/master/LICENSE)
