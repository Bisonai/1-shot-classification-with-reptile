# 1-shot classification with Reptile

Unofficial implementation of fine-tuning phase for meta-learning algorithm [Reptile](https://arxiv.org/abs/1803.02999) applied on image classification problem with 1-shot 3-way settings.

This repository contains a 1-shot 3-way Reptile model pre-trained on [Omniglot](https://github.com/brendenlake/omniglot) dataset.
We provide simple demos ([Omniglot demo](https://github.com/Bisonai/1-shot-classification-with-Reptile/blob/master/demo_omniglot.ipynb), [MNIST demo](https://github.com/Bisonai/1-shot-classification-with-Reptile/blob/master/demo_mnist.ipynb) and [OpenAI demo](https://github.com/Bisonai/1-shot-classification-with-Reptile/blob/master/demo_openai.ipynb)) on three different datasets ([Omniglot](https://github.com/brendenlake/omniglot), [MNIST](http://yann.lecun.com/exdb/mnist/) and [OpenAI](https://github.com/Bisonai/1-shot-classification-with-Reptile/tree/master/data_examples/openai)).

We characterized the performance of our pre-trained model by performing an evaluation on both [Omniglot](https://github.com/brendenlake/omniglot) and [MNIST](http://yann.lecun.com/exdb/mnist/) datasets with 1-shot 3-way settings.
[Cross entropy loss](https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits) was used as an evaluation metric.
The results can be found in the table below (under construction).

 | Dataset  | Learning rate | Average loss train | Average loss test |
 | -------  | ------------- | ------------------ | ----------------- |
 | MNIST    | 0.01          | 1.48e-04           | 1.43              |
 | MNIST    | 0.001         | 0.0154             | 0.685             |
 | Omniglot | 0.01          | 3.03e-04           | 0.252             |
 | Omniglot | 0.001         | 0.0191             | 0.294             |

## Requirements

 * Python 3.6+
 * Tensorflow
 * skimage
 * matplotlib

 ```shell
 pip install -r requirements.txt
 ```

## Demos

In the following demos, we randomly sample three classes and take one image from each of them.
Then, we fine-tune our [pre-trained model](https://github.com/Bisonai/1-shot-classification-with-Reptile/tree/master/pretrained_models/bisonai/1shot_3way_bisonai_ckpt_o15t) using three previously sampled images and finally we test performance by classifying another image belonging to one of three sampled classes.
We provide Jupyter notebooks version of our demos.
The learning rate and the number of epochs for the fine-tuning can be changed within Jupyter notebooks.
We used three following datasets:

 - [Omniglot](https://github.com/brendenlake/omniglot) dataset was used to pre-train our model ([Jupyter notebook](https://github.com/Bisonai/1-shot-classification-with-Reptile/blob/master/demo_omniglot.ipynb))
 - [MNIST](http://yann.lecun.com/exdb/mnist/) dataset shows how our pre-trained model performs on previously unseen data ([Jupyter notebook](https://github.com/Bisonai/1-shot-classification-with-Reptile/blob/master/demo_mnist.ipynb))
 - [OpenAI](https://github.com/Bisonai/1-shot-classification-with-Reptile/tree/master/data_examples/openai) dataset was used in the original [blogpost](https://openai.com/blog/reptile/) from [OpenAI](https://openai.com) ([Jupyter notebook](https://github.com/Bisonai/1-shot-classification-with-Reptile/blob/master/demo_openai.ipynb))

## License

[MIT License](https://github.com/Bisonai/1-shot-classification-with-Reptile/blob/master/LICENSE)
