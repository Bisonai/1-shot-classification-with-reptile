import re
import numpy as np

def omnoiglot_get_weights():
    shapes = []
    weights = []

    filepath = "pretrained_models/openai/pretrained_omniglot_openai"
    with open(filepath) as fp:
        for line in fp:
            try:
                shape = np.array(eval(re.search('jsnet.Tensor\((.*), ', line).group(1)))
                shapes.append(shape)
            except AttributeError:
                continue
            try:
                weight = np.array(eval(re.search(', (.*)\)', line).group(1)))
                weights.append(weight)
            except AttributeError:
                continue

    X = []
    for index in range(len(shapes)):
        X.append(weights[index].reshape(shapes[index]))

    return X
