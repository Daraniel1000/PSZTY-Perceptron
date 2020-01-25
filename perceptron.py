import numpy as np

class layer:

    def __init__(self, size_layer, size_next_layer, bias_flag = 1):
        self.theta = []

        epsilon = 4.0 * np.sqrt(6) / np.sqrt(size_layer + size_next_layer)
        if bias_flag:
            self.theta = epsilon * ((np.random.rand(size_next_layer, size_layer + 1) * 2.0) - 1)
        else:
            self.theta = epsilon * ((np.random.rand(size_next_layer, size_layer) * 2.0) - 1)
        self.input = np.array(np.zeros(size_layer))
        self.output = np.array(np.zeros(size_next_layer))


class Perceptron:

    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.layers = []
        for i in range(self.n_layers - 1):
            self.layers.append(layer(self.layer_sizes[i], self.layer_sizes[i+1]))



    def sigmoid(self, z):
        result = 1.0 / (1.0 + np.exp(-z))
        return result

    def sigmoid_derivative(self, z):
        result = self.sigmoid(z) * (1 - self.sigmoid(z))
        return result
