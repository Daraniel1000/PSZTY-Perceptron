import numpy as np


class Layer:

    def __init__(self, size_layer, size_next_layer, bias):
        self.size_layer = size_layer
        self.size_next_layer = size_next_layer
        self.bias = bias
        self.input = np.array(np.zeros(self.size_layer))
        self.output = np.array(np.zeros(self.size_next_layer))
        self.reinit()

    def reinit(self):
        self.theta = np.array([])

        epsilon = 4.0 * np.sqrt(6) / np.sqrt(self.size_layer + self.size_next_layer)
        if self.bias:
            self.theta = epsilon * ((np.random.rand(self.size_next_layer, self.size_layer + 1) * 2.0) - 1)
        else:
            self.theta = epsilon * ((np.random.rand(self.size_next_layer, self.size_layer) * 2.0) - 1)


class Perceptron:

    def __init__(self, layer_sizes, bias=True):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.layers = []
        self.output_activated: np.array([])
        self.bias = bias
        for i in range(self.n_layers - 1):
            self.layers.append(Layer(self.layer_sizes[i], self.layer_sizes[i + 1], self.bias))

    def train(self, X, Y, n_iterations, reset=False):
        if reset:
            for i in range(len(self.layers)):
                self.layers[i].reinit()
        for i in range(n_iterations):
            print(i)
            self.back_propagate(X, Y)

    def solve(self, X):
        self.forward(X)
        return self.output_activated

    def back_propagate(self, X, Y):
        n_examples = X.shape[0]
        self.forward(X)
        deltas = [None] * self.n_layers
        deltas[-1] = self.output_activated - Y
        for i in np.arange(self.n_layers - 2, 0, -1):
            theta_tmp = self.layers[i].theta
            if self.bias:
                # Removing weights for bias
                theta_tmp = np.delete(theta_tmp, np.s_[0], 1)
            deltas[i] = (np.matmul(theta_tmp.transpose(), deltas[i + 1].transpose())).transpose() * \
                        self.sigmoid_derivative(self.layers[i-1].output)                  #tak chyba działa? liczy delty na każdy layer
        for i in range(self.n_layers - 1):
            grad = np.matmul(deltas[i + 1].transpose(), self.layers[i].input)
            grad = grad / n_examples
            #self.layers[i].theta *= grad
            self.layers[i].theta = np.multiply(self.layers[i].theta, grad)
        #TODO PRZETESTOWAC

    def forward(self, X):
        input_layer = X
        for i in range(len(self.layers)):
            if self.bias:
                # Add bias element to every example in input_layer
                self.layers[i].input = np.concatenate((np.ones([input_layer.shape[0], 1]), input_layer), axis=1)
            else:
                self.layers[i].input = input_layer
            self.layers[i].output = np.matmul(self.layers[i].input, self.layers[i].theta.transpose())
            input_layer = self.sigmoid(self.layers[i].output)
        self.output_activated = input_layer

    def sigmoid(self, z):
        result = 1.0 / (1.0 + np.exp(-z))
        return result

    def sigmoid_derivative(self, z):
        result = self.sigmoid(z) * (1 - self.sigmoid(z))
        return result
