from typing import List
import numpy as np

# Refer to "Quant GANs: Deep Generation of Financial Time Series" by Wiese et al. (2019)


def LeakyReLU(x):
        return np.maximum(0.1*x, x)


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / e_x.sum(axis=0, keepdims=True)


class VanillaTCN:
    def __init__(self, input_size: int, dilations: List[int], kernel_sizes: List[int], hidden_sizes: List[int]):
        self.input_size = input_size
        self.dilations = dilations
        self.kernel_sizes = kernel_sizes
        self.hidden_sizes = hidden_sizes

        if not (len(dilations) == len(kernel_sizes) == len(hidden_sizes)):
            raise ValueError("Dilation list, kernel size list, and hidden layer size list must have the same length.")
        self.depth = len(dilations)
        self.T_f = 1 + sum([d * (k - 1) for d, k in zip(dilations, kernel_sizes)])  # receptive field size

        self.used_node_idx = self.initialize_used_node_indices()
        self.weights = self.initalize_weights()
        self.biases = [np.zeros((h,)) for h in hidden_sizes]

    def initalize_weights(self):
        # follows def 3.5
        # we reverse here to keep with the convention that we go from output to input layers
        weights = [0 for _ in range(self.depth)]
        for i, d, k in zip(range(self.depth), self.dilations, self.kernel_sizes):
            if i == self.depth - 1:
                weights[i] = np.random.randn(k, self.input_size, self.hidden_sizes[i]).astype(np.float32)
            else:
                weights[i] = np.random.randn(k, self.hidden_sizes[i-1], self.hidden_sizes[i]).astype(np.float32)
        return weights

    def initialize_used_node_indices(self):
        idx = np.zeros((self.depth, self.T_f), dtype=bool)
        idx[0, -1] = 1  # last node in last layer is
        for i in range(1, self.depth):
            for j in idx[i-1].nonzero()[0]:
                for k in range(self.kernel_sizes[i-1]):
                    idx[i, j - self.dilations[i-1]*k] = 1
        return idx

    def loss(self, inputs, output):
        # inputs shape: (input_size, T_f)

        # foward pass
        layer_input = inputs
        for i in reversed(range(self.depth)):
            D, K, N_O = self.dilations[i], self.kernel_sizes[i], self.hidden_sizes[i]
            layer_output = np.zeros((N_O, self.T_f), dtype=np.float32)

            for j in self.used_node_idx[i].nonzero()[0]:
                for k in range(1, K+1):
                    layer_output[:, j] += np.matmul(layer_input[:, j-D*(K-k)], self.weights[i][k-1, :, :])
                layer_output[:, j] += self.biases[i]

            if i != 0:
                layer_input = LeakyReLU(layer_output)
            else:
                layer_input = softmax(layer_output[:, -1])  # only last time step is relevant for output

        # cross-entropy loss
        loss = - np.sum(output * np.log(layer_input + 1e-8)) / inputs.shape[0]

        # gradients



if __name__ == "__main__":
    file = "input.txt"
    with open(file, "r") as f:
        text = f.read()
    chars = ''.join(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)

    model = VanillaTCN(input_size=4, dilations=[2, 1],
                       kernel_sizes=[2, 2], hidden_sizes=[4, 3])
    model.loss(np.array([[0, 0, 0, 1],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [1, 0, 0, 0]], dtype=np.float32))






