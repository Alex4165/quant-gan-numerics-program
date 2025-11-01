from typing import List
import numpy as np

# Refer to "Quant GANs: Deep Generation of Financial Time Series" by Wiese et al. (2019) for definitions and conventions


def leaky_re_lu(x): return np.maximum(0.01 * x, x)
def grad_leaky_re_lu(x): return np.where(x > 0, 1, 0.01)


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / e_x.sum(axis=0, keepdims=True)


class VanillaTCN:
    def __init__(self, input_size: int, dilations: List[int], kernel_sizes: List[int], hidden_sizes: List[int],
                 weight_scale: float = 0.1, bias_init: float = 1):
        self.input_size = input_size
        self.dilations = dilations
        self.kernel_sizes = kernel_sizes
        self.hidden_sizes = hidden_sizes

        if not (len(dilations) == len(kernel_sizes) == len(hidden_sizes)):
            raise ValueError("Dilation list, kernel size list, and hidden layer size list must have the same length.")
        self.depth = len(dilations)
        self.T_f = 1 + sum([d * (k - 1) for d, k in zip(dilations, kernel_sizes)])  # receptive field size
        self.node_vals = [0 for _ in range(self.depth+1)]  # to store intermediate values for backpropagation

        self.used_node_idx = self.initialize_used_node_indices()
        self.weights, self.dw = self.initalize_weights(weight_scale)
        self.biases, self.db = [bias_init+np.zeros((h,)) for h in hidden_sizes], [np.zeros((h,)) for h in hidden_sizes]

    def initalize_weights(self, scale: float):
        # follows def 3.5
        # convention is that we go from output to input layers (so 0 is output layer, depth-1 is first hidden layer)
        weights = [0 for _ in range(self.depth)]
        for i, d, k in zip(range(self.depth), self.dilations, self.kernel_sizes):
            if i == self.depth - 1:
                weights[i] = scale*np.random.randn(k, self.input_size, self.hidden_sizes[i]).astype(np.float32)
            else:
                weights[i] = scale*np.random.randn(k, self.hidden_sizes[i+1], self.hidden_sizes[i]).astype(np.float32)
        return weights, [np.zeros_like(w) for w in weights]  # dw initialized to zeros

    def scale_grads(self, scale):
        for i in range(self.depth):
            self.dw[i] *= scale
            self.db[i] *= scale

    def initialize_used_node_indices(self):
        idx = np.zeros((self.depth, self.T_f), dtype=bool)
        idx[0, -1] = 1  # last node in last layer is only one used
        for i in range(1, self.depth):
            for j in idx[i-1].nonzero()[0]:
                for k in range(self.kernel_sizes[i-1]):
                    idx[i, j - self.dilations[i-1]*k] = 1
        return idx

    def train_minibatch(self, inputs_batch, targets_batch, momentum: float = 0.9, learning_rate: float = 0.001,
                        return_loss: bool = False):
        m = inputs_batch.shape[0]
        total_loss = 0
        self.scale_grads(m*momentum)  # decay prev grads, scale by m to avg later
        for i in range(m):
            if return_loss:
                total_loss += self.update_gradients(inputs_batch[i, :, :], targets_batch[i, :])
            else:
                self.update_gradients(inputs_batch[i, :, :], targets_batch[i, :])
        self.scale_grads(1/m)
        for i in range(self.depth):
            self.weights[i] -= learning_rate * self.dw[i]
            self.biases[i] -= learning_rate * self.db[i]
        if return_loss:
            return total_loss / m

    def update_gradients(self, inputs, target):
        output, loss = self.forward_pass(inputs, target)
        self.back_prop(output, target)
        return loss

    def back_prop(self, output, target):
        # Last node by hand
        D, K, N_O = self.dilations[0], self.kernel_sizes[0], self.hidden_sizes[0]
        dL = output - target  # shape: (N_0, )
        idx = -1 - D * (K - np.arange(K) - 1)  # indices of child nodes
        self.dw[0] = leaky_re_lu(self.node_vals[1][:, idx].T)[:, :, None] * dL
        self.db[0] = dL  # (N_O, )

        # Notation below is phi is the convolution (output) and f is the activation function (output)
        dL_dphi = np.zeros((N_O, self.T_f))  # (N_O, T_f)
        dL_dphi[:, -1] = dL
        # Remaining nodes done iteratively
        for i in range(1, self.depth):
            D_prev, K_prev, N_O = self.dilations[i - 1], self.kernel_sizes[i - 1], self.hidden_sizes[i]
            D, K = self.dilations[i], self.kernel_sizes[i]

            # calculate dL/df = sum dL/dphi * dphi/df (for phi from prev layer)
            dL_df = np.zeros((N_O, self.T_f), dtype=np.float32)
            for i0 in self.used_node_idx[i - 1].nonzero()[0]:  # parent index
                for k in range(K_prev):
                    i1 = i0 - D_prev * (K_prev - k - 1)  # child index
                    dL_df[:, i1] += np.matmul(self.weights[i - 1][k, :, :], dL_dphi[:, i0])

            # calculate dL/dphi
            dL_dphi = dL_df * grad_leaky_re_lu(self.node_vals[i])  # (N_O, T_f)

            # calculate dL/dw and dL/db
            for j in self.used_node_idx[i].nonzero()[0]:
                idx = j - D * (K - np.arange(K) - 1)  # indices of child nodes
                self.dw[i] += leaky_re_lu(self.node_vals[i + 1][:, idx].T)[:, :, None] * dL_dphi[:, j]
                self.db[i] += dL_dphi[:, j]

    def forward_pass(self, inputs, target=None):
        # inputs shape: (input_size, T_f)
        layer_input = inputs
        for i in reversed(range(self.depth)):
            D, K, N_O = self.dilations[i], self.kernel_sizes[i], self.hidden_sizes[i]
            layer_output = np.zeros((N_O, self.T_f), dtype=np.float32)
            # TODO: pretty sure this can be vectorized as well
            for j in self.used_node_idx[i].nonzero()[0]:
                for k in range(K):
                    layer_output[:, j] += np.matmul(layer_input[:, j - D * (K - k - 1)], self.weights[i][k, :, :])
                layer_output[:, j] += self.biases[i]
            if i != 0:
                layer_input = leaky_re_lu(layer_output)
            else:
                layer_input = softmax(layer_output[:, -1])  # only last time step is relevant for output
            self.node_vals[i] = layer_output
        self.node_vals[-1] = inputs  # suboptimal but makes code simpler TODO: add "if"s to update_grad and remove this
        if target is None:
            return layer_input
        loss = - np.sum(target * np.log(layer_input))  # no +epsilon as softmax almost never outputs exact 0
        return layer_input, loss


if __name__ == "__main__":
    # --- gradient testing ---
    # TCN
    input_size = 3
    dilations = [2, 1]
    kernel_sizes = [2, 2]
    hidden_sizes = [input_size, 4]
    tcn = VanillaTCN(input_size, dilations, kernel_sizes, hidden_sizes)

    # random input and output
    inputs = np.random.randn(input_size, tcn.T_f).astype(np.float32)
    target = np.zeros((hidden_sizes[0],), dtype=np.float32)
    target[0] = 1.0  # one-hot

    # compute analytic gradient via update_grad
    tcn.update_gradients(inputs, target)

    # hardcoded weight indices to check
    layer_idx = 0  # this one's the main one to change
    k_idx = 0
    in_idx = 0
    out_idx = 0
    analytic = tcn.dw[layer_idx][k_idx, in_idx, out_idx]

    # numerical gradient
    eps = 1e-4
    orig = tcn.weights[layer_idx][k_idx, in_idx, out_idx].astype(np.float64)

    tcn.weights[layer_idx][k_idx, in_idx, out_idx] = (orig + eps).astype(np.float32)
    loss_plus = tcn.forward_pass(inputs, target)[1]

    tcn.weights[layer_idx][k_idx, in_idx, out_idx] = (orig - eps).astype(np.float32)
    loss_minus = tcn.forward_pass(inputs, target)[1]

    tcn.weights[layer_idx][k_idx, in_idx, out_idx] = orig.astype(np.float32)  # restore

    numerical = (loss_plus - loss_minus) / (2 * eps)

    print(tcn.forward_pass(inputs, target))

    # print results
    print("analytic:", float(analytic))
    print("numerical:", float(numerical))
    rel_err = abs(analytic - numerical) / max(abs(numerical), abs(analytic), 1e-8)
    print("relative error:", rel_err)









