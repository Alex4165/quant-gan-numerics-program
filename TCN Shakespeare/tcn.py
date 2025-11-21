from typing import List
import numpy as np


# Refer to "Quant GANs: Deep Generation of Financial Time Series" by Wiese et al. (2019) for TCN
# definitions and conventions


def leaky_re_lu(x): return np.maximum(0.1 * x, x)


def grad_leaky_re_lu(x): return np.where(x > 0, 1, 0.1)


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / e_x.sum(axis=0, keepdims=True)


class VanillaTCN:
    def __init__(self, input_size: int, dilations: List[int], kernel_sizes: List[int], hidden_sizes: List[int],
                 weight_scale: float = 0.01, bias_init: float = 0.1, input_p_keep=1.0, hidden_p_keep=1.0,
                 grad_clip: float = 1):
        self.input_size = input_size
        self.dilations = dilations
        self.kernel_sizes = kernel_sizes
        self.hidden_sizes = hidden_sizes
        self.input_p_keep, self.hidden_p_keep = input_p_keep, hidden_p_keep  # dropout
        self.t = 1  # time index for Adam
        self.grad_clip = grad_clip

        if not (len(dilations) == len(kernel_sizes) == len(hidden_sizes)):
            raise ValueError("Dilation list, kernel size list, and hidden layer size list must have the same length.")
        self.depth = len(dilations)
        self.T_f = 1 + sum([d * (k - 1) for d, k in zip(dilations, kernel_sizes)])  # receptive field size
        self.node_vals = [0 for _ in range(self.depth + 1)]  # to store intermediate values for backpropagation

        self.used_node_idx = self.initialize_used_node_indices()
        self.input_mask = np.ones((input_size, self.T_f), dtype=np.float32)  # for dropout
        self.hidden_masks = [np.ones((h, self.T_f), dtype=np.float32) for h in hidden_sizes]  # for dropout
        self.weights, self.dw = self.initalize_weights(weight_scale)
        self.biases, self.db = [bias_init + np.zeros((h,)) for h in hidden_sizes], [np.zeros((h,)) for h in
                                                                                    hidden_sizes]
        self.rw, self.rb = [np.zeros_like(w) for w in self.weights], [np.zeros_like(b) for b in self.biases]
        self.vw, self.vb = [np.zeros_like(w) for w in self.weights], [np.zeros_like(b) for b in self.biases]

    def initalize_weights(self, scale=0.1):
        """He initialization"""
        # follows def 3.5
        # convention is that we go from output to input layers (so 0 is output layer, depth-1 is first hidden layer)
        weights = [np.ndarray for _ in range(self.depth)]
        for i, d, k in zip(range(self.depth), self.dilations, self.kernel_sizes):
            if i == self.depth - 1:  # first hidden layer
                weights[i] = scale * np.random.randn(k, self.input_size, self.hidden_sizes[i]).astype(np.float32)
            else:
                weights[i] = scale * np.random.randn(k, self.hidden_sizes[i + 1], self.hidden_sizes[i]).astype(
                    np.float32)
        return weights, [np.zeros_like(w) for w in weights]  # dw initialized to zeros

    def scale_grads(self, scale):
        for i in range(self.depth):
            self.dw[i] *= scale
            self.db[i] *= scale

    def initialize_used_node_indices(self):
        idx = np.zeros((self.depth, self.T_f), dtype=bool)
        idx[0, -1] = 1  # last node in last layer is only one used
        for i in range(1, self.depth):
            for j in idx[i - 1].nonzero()[0]:
                for k in range(self.kernel_sizes[i - 1]):
                    idx[i, j - self.dilations[i - 1] * k] = 1
        # print(f"Used nodes matrix: \n {idx.astype(int)}")
        return idx

    def train_minibatch(self, inputs_batch, targets_batch, learning_rate: float = 0.001,
                        rho_1: float = 0.9, rho_2: float = 0.999, update_weights: bool = True):
        """Adam optimizer"""
        m = inputs_batch.shape[0]
        total_loss = 0
        self.scale_grads(0)

        for i in range(m):
            total_loss += self.update_gradients(inputs_batch[i], targets_batch[i], do_dropout=update_weights)
        self.scale_grads(1 / m)

        if not update_weights:
            return total_loss / m
        for i in range(self.depth):
            self.vw[i] = (rho_1 * self.vw[i] + (1 - rho_1) * self.dw[i])
            self.vb[i] = (rho_1 * self.vb[i] + (1 - rho_1) * self.db[i])

            if rho_2 > 0:
                self.rw[i] = (rho_2 * self.rw[i] + (1 - rho_2) * (self.dw[i] ** 2))
                self.rb[i] = (rho_2 * self.rb[i] + (1 - rho_2) * (self.db[i] ** 2))

                self.weights[i] += (- learning_rate * (self.vw[i] / (1 - rho_1 ** self.t))
                                    / (np.sqrt(self.rw[i] / (1 - rho_2 ** self.t)) + 1e-8))
                self.biases[i] += (- learning_rate * (self.vb[i] / (1 - rho_1 ** self.t))
                                   / (np.sqrt(self.rb[i] / (1 - rho_2 ** self.t)) + 1e-8))
            else:
                self.weights[i] += - learning_rate * (self.vw[i] / (1 - rho_1 ** self.t))
                self.biases[i] += - learning_rate * (self.vb[i] / (1 - rho_1 ** self.t))
        self.t += 1

        return total_loss / m

    def update_gradients(self, inputs, target, do_dropout=True):
        output, loss = self.forward_pass(inputs, target, do_dropout=do_dropout)
        if not do_dropout:
            return loss
        self.back_prop(output, target)
        return loss

    def back_prop(self, output, target):
        # Last node by hand
        D, K, N_O = self.dilations[0], self.kernel_sizes[0], self.hidden_sizes[0]
        dL = output - target  # shape: (N_0, )
        idx = -1 - D * (K - np.arange(K) - 1)  # indices of child nodes
        self.dw[0] += leaky_re_lu(self.node_vals[1][:, idx].T)[:, :, None] * dL
        self.db[0] += dL  # (N_O, )

        dL_dphi = np.zeros((N_O, self.T_f))
        dL_dphi[:, -1] = dL
        for i in range(1, self.depth):
            D_prev, K_prev, N_O = self.dilations[i - 1], self.kernel_sizes[i - 1], self.hidden_sizes[i]
            D, K = self.dilations[i], self.kernel_sizes[i]

            dL_df = np.zeros((N_O, self.T_f), dtype=np.float32)
            js_prev = self.used_node_idx[i - 1].nonzero()[0]
            if js_prev.size:
                idx_prev = js_prev[np.newaxis, :] - D_prev * (K_prev - np.arange(K_prev) - 1)[:, np.newaxis]
                dL_sel = dL_dphi[:, js_prev]
                w = self.weights[i - 1]
                contrib = np.tensordot(w, dL_sel, axes=([2], [0]))
                contrib_flat = contrib.reshape(contrib.shape[1], -1)
                flat_idx = idx_prev.ravel()
                np.add.at(dL_df, (slice(None), flat_idx), contrib_flat)

            dL_dphi = dL_df * grad_leaky_re_lu(self.node_vals[i]) * self.hidden_masks[i]

            js = self.used_node_idx[i].nonzero()[0]
            if js.size:
                idx = js[np.newaxis, :] - D * (K - np.arange(K) - 1)[:, np.newaxis]
                selected = self.node_vals[i + 1][:, idx]
                sel = leaky_re_lu(selected).transpose(1, 0, 2)
                dLj = dL_dphi[:, js]
                contrib_w = np.tensordot(sel, dLj, axes=([2], [1]))
                self.dw[i] += contrib_w
                self.db[i] += dLj.sum(axis=1)

            # Clip gradients:
            self.dw[i] = np.clip(self.dw[i], -self.grad_clip, self.grad_clip)

    def forward_pass(self, inputs, target=None, do_dropout=True):
        # inputs shape: (input_size, T_f)

        # if do_dropout:
        #     mask = (np.random.rand(*inputs.shape) < self.input_p_keep).astype(np.float32)
        #     self.input_mask = mask / self.input_p_keep
        #     layer_input = mask * self.input_mask
        # else:
        #     self.input_mask = np.ones_like(inputs, dtype=np.float32)
        #     layer_input = inputs
        layer_input = inputs

        for i in reversed(range(self.depth)):
            # dilated causal convolution
            D, K, N_O = self.dilations[i], self.kernel_sizes[i], self.hidden_sizes[i]
            layer_output = np.zeros((N_O, self.T_f), dtype=np.float32)

            js = self.used_node_idx[i].nonzero()[0]
            idx = js[np.newaxis, :] - D * (K - np.arange(K) - 1)[:, np.newaxis]  # (K, M)
            selected = layer_input[:, idx]  # (N_I, K, M)
            sel = selected.transpose(1, 0, 2)  # (K, N_I, M)
            w = self.weights[i].transpose(0, 2, 1)  # (K, N_O, N_I)
            # batched matmul -> (K, N_O, M), then sum over K -> (N_O, M)
            contrib = np.matmul(w, sel).sum(axis=0)
            layer_output[:, js] = contrib + self.biases[i][:, None]

            if do_dropout and i > 2:  # no dropout on final three layers (since sparse)
                mask = (np.random.rand(*layer_output.shape) < self.hidden_p_keep).astype(np.float32)
                scaled_mask = mask / self.hidden_p_keep
                layer_output *= scaled_mask  # weight-scaling done during training
                self.hidden_masks[i] = scaled_mask
            else:
                self.hidden_masks[i] = np.ones_like(layer_output, dtype=np.float32)

            if i != 0:
                layer_input = leaky_re_lu(layer_output)
            else:
                layer_output += inputs  # identity skip connection (effectively on most recent input)
                layer_input = softmax(layer_output[:, -1])  # only last time step is relevant for output

            self.node_vals[i] = layer_output  # store pre-activation values for backprop
        self.node_vals[-1] = inputs  # suboptimal but makes code simpler TODO: add "if"s to update_grad and remove this
        if target is None:
            return layer_input
        loss = - np.sum(target * np.log(layer_input + 1e-20))
        # print(f"target idx {np.argmax(target)} | pred idx {np.argmax(layer_input)} | loss {loss:.4f}")
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
