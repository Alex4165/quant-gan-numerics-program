import time

import numpy as np
from tcn import VanillaTCN

if __name__ == "__main__":
    # --- hyperparameters ---
    # training
    learning_rate = 0.001
    rho_1, rho_2 = 0.9, 0.999  # Adam params
    dropout_input_p_keep = 0.8
    dropout_hidden_p_keep = 0.8

    batch_size = 10
    max_epochs = 100
    early_stop_rel_tol = 1e-3
    data_size = 500  # limit data size for faster training, -1 for full data
    # number of backprop steps = O(batch_size * data_size * epochs)

    # model
    copies = 2  # copies of kernel-dilation list, so final_depth = depth * copies, see below
    depth = 3
    kernel_size = 3
    dilation_size = 2
    hidden_size = 20
    # num parameters = O(copies * depth * kernel_size * hidden_size^2)

    # generation
    seed_text = "To be, or not to be"
    generation_length = 50

    # --- data ---
    file = "input.txt"
    with open(file, "r") as f:
        text = f.read()
    chars = ''.join(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)
    text = text[:data_size]

    # model init
    dilations = copies * [dilation_size**i for i in range(depth-1, -1, -1)]
    kernel_sizes = copies * [kernel_size for _ in range(depth)]
    hidden_sizes = [vocab_size] + [hidden_size for _ in range(copies * depth-1)]
    model = VanillaTCN(input_size=vocab_size, dilations=dilations,
                       kernel_sizes=kernel_sizes, hidden_sizes=hidden_sizes,
                       input_p_keep=dropout_input_p_keep, hidden_p_keep=dropout_hidden_p_keep)
    T_f = model.T_f

    # --- print info ---
    print("Vocab size:", vocab_size)
    print(f"proportion ensemble disconnected approx: {(1-dropout_hidden_p_keep)**sum(model.used_node_idx[2]):.0e}")
    print("Receptive field:", T_f)
    print("Parameters:", sum([w.size for w in model.weights]) + sum([b.size for b in model.biases]))
    print(f"Expected number of total backprop evals: {batch_size * (len(text)-T_f) * max_epochs:.2e}")
    # (depth-2)*(kernel_size*hidden_size**2+hidden_size)+2*kernel_size*hidden_size*vocab_size+vocab_size+hidden_size

    # --- training loop ---
    t0 = time.time()
    prev_loss = 1e6
    end = len(text) - batch_size - 1
    for e in range(max_epochs):
        for i in range(T_f, end+1):
            inputs_batch = np.zeros((batch_size, vocab_size, T_f), dtype=np.float32)
            targets_batch = np.zeros((batch_size, vocab_size), dtype=np.float32)
            for b in range(batch_size):
                for t in range(T_f):
                    inputs_batch[b, char_to_idx[text[i - T_f + t + b]], t] = 1
                targets_batch[b, char_to_idx[text[i + b]]] = 1
            if i == end:
                loss = model.train_minibatch(inputs_batch, targets_batch,
                                             rho_1=rho_1, rho_2=rho_2,
                                             learning_rate=learning_rate, return_loss=True)
            else:
                model.train_minibatch(inputs_batch, targets_batch,
                                      rho_1=rho_1, rho_2=rho_2,
                                      learning_rate=learning_rate)

        print(f"Epoch {e+1}/{max_epochs}, Loss (on final batch): {loss:.4f} "
              f"(i.e. avg. prob. assigned: {np.exp(-loss):.4f}) "
              f"time elapsed: {time.time() - t0:.0f}s eta: {(time.time() - t0)/(e+1)*(max_epochs - e - 1)/60:.1f}m")

        # --- generate text ---
        if len(seed_text) < T_f:
            test = " " * (T_f - len(seed_text)) + seed_text
        else:
            print("Error: seed_text length greater than receptive field size.")
        for i in range(generation_length):
            input_seq = np.zeros((vocab_size, T_f), dtype=np.float32)
            for t in range(T_f):
                input_seq[char_to_idx[test[-T_f + t]], t] = 1
            output_probs = model.forward_pass(input_seq, do_dropout=False)
            next_char_idx = np.random.choice(range(vocab_size), p=output_probs.ravel())
            test += idx_to_char[next_char_idx]

        print("Generated text:")
        print(seed_text)

        # Stopped learning?
        if abs(prev_loss - loss)/max(loss, prev_loss, 1e-8) < early_stop_rel_tol:
            print("Converged, stopping training")
            break
        prev_loss = loss


