import time

import numpy as np
from matplotlib import pyplot as plt

from tcn import VanillaTCN
import pickle


def get_training_batch(idx, data, batch_length):
    i_batch = np.zeros((batch_length, vocab_size, T_f), dtype=np.float32)
    t_batch = np.zeros((batch_length, vocab_size), dtype=np.float32)
    for b in range(batch_length):
        for t in range(T_f):
            i_batch[b, char_to_idx[data[idx - T_f + t + b]], t] = 1
        t_batch[b, char_to_idx[data[idx + b]]] = 1
    return i_batch, t_batch


if __name__ == "__main__":
    # --- hyperparameters ---
    # training
    learning_rate = 0.0001
    rho_1, rho_2 = 0.9, 0.999  # Adam params. Set rho_2=0 for bias corrected momentum only
    dropout_input_p_keep = 1  # doesn't work right now. Leave at 1
    dropout_hidden_p_keep = 0.5

    max_epochs = 100
    validation_size = 1000
    data_index = 50000
    batch_size = 100
    # batch_size = np.ceil(1e-2 * data_index).astype(int) if data_index > 0 else 100
    # number of backprop steps = O(data_size * epochs)

    # model
    copies = 2  # copies of kernel-dilation list, so final_depth = depth * copies, see below
    depth = 3  # we use less depth to decrease gradient explosion/vanishing
    kernel_size = 5  # larger kernels effectively allow the network to memorize words
    dilation_size = 2  # dilation still allows large receptive field
    hidden_size = 70  # (larger than) vocab size is a natural choice
    # num parameters = O(copies * depth * kernel_size * hidden_size^2)

    # generation
    seed_text = "To be, or not to be"
    generation_length = 1000
    num_plots_shown = 10

    # --- data ---
    file = "input.txt"
    with open(file, "r") as f:
        text = f.read()
    chars = ''.join(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)
    training_data = text[:data_index]
    validation_data = text[data_index:data_index + validation_size]

    # --- initialize model ---
    dilations = copies * [dilation_size ** i for i in range(depth - 1, -1, -1)]
    kernel_sizes = copies * [kernel_size for _ in range(depth)]
    hidden_sizes = [vocab_size] + [hidden_size for _ in range(copies * depth - 1)]
    model = VanillaTCN(input_size=vocab_size, dilations=dilations,
                       kernel_sizes=kernel_sizes, hidden_sizes=hidden_sizes,
                       input_p_keep=dropout_input_p_keep, hidden_p_keep=dropout_hidden_p_keep)
    T_f = model.T_f

    # --- print info ---
    print("Vocab size:", vocab_size)
    print(f"Total available data size: {len(text):.2e} "
          f"(using {data_index + validation_size if data_index > 0 else len(text) :.2e})")
    if depth*copies > 3:
        print(f"proportion dropout disconnected approx: {(1 - dropout_hidden_p_keep) ** sum(model.used_node_idx[3]):.0e}")
    print("Receptive field:", T_f)
    print(f"Parameters: {sum([w.size for w in model.weights]) + sum([b.size for b in model.biases]):.2e}")
    print(f"Expected number of total backprop evals: {(len(training_data) - T_f) * max_epochs:.2e}")
    print("------------------------------------------------------------------------------")

    # --- training loop ---
    prev_loss = 1e6
    end_idx = len(training_data) - batch_size
    if end_idx + 1 <= T_f:
        raise ValueError("Dataset length too short")
    num_batches = (len(training_data) - T_f) // batch_size
    if num_batches * batch_size + T_f < len(training_data):
        print(f"Warning not using last {len(training_data) - (num_batches * batch_size + T_f)} "
              f"characters in training data")
        # TODO: could add to validation set instead
    t0 = time.time()
    for e in range(max_epochs):
        # --- training epoch ---
        training_losses = np.zeros(num_batches)
        for i in range(num_batches):
            inputs_batch, targets_batch = get_training_batch(T_f + i * batch_size, training_data, batch_size)
            l = model.train_minibatch(inputs_batch, targets_batch,
                                                             rho_1=rho_1, rho_2=rho_2,
                                                             learning_rate=learning_rate)
            if e == 0 and i < 10:
                print(f"initial training losses of batch {i+1}: {l:.4f}")
            elif e == 0 and i == 10:
                print(f"ETA: {(time.time() - t0) / 10 * num_batches * max_epochs / 60:.1f}m")
            if l > 30:
                print(f"Warning: Error blew up. Reduce learning rate from {learning_rate}")
                break
            training_losses[i] = l
        if e % (max_epochs // num_plots_shown) == 0:
            plt.plot(training_losses)
            plt.title(f"Epoch {e + 1} training losses")
            plt.xlabel("Batch")
            plt.ylabel("Loss")
            plt.axhline(y=np.log(vocab_size), color='r', linestyle='--', label='random guess loss')
            plt.show()

        # --- validation ---
        inputs_batch, targets_batch = get_training_batch(T_f, validation_data, len(validation_data) - T_f)
        validation_loss = model.train_minibatch(inputs_batch, targets_batch, update_weights=False)
        print(f"Epoch {e + 1}/{max_epochs} | avg training loss {np.mean(training_losses):.4f} | "
              f"validation loss {validation_loss:.4f} | "
              f"eta {(time.time() - t0) / (e + 1) * (max_epochs - e - 1) / 60:.1f}m")

        # --- generate text ---
        if (e + 1) % (max_epochs // 10) == 0 or e == max_epochs - 1:
            if len(seed_text) < T_f:
                test = " " * (T_f - len(seed_text)) + seed_text
            else:
                test = seed_text[-T_f:]
                print("Warning: seed_text length greater than receptive field size.")
            for i in range(generation_length):
                input_seq = np.zeros((vocab_size, T_f), dtype=np.float32)
                for t in range(T_f):
                    input_seq[char_to_idx[test[-T_f + t]], t] = 1
                output_probs = model.forward_pass(input_seq, do_dropout=False)
                next_char_idx = np.random.choice(range(vocab_size), p=output_probs.ravel())
                test += idx_to_char[next_char_idx]

            print("Generated text:")
            print(test, '\n')

    print("Training complete. Saving model...")
    hyperparams = {
        "learning_rate": learning_rate,
        "rho_1": rho_1,
        "rho_2": rho_2,
        "dropout_input_p_keep": dropout_input_p_keep,
        "dropout_hidden_p_keep": dropout_hidden_p_keep,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "validation_size": validation_size,
        "copies": copies,
        "depth": depth,
        "kernel_size": kernel_size,
        "dilation_size": dilation_size,
        "hidden_size": hidden_size,
        "seed_text": seed_text,
        "generation_length": generation_length,
        "vocab_size": vocab_size,
        "T_f": T_f,
    }

    state = {
        "weights": [w.copy() for w in model.weights],
        "biases": [b.copy() for b in model.biases],
        "hyperparams": hyperparams,
    }

    with open("model_checkpoint.pkl", "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done, thanks, that data was delicious")
