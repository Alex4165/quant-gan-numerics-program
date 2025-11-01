import numpy as np
from tcn import VanillaTCN

if __name__ == "__main__":
    # --- hyperparameters ---
    # training
    momentum = 0.9
    learning_rate = 0.01
    batch_size = 16
    max_epochs = 20
    early_stop_rel_tol = 5e-3
    data_size = 1000  # limit data size for faster training, -1 for full data

    # model
    depth = 4
    kernel_size = 2
    dilation_size = 2
    hidden_size = 64

    # generation
    seed_text = "To be, or not to be"
    generation_length = 200

    # --- data ---
    file = "input.txt"
    with open(file, "r") as f:
        text = f.read()
    chars = ''.join(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)
    print("Vocab size:", vocab_size)

    text = text[:data_size]

    dilations = [dilation_size**i for i in range(depth-1, -1, -1)]
    kernel_sizes = [kernel_size for _ in range(depth)]
    hidden_sizes = [vocab_size] + [hidden_size for _ in range(depth-1)]
    model = VanillaTCN(input_size=vocab_size, dilations=dilations,
                       kernel_sizes=kernel_sizes, hidden_sizes=hidden_sizes)
    T_f = model.T_f
    print("Receptive field:", T_f)
    print("Parameters:", sum([w.size for w in model.weights]) + sum([b.size for b in model.biases]))

    # --- training loop ---
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
                                         momentum=momentum, learning_rate=learning_rate, return_loss=True)
            else:
                model.train_minibatch(inputs_batch, targets_batch,
                                      momentum=momentum, learning_rate=learning_rate)

        print(f"Epoch {e+1}/{max_epochs}, Loss: {loss:.4f}")
        if abs(prev_loss - loss)/max(loss, prev_loss, 1e-8) < early_stop_rel_tol:
            print("Converged, stopping training")
            break
        prev_loss = loss

    # --- generate text ---
    if len(seed_text) < T_f:
        seed_text = " " * (T_f - len(seed_text)) + seed_text
    else:
        print("Warning: seed_text length greater than receptive field size.")
    for i in range(generation_length):
        input_seq = np.zeros((vocab_size, T_f), dtype=np.float32)
        for t in range(T_f):
            input_seq[char_to_idx[seed_text[-T_f + t]], t] = 1
        output_probs = model.forward_pass(input_seq)
        next_char_idx = np.random.choice(range(vocab_size), p=output_probs.ravel())
        seed_text += idx_to_char[next_char_idx]

    print("Generated text:")
    print(seed_text)


