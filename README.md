Run main.py to initiate training of the model.

## Comments
Primary goal was to experiment with (vanilla) Temporal Convolutional Networks (TCN), in particular test their performance and become more comfortable with the architecture.
Secondary was implementing techniques from Deep Learning by Goodfellow et al. chapters 6 through 8. 
To this end I implemented a TCN from scratch (i.e. using only numpy) and trained it for next-character-prediction on a text file of some (all?) collected pieces of Shakespeare (the idea and data for which are taken from Karpathy's blog: https://karpathy.github.io/2015/05/21/rnn-effectiveness/).
The reason I was interested in TCNs in the first place was the following paper: "Quant GANs: Deep Generation of Financial Time Series" by Wiese et al. (https://arxiv.org/abs/1907.06673), which itself uses TCNs instead of a more conventional choice like LSTM based on a study by Bai et al. in 2018: "An empirical evaluation of generic convolutional and recurrent networks for sequence modeling" (https://arxiv.org/abs/1803.01271) that showed, among other things, TCNs capture longer range dependencies better.

After implementing basic vanilla backprop (with numerical gradient testing) and simple training infrastructure, I experimented with momentum, RMSprop, Adam, and dropout. 
Added gradient clipping because gradients exploded sometimes (I'd like to look into this more, especially as Wiese et al. mention it shouldn't happen in TCN architectures. Might be a bug.)
I also added a residual/skip connection following Bai et al. (but this would make more sense if the next character were generally a small distance away from the previous, but there's no reason to expect that with simple one-hot character embeddings).
It was easy to get both training and validation error to be better than guessing (e.g. the model gives 6x the guessing probability to the correct answer), which would generate something like

> To be, or not to beey ar
> tchith to tof, an, rie in do iurthe in aba s

when prompted "To be, or not to be".
I wasn't able to get any real English within the timeframe I wanted to spend on it, I think due to me not systematically tweaking hyperparameters (in particular the network parameters/shape of the TCN seemed to influence the performance strongly) and not using all available data (which would've taken my or even my uni's computers forever).
