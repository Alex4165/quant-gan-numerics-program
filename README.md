# quant-gan-numerics-program (CURRENTLY WIP)
Code for a project at the University of Bordeaux concerning "Quant GANs: Deep Generation of Financial Time Series" by Wiese et al.

## TCN Shakespeare
I wanted to experiment with (vanilla) Temporal Convolutional Networks (TCN), in particular test their performance and become more comfortable with their workings. To do this I implemented a TCN from scratch (i.e. using only numpy) and trained it for next-character-prediction on a text file of some colleceted pieces of Shakespeare (the idea and data for which are taken from Karpathy's blog: https://karpathy.github.io/2015/05/21/rnn-effectiveness/).

Currently uses a Adam optimization and dropout. I'll play with the hyperparameters a little and see how good the model can do.
