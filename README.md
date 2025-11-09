# quant-gan-numerics-program (WIP)
Code for a project concerning "Quant GANs: Deep Generation of Financial Time Series" by Wiese et al.

## TCN Shakespeare
I wanted to experiment with (vanilla) Temporal Convolutional Networks (TCN), in particular test their performance and become more comfortable with their workings. To do this I implemented a TCN from scratch (i.e. using only numpy) and trained it for next-character-prediction on a text file of some colleceted pieces of Shakespeare (the idea and data for which are taken from Karpathy's blog: https://karpathy.github.io/2015/05/21/rnn-effectiveness/).

Currently uses a Adam optimization. Will write some notes here on the results soon.

## English GAN
Next, I wanted to experiment with GANs and learn basics of Pytorch. The idea is to use the GAN framework to generate fake English words. Data from https://github.com/dwyl/english-words.
