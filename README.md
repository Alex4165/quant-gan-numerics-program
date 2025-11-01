# quant-gan-numerics-program (CURRENTLY WIP)
Code for a project at the University of Bordeaux concerning "Quant GANs: Deep Generation of Financial Time Series" by Wiese et al.

## TCN Shakespeare
I wanted to experiment with (vanilla) Temporal Convolutional Networks (TCN), in particular test their performance and become more comfortable with its workings. To do this I will implement a TCN from scratch (i.e. using only numpy) and train it for next-character-prediction on a text file of some colleceted pieces of Shakespeare (the idea and data for which are taken from Karpathy's blog: https://karpathy.github.io/2015/05/21/rnn-effectiveness/).

Currently uses a vanilla momentum backprop. Next I'll add dropout.

Here is a string the model (which overfitted on a subset of the dataset) produced when prompted with "To be, or not to be" (which is not in the dataset):
To be, or not to ber thib res: nor aff rearmi sa az nunond treair thit
 tsrmh frisver ron n.a
gannnoo; vits eir them we gf hor
rntsl Saicnann toanies us giaan.

nlous dirglef thawd gflF re ass mi ncefnreare shln
zgifs, 
