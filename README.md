# quant-gan-numerics-program (WIP)
Code for a project concerning "Quant GANs: Deep Generation of Financial Time Series" by Wiese et al.

## TCN Shakespeare
I wanted to experiment with (vanilla) Temporal Convolutional Networks (TCN), in particular test their performance and become more comfortable with their workings. To do this I implemented a TCN from scratch (i.e. using only numpy) and trained it for next-character-prediction on a text file of some colleceted pieces of Shakespeare (the idea and data for which are taken from Karpathy's blog: https://karpathy.github.io/2015/05/21/rnn-effectiveness/).

Currently uses a Adam optimization. Will write some notes here on the results soon.
Vocab size: 65
Total available data size: 1.12e+06 (using 5.10e+04)
Receptive field: 29
Parameters: 7.02e+04
Expected number of total backprop evals: 5.00e+06
------------------------------------------------------------------------------
Warning not using last 71 characters in training data
initial training losses of batch 1: 4.3751
initial training losses of batch 2: 4.3129
initial training losses of batch 3: 4.2613
initial training losses of batch 4: 4.4083
initial training losses of batch 5: 4.3955
initial training losses of batch 6: 4.3805
initial training losses of batch 7: 4.4330
initial training losses of batch 8: 4.4229
initial training losses of batch 9: 4.4097
initial training losses of batch 10: 4.4273
ETA: 77.8m
Epoch 1/100 | avg training loss 3.673.6741 | validation loss 3.547141 | validation loss 3.5471 | eta 101.7m
Epoch 2/100 | avg training loss 3.4472 | validation loss 3.5366 | eta 100.0m
Epoch 3/100 | avg training loss 3.4342 | validation loss 3.5250 | eta 98.8m
Epoch 4/100 | avg training loss 3.4199 | validation loss 3.5131 | eta 99.5m
Epoch 5/100 | avg training loss 3.4031 | validation loss 3.4972 | eta 98.5m
Epoch 6/100 | avg training loss 3.3784 | validation loss 3.4717 | eta 97.0m
Epoch 7/100 | avg training loss 3.3478 | validation loss 3.4386 | eta 95.6m
Epoch 8/100 | avg training loss 3.3115 | validation loss 3.3994 | eta 95.9m
Epoch 9/100 | avg training loss 3.2701 | validation loss 3.3558 | eta 94.4m
Epoch 10/100 | avg training loss 3.2248 | validation loss 3.3094 | eta 93.2m
Generated text:
          To be, or not to beaa nfeeo ewwiid b nesieur,eoas . eo 
 eefet shn ne 

Epoch 11/100 | avg training loss 3.1770 | validation loss 3.2598 | eta 91.8m
Epoch 12/100 | avg training loss 3.1282 | validation loss 3.2092 | eta 90.5m
Epoch 13/100 | avg training loss 3.0800 | validation loss 3.1597 | eta 89.4m
Epoch 14/100 | avg training loss 3.0338 | validation loss 3.1115 | eta 88.6m
Epoch 15/100 | avg training loss 2.9900 | validation loss 3.0659 | eta 87.6m
Epoch 16/100 | avg training loss 2.9488 | validation loss 3.0229 | eta 86.5m
Epoch 17/100 | avg training loss 2.9102 | validation loss 2.9825 | eta 85.4m
Epoch 18/100 | avg training loss 2.8743 | validation loss 2.9450 | eta 84.2m
Epoch 19/100 | avg training loss 2.8411 | validation loss 2.9100 | eta 83.0m
Epoch 20/100 | avg training loss 2.8102 | validation loss 2.8777 | eta 82.0m
Generated text:
          To be, or not to beres the the c thev mrsivP the  ahe sasr bne dorhaA 

Epoch 21/100 | avg training loss 2.7815 | validation loss 2.8481 | eta 80.8m
Epoch 22/100 | avg training loss 2.7549 | validation loss 2.8204 | eta 79.6m
Epoch 23/100 | avg training loss 2.7302 | validation loss 2.7941 | eta 78.5m
Epoch 24/100 | avg training loss 2.7070 | validation loss 2.7695 | eta 77.7m
Epoch 25/100 | avg training loss 2.6852 | validation loss 2.7465 | eta 76.6m
Epoch 26/100 | avg training loss 2.6647 | validation loss 2.7248 | eta 75.6m
Epoch 27/100 | avg training loss 2.6453 | validation loss 2.7046 | eta 74.5m
Epoch 28/100 | avg training loss 2.6271 | validation loss 2.6857 | eta 73.6m
Epoch 29/100 | avg training loss 2.6098 | validation loss 2.6678 | eta 72.6m
Epoch 30/100 | avg training loss 2.5933 | validation loss 2.6509 | eta 71.5m
Generated text:
          To be, or not to berthing dnwipe
ciugeswotit
 o uoh frrfet, ho aans r 

Epoch 31/100 | avg training loss 2.5777 | validation loss 2.6350 | eta 70.5m
Epoch 32/100 | avg training loss 2.5628 | validation loss 2.6199 | eta 69.4m
Epoch 33/100 | avg training loss 2.5485 | validation loss 2.6056 | eta 68.3m
Epoch 34/100 | avg training loss 2.5348 | validation loss 2.5926 | eta 67.2m
Epoch 35/100 | avg training loss 2.5226 | validation loss 2.5846 | eta 66.1m
Epoch 36/100 | avg training loss 2.5114 | validation loss 2.5713 | eta 65.0m
Epoch 37/100 | avg training loss 2.4992 | validation loss 2.5589 | eta 64.0m
Epoch 38/100 | avg training loss 2.4873 | validation loss 2.5474 | eta 63.0m
Epoch 39/100 | avg training loss 2.4761 | validation loss 2.5365 | eta 61.9m
Epoch 40/100 | avg training loss 2.4653 | validation loss 2.5264 | eta 60.9m
Generated text:
          To be, or not to bees, whot ace tur?
is: Iire

not Iliim thanrit, Th  

Epoch 41/100 | avg training loss 2.4550 | validation loss 2.5169 | eta 59.8m
Epoch 42/100 | avg training loss 2.4451 | validation loss 2.5079 | eta 58.8m
Epoch 43/100 | avg training loss 2.4357 | validation loss 2.4994 | eta 57.7m
Epoch 44/100 | avg training loss 2.4266 | validation loss 2.4913 | eta 56.7m
Epoch 45/100 | avg training loss 2.4179 | validation loss 2.4836 | eta 55.7m
Epoch 46/100 | avg training loss 2.4095 | validation loss 2.4762 | eta 54.6m
Epoch 47/100 | avg training loss 2.4014 | validation loss 2.4693 | eta 53.6m
Epoch 48/100 | avg training loss 2.3936 | validation loss 2.4627 | eta 52.6m
Epoch 49/100 | avg training loss 2.3860 | validation loss 2.4563 | eta 51.5m
Epoch 50/100 | avg training loss 2.3787 | validation loss 2.4502 | eta 50.6m
Generated text:
          To be, or not to beret'n to mert vot
eph. of bpvet od whn y
f thewers 

Epoch 51/100 | avg training loss 2.3717 | validation loss 2.4444 | eta 49.5m
Epoch 52/100 | avg training loss 2.3648 | validation loss 2.4388 | eta 48.5m
Epoch 53/100 | avg training loss 2.3582 | validation loss 2.4333 | eta 47.5m
Epoch 54/100 | avg training loss 2.3518 | validation loss 2.4281 | eta 46.6m
Epoch 55/100 | avg training loss 2.3456 | validation loss 2.4230 | eta 45.6m
Epoch 56/100 | avg training loss 2.3396 | validation loss 2.4181 | eta 44.5m
Epoch 57/100 | avg training loss 2.3338 | validation loss 2.4134 | eta 43.5m
Epoch 58/100 | avg training loss 2.3282 | validation loss 2.4090 | eta 42.4m
Epoch 59/100 | avg training loss 2.3228 | validation loss 2.4047 | eta 41.4m
Epoch 60/100 | avg training loss 2.3175 | validation loss 2.4006 | eta 40.4m
Generated text:
          To be, or not to beey ar
tchith to tof, an, rie in do iurthe in aba s 

Epoch 61/100 | avg training loss 2.3123 | validation loss 2.3965 | eta 39.4m
Epoch 62/100 | avg training loss 2.3073 | validation loss 2.3927 | eta 38.4m
Epoch 63/100 | avg training loss 2.3024 | validation loss 2.3890 | eta 37.4m

## English GAN
Next, I wanted to experiment with GANs and learn basics of Pytorch. The idea is to use the GAN framework to generate fake English words. Data from https://github.com/dwyl/english-words.
