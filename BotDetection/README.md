# BotDetection 

The reddit bot that detects other bots.

## Introduction

This repository hosts an Artifical Neural Network (ANN) that can be used to detect comment spambots on Reddit. Using 9 key statistics and over 400 examples, the bot is able to successfully classify most users with high confidence. The Neural Network itself is a basic feed-forward, with hyperparameters as follows: 

- 0.15 learning rate

- 9 inputs, 64 hidden units, 1 output

- sigmoid activation function

- 10,000 training iterations

All of these can be adjusted easily in the source code (note that if you change the activation function, you will also have to adjust the calculations for gradient descent). The code is heavily commented and informed pull requests are welcome -- I am definitely still learning myself.

## Usage and Installation

You will need to install praw, nltk, and scipy if they are not already installed. When installing nltk, be sure to download the relevant material via `nltk.download`.

Once this is done, clone the repository and run `detector.py` using Python 3. If you get any errors about Reddit authentication, you will need to go into `utils.py` and supply your credentials where the PRAW object is authenticated (line 6). Instructions on how to use OAuth can be found [here](http://praw.readthedocs.io/en/latest/getting_started/authentication.html).


You can now open `detector.py`; run the `isABot` function with any username, and you will get back a number between 0 and 1 that represents how much the Neural Network thinks the user is a bot.

## Retraining and Contributing

The current code is far from optimized. If you would like to add more statistics for the bot to consider, you can do so in the `utils.py` file, which hosts the main `User`. Implementing additional features will require you to update the `User` class to reflect them, as well as re-generate the `data.csv` file and increase the number of inputs the network accepts. 

If you simply want to adjust the hyperparameters, then no data gathering or additional adjustments are necessary.
