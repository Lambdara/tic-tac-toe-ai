# tic-tac-toe-ai
## Introduction

Combining an implementation of neural networks with one of tic-tac-toe, this repository provides code for a self-teaching tic-tac-toe AI based on neural networks.
The results:

![result](/result.png)

On the Y-axis is the winrate of the AI, on the X-axis is the amount of games for which the AI has trained. Graphed are the winrates over time of the AI versus a random move generator, and versus a random legal move generator.

The AI which produced this graph has been tested and found to never lose. Though of course this is no guarantee because the initialization and the learning process are both not completely deterministic.

## Usage

Load the file `main.rkt` in your REPL. You can then use `(go filename)` to start the training process and log the progress to `filename` (this was used to create the data for the graph in this readme) or use `(train-nn number-of-games)` to train the network for a certain `number-of-games`. You can play against the trained AI using `(play my-turn)` where `my-turn` is `'cross` to play as cross, `'circle` to play as circle, or anything else to let the AI play against itself. You can use a different network topology using `set-nn-topology!`, examples that have been tested and found to work well are `(set-nn-topology! 27 81 9)` and `(set-nn-topology! 27 40 40 9)`. Please note that it always has to start with a layer of 27 and end with a layer of 9, because those are the assumed sizes of the input- and output-vectors respectively.
