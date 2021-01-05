# An Experiment of Deep Q-Learning with Robocode

## Introduction
This project aims to perform different experiments to have a better practical understanding about the Reinforcement Learning (RL) using Q-Learning and using Function Approximation with Backpropagation Neural Network.
The implementation has two primary parts: a backpropagation neural network, and a reinforcement learning to build a tank on Robocode platform.

### Part 1: backpropagation neural network

The neural network has been trained by backpropagation algorithm with different activation function options. 
The implemented neural network has multiple inputs, one hidden layers with multiple neurons, and multiple outputs as shown on image blow.
Both hidden and output layer have bias with variant weights on each bias.

<p align="center">
<img src="./readme/nn-multi.png" width="650" alt=""/>
</p>

*Testing*

It has been fully unit tested using XOR and BIPOLAR XOR presentation. The XorNeuralNetRunner is written for experimenting different hyper-parameters manually.

*Implementation*

The implementation is on "src\main\java\backpropagation", and more details is written on my article about ["backpropagation algorithm"](https://phuongle.github.io/2021/01/backpropagation-neural-net.html).


### Part 2: Reinforcement Learning to build a robot tank

*Testing*

This robot tank has been tested against enemy Tracker, 

The performance of using Q-Learning is as on the image below.
<p align="center">
   <img src="./readme/Q-Learning-performance.png" width="650" alt=""/>
</p>
The winning rate shows the different performance between on-policy and off-policy algorithm.

The performance of using Q-Function Approximation is as on the image below.
<p align="center">
   <img src="./readme/Q-Function-Approximation-performance.png" width="650" alt=""/>
</p>
The winning rate can reach above 90% after around 3000 rounds.

*Implementation*

The implementation can be found at "src\main\java\reinforcement". The overall design is on the next section.

## Implementation Objectives

This project has following learning objectives. Most of them were implemented, and tested.

### Neural Network

+ [How to train a neural network by using Backpropagation algorithm?](https://phuongle.github.io/2021/01/backpropagation-neural-net.html)
- How to measure a neural network's performance?
- Bias and Weights

### Reinforcement Learning

- How to implement Reinforcement Learning with Q-Learning algorithm to build a Robot tank in Robocode platform?
- How to implement on-policy vs off-policy algorithm (aka Q-Learning vs SARSA)?, and why off-policy is more preferable than on-policy algorithm?
- How to measure a Reinforcement Learning performance?
- How to implement Q-Function Approximation using Backpropagation Neural Network?
- How to implement State-Action space less reduction in Q-Function Approximation?
- How to implement Memory Replay in Deep Q-Network training?
