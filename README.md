# An Experiment of Deep Q-Learning with Robocode

## Introduction
This project aims to perform different experiments to have a better practical understanding about the Reinforcement Learning (RL) using Q-Learning and using Function Approximation with Backpropagation Neural Network.
The implementation has two primary parts: backpropagation neural network, and reinforcement learning to build a tank on Robocode platform.

### Part 1: backpropagation neural network

The neural network has been trained by backpropagation algorithm with different activation function options. 
The network has multiple inputs, one hidden layers with multiple neurons, and multiple outputs as shown on image blow.
More details can be found on my post ["backpropagation algorithm"](https://phuongle.github.io/2021/01/backpropagation-neural-network.html).

<p align="center">
<img src="./readme/nn-multi.png" width="650" alt=""/>
</p>

It has been fully unit tested with XOR and BIPOLAR XOR presentation. It can also be manually tested with XorNeuralNetRunner.

### Part 2: Reinforcement Learning to build a robot tank

This part is to study how to use Reinforcement Learning algorithm to train a robot tank to fight against the enemy Tracker 
on [Robocode platform](https://robocode.sourceforge.io/), in which this implementation has two robots
- Robot "QLearningRobo": is trained by RL using Q-Learning algorithm (an extension of Temporal-Difference learning TD(0)).
- Robot "QNetworkRobo": is trained by deep RL using Q-Function approximation with Neural Network.

The overall design can be found on my post ["How to implement a Deep Reinforcement Learning to build a Robot tank"](https://phuongle.github.io/2021/01/deep-reinforcement-learning.html)


The performance of QLearningRobo is as on the image below.
<p align="center">
   <img src="./readme/Q-Learning-performance.png" width="500" alt=""/>
</p>
The winning rate shows the different performance between on-policy and off-policy algorithm.

The performance of QNetworkRobo is as on the image below.
<p align="center">
   <img src="./readme/Q-Function-Approximation-performance.png" width="500" alt=""/>
</p>
The winning rate can reach above 90% after around 3000 rounds.

## Implementation Objectives

This project has following learning objectives. Most of them were implemented, and tested.

### Neural Network

+ [How to train a neural network by using Backpropagation algorithm?](https://phuongle.github.io/2021/01/backpropagation-neural-network.html)
- How to measure a neural network's performance?
- Bias and Weights

### Reinforcement Learning

- [How to implement Deep Reinforcement Learning to build a Robot tank?](https://phuongle.github.io/2021/01/deep-reinforcement-learning.html)
- How to implement on-policy vs off-policy algorithm (aka Q-Learning vs SARSA)?, and why off-policy is more preferable than on-policy algorithm?
- How to measure a Reinforcement Learning performance?
- How to implement Q-Function Approximation using Backpropagation Neural Network?
- How to implement State-Action space less reduction in Q-Function Approximation?
- How to implement Memory Replay in Deep Q-Network training?

## Setup
Please refer to my article [How to implement Deep Reinforcement Learning to build a Robot tank?](https://phuongle.github.io/2021/01/deep-reinforcement-learning.html) for more details.



