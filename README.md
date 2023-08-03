# Physics-informed neural networks

This repository contains Jupyter notebooks for implementing physics-informed neural networks (PINNs) [see Wikipedia article here](https://en.wikipedia.org/wiki/Physics-informed_neural_networks). PyTorch is the deep learning framework of choice here, and the [Burgers' equation](https://en.wikipedia.org/wiki/Burgers%27_equation), a one-dimensional dynamical equation is used for benchmarking. This was a project for ECE 283 Machine learning at UC Santa Barbara during the Spring quarter of 2022. 


## Nontechnical description of PINNs

A large number of problems in science and engineering are expressed in terms of partial differential equations, which govern the spatiotemporal dynamics of the system (e.g. water flow, solid deformation, quantum wave function).
However, obtaining solutions to these equations in practical settings is generally challenging and expensive. 
PINNs combine the power of neural networks (which are great at pattern recognition and approximation) with our knowledge of the underlying physical laws to solve complex partial differential equations.
The basic idea behind PINN is to train a neural network to learn the underlying physics governing the system by observing its behavior at different points in time and space. We provide the neural network with data from the physical system, like measurements or observations of what the system looks like at a certain time at certain locations. We also include the known laws of physics as constraints during the training process.
Once the PINN is trained, you can use it to predict the behavior of the physical system at any point in time and space, even in regions where you might not have direct measurements. Essentially, it learns the patterns in the data and the underlying physics to provide predictions. The main advantages it has over traditional numerical methods is that it can take sparse and noisy measurement data as inputs, and that it can handle complex geometry and systems in high dimension.



## Technical References
The technical aspects of PINN can be found in [this paper](https://arxiv.org/abs/1711.10561). This repository also contains implementation of models with three separate neural networks, as suggested in [this paper](https://arxiv.org/abs/1711.06464).

## Acknoledgement
ECE 283 was taught by Prof. Upamanyu Madhow who provided insightful suggestions. My collaborators were Savya Tanikella and Max Linnander. My main contribution was implementing and testing the three neural network architecture to handle sparse data and extreme parameter regimes.