# Physics-informed neural networks

This repository contains Jupyter notebooks for implementing physics-informed neural networks [PINNs](https://en.wikipedia.org/wiki/Physics-informed_neural_networks). PyTorch is the deep learning framework of choice, and the [Burgers' equation](https://en.wikipedia.org/wiki/Burgers%27_equation), is used for benchmarking. This was a project for ECE 283 Machine learning at UC Santa Barbara during the Spring quarter of 2022. 


## Nontechnical description of PINNs

A large number of problems in science and engineering are expressed in terms of partial differential equations, which are equations that govern the spatiotemporal dynamics of the system (e.g. water flow, solid deformation, quantum wave function).
However, obtaining solutions to these equations in practical settings is challenging and expensive. 
PINNs combine the power of neural networks (which are great at pattern recognition and approximation) with our knowledge of the underlying physical laws to solve complex physics problems.
The basic idea behind PINN is to train a neural network to learn the underlying physics governing the system by observing its behavior at different points in time and space. We provide the neural network with data from the physical system, like measurements of what the system looks like at a certain time at certain locations. Crucially, we also include the known laws of physics as constraints during the training process.
Once the PINN is trained, we can use it to predict the behavior of the physical system at any point in time and space, even in regions where we might not have direct measurements. Essentially, it learns the patterns in the data and the underlying physics to provide predictions. The main advantages it has over traditional numerical methods is that it can take sparse and noisy measurement data as inputs, and that it can handle complex geometries and systems in high dimension.



## Technical References
The technical aspects of PINN can be found in [this paper](https://arxiv.org/abs/1711.10561). This repository also contains implementation of PINNs with three separate neural networks, as suggested in [this paper](https://arxiv.org/abs/1711.06464).

## Acknowledgements
ECE 283 was taught by Prof. Upamanyu Madhow who provided insightful suggestions. My collaborators were Savya Tanikella and Max Linnander. My main contribution was implementing and testing the three neural network architecture to handle sparse data and extreme parameter regimes.