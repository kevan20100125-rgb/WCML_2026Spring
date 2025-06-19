# Exercise 6.4: FedProx Implementation for MNIST Classification

This repository provides an implementation of the **FedProx** (Federated Proximal) algorithm for distributed machine learning on the MNIST dataset. FedProx is an extension of the **FedAvg** algorithm, which addresses the challenges of heterogeneous federated networks by adding a proximal term to the local objective function.

## Table of Contents

- [Overview](#overview)
- [Configuration Parameters](#configuration-parameters)
- [Key Components](#key-components)
  - [Model Architecture](#model-architecture)
  - [Data Partitioning](#data-partitioning)
  - [FedProx Algorithm](#fedprox-algorithm)
  - [Client Selection](#client-selection)
- [Running the Experiments](#running-the-experiments)
- [Results](#results)

## Overview

This implementation includes the following key features:

- A CNN model for MNIST classification.
- Non-IID data partitioning using the Dirichlet distribution to simulate real-world federated learning scenarios.
- An implementation of the FedProx algorithm with a configurable proximal term (μ).
- Visualization of training results to assess the model's performance over communication rounds.

## Configuration Parameters

The following configuration parameters are used for training:

```python
NUM_CLIENTS = 20            # Number of participating clients
COMM_ROUNDS = 100           # Number of communication rounds
LOCAL_EPOCHS = 5            # Number of local training epochs per client
BATCH_SIZE = 64             # Batch size for training
MU_VALUES = [0, 0.01, 0.1]  # Proximal term coefficients to test
ALPHA = 0.5                 # Dirichlet concentration parameter for non-IID partitioning
```

## Key Components
### Model Architecture
The model used for MNIST classification is a CNN with the following architecture:
- Two convolutional layers with 10 and 20 filters, respectively.
- Two fully connected layers with 50 neurons and 10 output classes (for the 10 MNIST digits).
- ReLU activation functions and max-pooling layers.

### Data Partitioning
To simulate non-IID data distributions, the implementation uses a Dirichlet distribution with a concentration parameter (α). This creates heterogeneous data splits among the clients. Lower values of α result in more heterogeneous (non-IID) distributions.

### FedProx Algorithm
FedProx extends the FedAvg algorithm by adding a proximal term to the local objective function, which helps address issues arising from client heterogeneity. The local objective is defined as:

$$
L_i(w; w_t) = F_i(w) + \frac{\mu}{2}\left\\| w - w_t \right\\|^2,
$$

where:
- $F_i(w)$ is the original loss function of client i.
- $w_t$ is the global model parameters.
- $\mu$ is the proximal term coefficient.

### Client Selection
In this implementation, all clients participate in each communication round. The client selection strategy can be extended as needed for specific use cases.

## Running the Experiments
To run the experiments, follow these steps:

1. Initialize a global model: A global model is created at the start of the experiment.
2. Partition the MNIST data: The MNIST dataset is partitioned among clients in a non-IID manner using the Dirichlet distribution.
3. Run federated learning: Federated learning is executed for each value of the proximal term coefficient (μ).
4. Track and visualize accuracy: The model's test accuracy is tracked and visualized over communication rounds.

To execute the code, run the following command:

```python
python exercise_6.4.py
```

## Results
After running the experiments, the implementation generates a plot that shows test accuracy over communication rounds for different values of μ. This visualization helps to assess the impact of the proximal term on the convergence and final performance of the model.

- μ = 0: Equivalent to the FedAvg algorithm (no proximal term).
- μ > 0: FedProx with varying degrees of regularization applied through the proximal term.

The results are saved as an image file named 'fedprox_results.png' in the current directory.

