# Exercise 6.4 (FedProx Implementation for MNIST)

This repository contains an implementation of FedProx (Federated Proximal) algorithm for distributed machine learning on the MNIST dataset. FedProx is an extension of FedAvg that addresses the challenges of heterogeneous federated networks by adding a proximal term to the local objective.

## Usage

### Basic Execution:

```bash
python exercise_6.4.py 
```

## Overview

The implementation includes:
- A CNN model architecture for MNIST classification
- Non-IID data partitioning using Dirichlet distribution
- FedProx algorithm with configurable proximal term (μ)
- Visualization of training results

## Configuration Parameters

The main configuration parameters are:

```python
NUM_CLIENTS = 20        # Number of participating clients
COMM_ROUNDS = 100       # Number of communication rounds
LOCAL_EPOCHS = 5        # Number of local training epochs
BATCH_SIZE = 32         # Batch size for training
MU_VALUES = [0, 0.01, 0.1]  # Proximal term coefficients to test
ALPHA = 0.5             # Dirichlet concentration parameter for non-IID partitioning
```

## Key Components

### Model Architecture

A convolutional neural network with:
- Two convolutional layers (32 and 64 filters)
- Two fully connected layers (512 neurons and 10 output classes)
- ReLU activation and max pooling

### Data Partitioning

The implementation uses a Dirichlet distribution (parameterized by α) to create non-IID data partitions among clients. Lower α values create more heterogeneous distributions.

### FedProx Algorithm

FedProx extends FedAvg by adding a proximal term to the local objective:

```
L_i(w; w_t) = F_i(w) + (μ/2)||w - w_t||^2
```

Where:
- F_i(w) is the original loss function of client i
- w_t is the global model parameters
- μ is the proximal term coefficient

### Client Selection

In each round, 10% of clients are randomly selected to participate in training.

## Running the Experiments

The code will:
1. Initialize a global model
2. Partition MNIST data among clients in a non-IID fashion
3. Run federated learning for each μ value
4. Track and visualize test accuracy over communication rounds

## Results

The implementation generates a plot showing test accuracy over communication rounds for different μ values. This helps visualize how the proximal term affects convergence and final model performance.

- μ = 0: Equivalent to FedAvg
- μ > 0: FedProx with varying degrees of regularization

The results are saved as 'fedprox_results.png' in the current directory.

## References

Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020). Federated Optimization in Heterogeneous Networks. MLSys 2020.
