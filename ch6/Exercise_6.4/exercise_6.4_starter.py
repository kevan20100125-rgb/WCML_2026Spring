
"""
Exercise 6.4: FedProx Implementation for MNIST Classification

This script simulates federated learning on a non-i.i.d. MNIST dataset
(Dirichlet split, α=0.5). It compares the convergence behavior across
different proximal-term coefficients (μ ∈ {0, 0.01, 0.1}).

TODO:
Complete the `local_train` function. You need to combine the standard
NLLLoss with the FedProx proximal penalty:
    Loss = NLLLoss + (μ / 2) * ||w - w_t||^2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import copy
from torch.utils.data import DataLoader, Dataset, Subset


# ────────────────────────  Experiment Configuration  ───────────────────────── #
NUM_CLIENTS   = 20              # Total number of simulated clients
COMM_ROUNDS   = 100             # Global communication (aggregation) rounds
LOCAL_EPOCHS  = 5               # Local training epochs per round
BATCH_SIZE    = 64              # Mini-batch size for client training
ALPHA         = 0.5             # Dirichlet concentration parameter (smaller ⇒ more skew)
MU_VALUES     = [0, 0.01, 0.1]  # Proximal-term coefficients to test
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ────────────────────────  Model Definition  ──────────────────────────────── #
class MNISTCNN(nn.Module):
    """2-conv CNN for MNIST classification (10 classes)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1   = nn.Linear(320, 50)
        self.fc2   = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)  # Log-probs for NLL loss



# ────────────────────────  Data Preparation  ──────────────────────────────── #
def dirichlet_split(dataset, num_clients, alpha):
    """
    Partition `dataset` into `num_clients` subsets using a Dirichlet distribution.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to partition.
        num_clients (int): Number of client subsets to create.
        alpha (float): Dirichlet concentration parameter; smaller ⇒ more non-IID.

    Returns:
        list[Subset]: List of `num_clients` PyTorch `Subset`s.
    """
    labels       = np.array(dataset.targets)
    all_indices  = np.arange(len(dataset))
    class_indices = [all_indices[labels == c] for c in range(10)]

    client_indices = [[] for _ in range(num_clients)]
    for c_idx in class_indices:
        np.random.shuffle(c_idx)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        splits = (np.cumsum(proportions) * len(c_idx)).astype(int)[:-1]
        for client_id, idx_split in enumerate(np.split(c_idx, splits)):
            client_indices[client_id].extend(idx_split)

    return [Subset(dataset, idxs) for idxs in client_indices]


# ────────────────────────  Local Training (TODO)  ────────────────────────── #
def local_train(model, global_model, loader, mu):
    """
    Perform FedProx local training on one client.

    TODO:
        • Compute the standard NLL loss on the mini-batch.
        • Add FedProx’s proximal penalty: (μ / 2) * Σ ||w – w_t||²
        • Back-propagate and update local parameters.

    Args:
        model (nn.Module): Client model (initialised with global weights)
        global_model (nn.Module): Current global model (needed for proximal term)
        loader (DataLoader): The client’s data loader
        mu (float): Proximal-term coefficient
    """
    model.train()
    opt      = optim.SGD(model.parameters(), lr=0.01)
    loss_fn  = nn.NLLLoss()

    for _ in range(LOCAL_EPOCHS):
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)

            # ─── YOUR CODE HERE ──────────────────────────────────────────── #
            # 1. Zero existing gradients
            # 2. Forward pass
            # 3. Compute FedProx loss (NLL + proximal term)
            # 4. Backward + step
            # Remove "pass" and "raise NotImplementedError" statements once implemented.
            pass
            # ─────────────────────────────────────────────────────────────── #
            pass

    raise NotImplementedError


# ────────────────────────  Evaluation  ─────────────────────────────────────── #
def evaluate(model, test_loader):
    """Return test-set accuracy (%) for `model` given a loader."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            pred = model(data).argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return 100.0 * correct / len(test_loader.dataset)


# ────────────────────────  FedProx Experiment  ────────────────────────────── #
def run_fedprox_experiment(mu):
    """
    Execute FedProx training for a given proximal coefficient `mu`.

    Args:
        mu (float): Proximal-term coefficient.

    Returns:
        list[float]: Test accuracy per communication round.
    """
    # — Data preparation moved inside the function —
    transform   = transforms.Compose([transforms.ToTensor()])
    train_data  = datasets.MNIST("./data", train=True,  download=True, transform=transform)
    test_data   = datasets.MNIST("./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=1000)

    # Non-IID client partitions
    client_datasets = dirichlet_split(train_data, NUM_CLIENTS, ALPHA)

    # Global model init
    global_model = MNISTCNN().to(DEVICE)
    accuracy_history = []

    for rnd in range(COMM_ROUNDS):
        # Local training
        local_models = []
        for cid in range(NUM_CLIENTS):
            client_model = copy.deepcopy(global_model)
            loader = DataLoader(client_datasets[cid], batch_size=BATCH_SIZE, shuffle=True)
            local_train(client_model, global_model, loader, mu)
            local_models.append(client_model)

        # Global aggregation
        gdict = global_model.state_dict()
        total_samples = sum(len(client_datasets[cid]) for cid in range(NUM_CLIENTS))
        for k in gdict:
            weighted_sum = torch.zeros_like(gdict[k], dtype=torch.float)
            for cid, local_model in enumerate(local_models):
                weight = len(client_datasets[cid]) / total_samples
                weighted_sum += local_model.state_dict()[k].float() * weight
            gdict[k] = weighted_sum
        global_model.load_state_dict(gdict)

        # Evaluation
        acc = evaluate(global_model, test_loader)
        accuracy_history.append(acc)
        print(f"Round {rnd + 1:02d}, μ={mu}: Acc {acc:.2f}%")

    return accuracy_history

# ────────────────────────  Main Execution  ────────────────────────────────── #
if __name__ == "__main__":
    results = {}
    for mu in MU_VALUES:
        print(f"\n=== Experiment with μ = {mu} ===")
        results[mu] = run_fedprox_experiment(mu)

    # Visualization
    plt.figure(figsize=(10, 5))
    for mu, acc_hist in results.items():
        plt.plot(acc_hist, label=f"μ = {mu}")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Test Accuracy (%)")
    plt.title("FedProx Performance across Proximal Terms")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fedprox_mu_impact.png", dpi=300)
    plt.show()
