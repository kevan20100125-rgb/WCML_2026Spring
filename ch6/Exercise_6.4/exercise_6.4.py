import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from collections import defaultdict
import copy

# Set random seeds for reproducibility
torch.manual_seed(123)
np.random.seed(123)

# Configuration
NUM_CLIENTS = 20
COMM_ROUNDS = 100
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
MU_VALUES = [0, 0.01, 0.1]  # Proximal term coefficients
ALPHA = 0.5  # Dirichlet concentration parameter


# Model definition
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Non-IID data partitioning using Dirichlet distribution
def dirichlet_split(dataset, num_clients, alpha):
    num_classes = len(dataset.classes)
    client_indices = {i: [] for i in range(num_clients)}

    # Get class indices
    class_indices = [[] for _ in range(num_classes)]
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    # Sample from Dirichlet distribution for each class
    for c in range(num_classes):
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.cumsum(proportions)
        proportions = proportions / proportions[-1]
        splits = (len(class_indices[c]) * proportions).astype(int)[:-1]
        client_class_data = np.split(np.random.permutation(class_indices[c]), splits)

        for client_idx in range(num_clients):
            if client_idx < len(client_class_data):
                client_indices[client_idx].extend(client_class_data[client_idx])

    return [Subset(dataset, indices) for indices in client_indices.values()]


# FedProx loss function
def proximal_loss(local_model, global_model, criterion, output, target, mu):
    basic_loss = criterion(output, target)
    if mu == 0:
        return basic_loss

    proximal_term = 0.0
    for (name, local_param), (_, global_param) in zip(
            local_model.named_parameters(), global_model.named_parameters()
    ):
        proximal_term += torch.norm(local_param - global_param, p=2) ** 2

    return basic_loss + (mu / 2) * proximal_term


# Client update function
def client_update(client_model, global_model, train_loader, mu, device):
    client_model.train()
    optimizer = torch.optim.SGD(client_model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.NLLLoss()

    for _ in range(LOCAL_EPOCHS):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = client_model(data)
            loss = proximal_loss(client_model, global_model, criterion, output, target, mu)
            loss.backward()
            optimizer.step()

    return client_model.state_dict()


# Main training loop
def federated_learning(mu_value):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and partition data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    # Non-IID data partitioning
    client_datasets = dirichlet_split(train_dataset, NUM_CLIENTS, ALPHA)
    client_loaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True) for ds in client_datasets]
    test_loader = DataLoader(test_dataset, batch_size=1000)

    # Initialize models
    global_model = MNISTNet().to(device)
    client_models = [MNISTNet().to(device) for _ in range(NUM_CLIENTS)]
    best_acc = 0.0
    acc_history = []

    for round in range(COMM_ROUNDS):
        # Randomly select clients (50% participation rate)
        selected_clients = np.random.choice(NUM_CLIENTS, size=max(1, NUM_CLIENTS // 2), replace=False)

        # Local training
        local_weights = []
        for client_idx in selected_clients:
            client_model = copy.deepcopy(global_model)
            weights = client_update(
                client_model,
                global_model,
                client_loaders[client_idx],
                mu_value,
                device
            )
            local_weights.append(weights)

        # Aggregate updates (FedAvg)
        global_weights = {}
        for key in local_weights[0].keys():
            global_weights[key] = torch.stack(
                [weights[key] for weights in local_weights], 0
            ).mean(0)

        # Update global model
        global_model.load_state_dict(global_weights)

        # Evaluation
        global_model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = global_model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        acc = 100. * correct / len(test_loader.dataset)
        acc_history.append(acc)
        if acc > best_acc:
            best_acc = acc

        print(f'Round {round + 1}, μ={mu_value}: Test Accuracy: {acc:.2f}%')

    return acc_history


# Run experiments for different μ values
results = {}
for mu in MU_VALUES:
    print(f"\nRunning FedProx with μ = {mu}")
    results[mu] = federated_learning(mu)

# Plot results
plt.figure(figsize=(10, 6))
for mu, acc_history in results.items():
    plt.plot(acc_history, label=f'μ = {mu}')
plt.xlabel('Communication Rounds')
plt.ylabel('Test Accuracy (%)')
plt.title('FedProx Performance with Different Proximal Terms')
plt.legend()
plt.grid()
plt.savefig('fedprox_results.png')
plt.show()