# Exercise 6.4: FedProx Implementation for MNIST Classification

This repository provides the starter code for Exercise 6.4. Your objective is to implement the **proximal term** of the FedProx algorithm and evaluate its performance on a non-i.i.d. MNIST dataset.

## Experiment Setup
The script is pre-configured with the specific parameters from the textbook:
* **Clients $N$ :** 20
* **Communication Rounds $T$ :** 100
* **Local Epochs $E$ :** 5
* **Non-i.i.d. Data:** Dirichlet distribution ($\alpha = 0.5$)
* **Proximal Coefficients $\mu$ :** 0 (FedAvg baseline), 0.01, and 0.1

## What You Need to Do

| Checklist | Details |
|-----------|---------|
| **Code** | Open `exercise_6.4_starter.py` and look for the `# YOUR CODE HERE` block inside **`local_train`**. Replace the placeholder with code that:<br>  • zero gradients;<br>  • computes NLL on the batch;<br>  • computes the proximal penalty;<br>  • `backward()` and `opt.step()`.  |
| **Run** | Execute: `python exercise_6.4_starter.py` |
| **Observe** | A plot `fedprox_mu_impact.png` is saved showing test accuracy vs. communication rounds for each $\mu$. |


> **Hint:** Re-use the pattern `((w - w_glob.detach()) ** 2).sum()` to compute
> $||w - w_t||^2$ across all parameters.



## Files

| File | Purpose |
|------|---------|
| `exercise_6.4_starter.py` | Starter script (with TODO). |
| `fedprox_mu_impact.png` | Generated after you run the script. |


