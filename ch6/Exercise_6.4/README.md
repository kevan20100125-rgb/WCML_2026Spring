# Exercise 6.4: FedProx Implementation for MNIST Classification

Welcome! Your task is to finish an **incomplete implementation of FedProx** for
federated MNIST classification. You will:

1. **Fill in the missing loss definition** inside `local_train`
   – combine the usual negative-log-likelihood (NLL) with FedProx’s proximal
   penalty.

2. **Run the experiment** for three $\mu$ values (`0, 0.01, 0.1`) and inspect the
   convergence behaviour.

## Background

FedProx augments Federated Averaging (FedAvg) by adding a proximal term to the local objective function, which helps address issues arising from client heterogeneity. The local objective is defined as:

$$
L_i(w; w_t) = F_i(w) + \frac{\mu}{2}\left\\| w - w_t \right\\|^2,
$$

where:
- $F_i(w)$ is the original loss function of client i.
- $w_t$ is the global model parameters.
- $\mu$ is the proximal term coefficient.



## What You Need to Do

| Checklist | Details |
|-----------|---------|
| **Code** | Open `exercise_6.4_starter.py` and look for the `# YOUR CODE HERE` block inside **`local_train`**. Replace the placeholder with code that:<br>  • zero gradients;<br>  • computes NLL on the batch;<br>  • computes the proximal penalty;<br>  • sums them and performs a backward/step update. |
| **Run** | Execute:<br>`python exercise_6.6_starter.py` |
| **Observe** | A plot `fedprox_mu_impact.png` is saved showing test accuracy vs. communication rounds for each $\mu$. |


> **Hint:** Re-use the pattern `((w - w_glob.detach()) ** 2).sum()` to compute
> $\left\\| w - w_t \right\\|^2$ across all parameters.


## Expected Outcome

* **$\mu$ = 0** should reproduce FedAvg.
* **Higher $\mu$** should smooth or slow convergence (exact behaviour depends on
  data skew).



## Files

| File | Purpose |
|------|---------|
| `exercise_6.4_starter.py` | Starter script (with TODO). |
| `exercise_6.4_solution.py` | Completed script. |
| `fedprox_mu_impact.png` | Generated after you run the script. |
