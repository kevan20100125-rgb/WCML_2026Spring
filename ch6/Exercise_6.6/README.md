# Exercise 6.6: FedAvg Implementation under Packet Loss

Welcome! Your task is to explore how unreliable communication degrades
federated learning. You will:

1. **Implement the packet-loss simulator** in `unreliable_channel`.
2. **Finish the local-training routine** in `client_train` so that each client
   optimises the negative-log-likelihood (NLL) loss.
3. **Run the experiment** for three loss-rate settings (`0.01  0.05  0.10`) and
   examine the resulting convergence curves.


## Background

Real-world edge devices often drop messages.
We model this as *packet loss*: each client update is discarded with
probability *p*. The server then aggregates only the surviving updates using
Federated Averaging (FedAvg).


## What You Need to Do

| Checklist | Details |
|-----------|---------|
| **Code #1** | Open **`fedavg_packet_loss_exercise.py`** and locate the `# YOUR CODE HERE` block inside **`unreliable_channel`**. Write code that drops each update with probability `loss_rate` (replace dropped items with `None` *or* filter them out—document your choice). |
| **Code #2** | In **`client_train`**, complete the `# YOUR CODE HERE` block:<br>  • zero gradients;<br>  • forward pass;<br>  • compute NLL loss with `torch.nn.functional.nll_loss`;<br>  • `backward()` and `opt.step()`. |
| **Run** | Execute: `python exercise_6.6_starter.py` |
| **Observe** | A figure **`packet_loss_impact.png`** is saved, showing *test accuracy* and *NLL loss* versus communication rounds for each loss rate. |

> **Hint:** use `np.random.rand() > loss_rate` to decide whether to keep each
> update.


## Configuration (default)

```python
NUM_CLIENTS   = 100                 # Total simulated clients
COMM_ROUNDS   = 100                 # Global communication rounds
LOCAL_EPOCHS  = 5                   # Local epochs per round
BATCH_SIZE    = 32                  # Mini-batch size for client training
ALPHA         = 0.5                 # Dirichlet concentration (smaller ⇒ more skew)
LOSS_RATES    = [0.01, 0.05, 0.10]  # Packet-loss probabilities to test
```

## Expected Outcome

* **1% packet-loss rate** should behave close to vanilla FedAvg.
* **5-10% packet-loss rate** will slow convergence and may lower the final accuracy; the
NLL curve should flatten more slowly as `LOSS_RATES` increases.



## Files

| File | Purpose |
|------|---------|
| `exercise_6.6_starter.py` | Starter script (with TODO). |
| `exercise_6.6_solution.py` | Completed script. |
| `packet_loss_impact.png` | Generated after you run the script. |