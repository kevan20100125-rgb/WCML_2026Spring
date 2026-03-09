# Exercise 1.12: PPO Implementation for MountainCar-v0

This repository provides the starter code for Exercise 1.12. Your task is to implement the **Proximal Policy Optimization (PPO)** algorithm to solve the classic `MountainCar-v0` reinforcement learning task.

## Verification

- **Run command**
  - `cd WCML/ch1/Exercise_1.12`
  - `python PPO_MountainCar-v0.py`
- **Expected log (stdout)**
  - Early in training, you should see lines like: `I_ep 0 ，train 0 times`
  - During longer training, `train` value increases (for example, `... train 1000 times`)
  - Program exits with final line: `end`
- **Verification method**
  - Confirm there is **no runtime error** and process ends normally with `end`.
  - Confirm PPO update is actually running by checking at least one `I_ep ... train ... times` line.
  - Optional learning check: run `tensorboard --logdir ../exp` and verify `Steptime/steptime` is being logged.
