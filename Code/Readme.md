
# Breaking Multi-Task Curse: Reward-Weighted Evolution for Many Heterogeneous Tasks

This repository contains the implementation of **MES-RET** and **sep-MES-RET (sMES-RET)**, designed for solving both standard many-task optimization problems and policy search tasks using reward-weighted evolution.

## Directory Structure

```
.
├── Algorithms/      # Contains the proposed algorithms and baselines
├── Problems/        # Contains benchmark functions and policy search environments
```

## Getting Started

> ⚠️ **Important:** All code in this repository is designed to run **within the [MTO-Platform (MToP)](https://github.com/intLyc/MTO-Platform)**.

Please **clone or download the MToP** and place corresponding files inside the `Algorithms/` and `Problems/` directories. More baselines can be found in MToP.
1
## For Policy Search Tasks

The policy search environments involve **Python-MATLAB hybrid programming** using **MATLAB's Python interface**. To run these tasks:

- MATLAB **R2023b or newer** is **mandatory**.
- A working Python environment (e.g., Python 3.9+) with required libraries (as shown in `EMaT-Gym/requirements.txt`) must be properly configured.
- You can test and set the Python-MATLAB bridge via `pyenv`

