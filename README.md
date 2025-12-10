# Incremental Forecast Fusion Deep Reinforcement Learning

This project is a reinforcement learning-based stock trading system. It allows you to train and test models to optimize stock trading strategies.

## Dependencies

Before running the project, make sure you have the following dependencies installed:

* Python 3.9.12
* PyTorch 1.12.1
* OmegaConf
* Tqdm
* Numpy
* Pandas
* Matplotlib
* Tabulate
* Pillow

To execute the code, simply run the following command:

```bash
python main.py
```

## Pair trading system (MDPI 17(12) 555)

This repository now includes a lightweight implementation of the article *Reinforcement Learning Pair Trading: A Dynamic Scaling Approach* (MDPI Journal of Risk and Financial Management 17(12):555). The pipeline mirrors the paper's dynamic position scaling over a pair of correlated assets:

- `configs/pair_trading.yaml` captures the pair-specific hyperparameters, action scaling grid, and dataset location.
- `dataLoad/PairTradingDataLoader.py` builds time-series windows over two assets and the spread.
- `Env/PairTradingEnv.py` executes scaled pair positions and computes rewards based on spread changes and transaction costs.
- `trainers/PairTester.py` ties the PPO agent to the new environment and loader.
- `pair_main.py` is the entry point for running inference over the provided sample dataset.

Run the pair-trading evaluator with:

```bash
python pair_main.py
```
