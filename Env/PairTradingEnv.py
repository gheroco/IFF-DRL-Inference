import numpy as np
import pandas as pd
import torch


class PairTradingEnv:
    """Environment that mirrors the dynamic scaling design from the MDPI article."""

    def __init__(self, cfg, init_money=100000):
        self.cfg = cfg
        self.init_money = init_money
        self.scales = cfg.action_scales
        self.transaction_cost = cfg.transaction_cost
        self.reward_horizon = cfg.reward_horizon
        self.reset(0)

    def reset(self, total_steps):
        self.length = total_steps
        self.t = 0
        self.position_scale = 0.0
        self.account = pd.DataFrame(
            index=np.arange(0, total_steps + 1),
            data={
                'Cash': np.full(total_steps + 1, float(self.init_money)),
                'PositionScale': np.zeros(total_steps + 1),
                'Capitals': np.full(total_steps + 1, float(self.init_money)),
                'Reward': np.zeros(total_steps + 1),
            },
        )

    def agent_step(self, spread_today: torch.Tensor, spread_future: torch.Tensor, action_idx: torch.Tensor):
        desired_scale = float(self.scales[int(action_idx)])
        delta_scale = desired_scale - self.position_scale

        spread_change = (spread_future[-1] - spread_today).item()
        reward = desired_scale * spread_change
        reward -= abs(delta_scale) * self.transaction_cost

        self.position_scale = desired_scale
        self.t += 1

        self.account.loc[self.t, 'PositionScale'] = self.position_scale
        self.account.loc[self.t, 'Reward'] = reward
        self.account.loc[self.t, 'Cash'] = self.account.loc[self.t - 1, 'Cash']
        self.account.loc[self.t, 'Capitals'] = self.account.loc[self.t - 1, 'Capitals'] + reward
        return torch.tensor(reward)
