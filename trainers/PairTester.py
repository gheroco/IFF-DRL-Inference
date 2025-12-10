import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from agents.PPOagent import PPO
from dataLoad.PairTradingDataLoader import PairTradingDataset, pair_batch_fn
from Env.PairTradingEnv import PairTradingEnv


class PairTester:
    """Lightweight tester for the pair-trading environment."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.agent = PPO(cfg)
        self.env = PairTradingEnv(cfg.pair_env, init_money=cfg.train.init_money)

    def build_loader(self):
        dataset = PairTradingDataset(
            csv_path=self.cfg.pair_env.dataset,
            start_date=self.cfg.time.train_startingDate,
            end_date=self.cfg.time.test_endingDate,
            input_length=self.cfg.pair_env.input_length,
            reward_horizon=self.cfg.pair_env.reward_horizon,
        )
        return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=pair_batch_fn)

    def test(self):
        loader = self.build_loader()
        self.env.reset(len(loader))
        total_reward = 0
        with tqdm(total=len(loader), desc='PairTest') as bar:
            for batch in loader:
                window, spread_yesterday, spread_today, spread_future = [i.to(self.cfg.device) for i in batch]
                action, _ = self.agent.act(window, train=False)
                reward = self.env.agent_step(spread_today.squeeze(), spread_future.squeeze(), action)
                total_reward += reward.item()
                bar.update(1)
        return total_reward
