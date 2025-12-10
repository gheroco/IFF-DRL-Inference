from datetime import datetime

from omegaconf import OmegaConf

from trainers.PairTester import PairTester


if __name__ == '__main__':
    cfg = OmegaConf.load('configs/pair_trading.yaml')
    cfg.time.train_startingDate = datetime.strptime(cfg.time.train_startingDate, '%Y-%m-%d')
    cfg.time.test_endingDate = datetime.strptime(cfg.time.test_endingDate, '%Y-%m-%d')
    tester = PairTester(cfg)
    tester.test()
