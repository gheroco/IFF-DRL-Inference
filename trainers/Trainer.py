import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from utils.estimator import PerformanceEstimator
from utils.ReplayBuffer import ReplayBuffer
from agents.ACagent import A2C
from agents.PPOagent import PPO
from agents.DQNagent import DDQN
from Env.TradingEnv import TradingEnv
from dataLoad.utils import normalization
from utils.RewardRecord import RewardRecord
from dataLoad.utils import confirm_makedirs
import os
from utils.Visualizer import Visualizer

class AgentTrainer:

    def __init__(self, cfg, AgentName):
        """
        :param cfg: common.yaml
        :param AgentName: 模型名称，用来保存训练文件和加载配置文件（.yaml）
        """
        self.cfg = cfg
        self.env = TradingEnv(cfg, init_money=cfg.train.init_money)
        self.agent = eval(cfg.agent.algorithm)(cfg)
        self.memory = ReplayBuffer(self.cfg.agent.buffer_size)
        self.saveDir = os.path.join('result', cfg.expName, AgentName)
        self.agentName = AgentName
        self.saveModelDir = os.path.join(self.saveDir, 'model')
        self.trainDir = os.path.join(self.saveDir, 'train')
        confirm_makedirs(self.saveModelDir)
        confirm_makedirs(self.trainDir)
        self.visualizer = Visualizer()
        self.lossRecord = RewardRecord(self.trainDir, self.cfg)
        self.best_reward = -999999
        self.num_count = []
        self.step_count = 0

    def train(self, train_data, val_data=None):
        """
        Training procedure for IFF-DRL agent
        """
        self.agent.policy_net.train()
        total_epochs = self.cfg.train.epochs
        train_interval = self.cfg.train.train_interval
        
        for epoch in range(total_epochs):
            print(f'\nEpoch {epoch+1}/{total_epochs}')
            totalStep = len(train_data)
            self.env.reset(totalStep)
            TotalReward = 0
            episode_reward = 0
            
            with tqdm(total=totalStep, desc=f'Epoch {epoch+1}') as pbar:
                for iteration, batch in enumerate(train_data):
                    datas, priceTMinus1, priceT, priceTPlusN, datasNext, mask, maskNext = [i.to(self.cfg.device) for i in batch]
                    
                    if self.cfg.normalization:
                        datas = normalization(datas)
                        datasNext = normalization(datasNext)
                    
                    if iteration == 0:
                        self.env.setInitPrice(priceT)
                    
                    self.env.setState(iteration, priceTMinus1, priceT, priceTPlusN)
                    self.env.getNday()
                    
                    # Agent selects action
                    if self.cfg.agent.algorithm == 'PPO':
                        agent_action, log_prob = self.agent.act(datas, train=True)
                    else:
                        agent_action = self.agent.act(datas, train=True)
                    
                    # Execute action and get reward
                    reward = self.env.agentStep(datas, agent_action)
                    TotalReward += reward.item()
                    episode_reward += reward.item()
                    
                    # Store experience in replay buffer immediately
                    if self.cfg.agent.algorithm == 'PPO':
                        # For PPO, store with log_prob
                        self.memory.PPO_Agent_add(datas, agent_action, reward, datasNext, log_prob)
                    else:
                        self.memory.Agent_add(datas, agent_action, reward, datasNext)
                    
                    # Learning step: sample from buffer and update agent periodically
                    if (iteration + 1) % train_interval == 0 and len(self.memory) >= self.cfg.train.batch_size:
                        batch_size = min(self.cfg.train.batch_size, len(self.memory))
                        if self.cfg.agent.algorithm == 'PPO':
                            batch = self.memory.PPO_Agent_sample(batch_size)
                            self.agent.learn(batch)
                        else:
                            batch = self.memory.Agent_sample(batch_size)
                            self.agent.learn(batch)
                    
                    self.step_count += 1
                    pbar.update(1)
                    pbar.set_postfix({'Reward': f'{TotalReward:.2f}', 'Step': self.step_count})
            
            # Record reward for this episode
            self.lossRecord.appendRewardWithVal(TotalReward, self.agentName)
            
            # Save model periodically
            if (epoch + 1) % self.cfg.modelSavePerEpoch == 0:
                model_path = os.path.join(self.saveModelDir, f'{self.cfg.agent.algorithm}_{self.cfg.dataSetName}_epoch_{epoch+1}.pth')
                torch.save(self.agent.policy_net.state_dict(), model_path)
            
            # Save best model
            if TotalReward > self.best_reward:
                self.best_reward = TotalReward
                best_model_path = os.path.join(self.saveModelDir, f'{self.cfg.agent.algorithm}_{self.cfg.dataSetName}_best.pth')
                torch.save(self.agent.policy_net.state_dict(), best_model_path)
                print(f'New best model saved with reward: {self.best_reward:.2f}')
                
                # Plot training capital
                self.lossRecord.training_CapitalPlot(self.env.account, self.agentName)
            
            # Validation on training set (for monitoring)
            if val_data is not None and (epoch + 1) % 10 == 0:
                print(f'Evaluating on validation set at epoch {epoch+1}...')
                self.agent.policy_net.eval()
                val_reward = self._validate(val_data)
                self.agent.policy_net.train()
                print(f'Validation reward: {val_reward:.2f}')
        
        # Save final model
        final_model_path = os.path.join(self.saveModelDir, f'{self.cfg.agent.algorithm}_{self.cfg.dataSetName}_final.pth')
        torch.save(self.agent.policy_net.state_dict(), final_model_path)
        print(f'Training completed. Final model saved to {final_model_path}')
        
        # Save model with standard naming convention for testing
        standard_model_path = os.path.join('result', 'Trained_Model', f'{self.cfg.agent.algorithm}_{self.cfg.dataSetName}.pth')
        confirm_makedirs(os.path.dirname(standard_model_path))
        torch.save(self.agent.policy_net.state_dict(), standard_model_path)
        print(f'Model also saved to {standard_model_path} for testing')

    def _validate(self, val_data):
        """
        Quick validation during training
        """
        self.agent.policy_net.eval()
        totalStep = len(val_data)
        self.env.reset(totalStep)
        TotalReward = 0
        
        with torch.no_grad():
            for iteration, batch in enumerate(val_data):
                datas, priceTMinus1, priceT, priceTPlusN, datasNext, mask, maskNext = [i.to(self.cfg.device) for i in batch]
                if self.cfg.normalization:
                    datas = normalization(datas)
                if iteration == 0:
                    self.env.setInitPrice(priceT)
                self.env.setState(iteration, priceTMinus1, priceT, priceTPlusN)
                self.env.getNday()
                if self.cfg.agent.algorithm == 'PPO':
                    agent_action, _ = self.agent.act(datas, train=False)
                else:
                    agent_action = self.agent.act(datas, train=False)
                reward = self.env.agentStep(datas, agent_action)
                TotalReward += reward.item()
        
        return TotalReward
