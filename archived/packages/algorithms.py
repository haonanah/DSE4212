from __future__ import annotations

import copy
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import your policy architecture and utility functions
from .architectures import EIIE  # Replace with the correct import path
from .utils import apply_portfolio_noise, PVM, ReplayBuffer, RLDataset


class PolicyGradient:
    """Class implementing policy gradient algorithm to train portfolio
    optimization agents.

    Note:
        During testing, the agent is optimized through online learning.
        The parameters of the policy are updated repeatedly after a constant
        period of time. To disable it, set learning rate to 0.

    Attributes:
        train_env: Environment used to train the agent.
        train_policy: Policy used in training.
        validation_env: Environment used for validation.
        test_env: Environment used to test the agent.
        test_policy: Policy after test online learning.
    """

    def __init__(
        self,
        env,
        policy=EIIE,
        policy_kwargs=None,
        validation_env=None,
        batch_size=100,
        lr=1e-3,
        action_noise=0,
        optimizer=AdamW,
        device="cpu",
    ):
        """Initializes Policy Gradient for portfolio optimization.

        Args:
            env: Training environment.
            policy: Policy architecture to be used.
            policy_kwargs: Arguments to be used in the policy network.
            validation_env: Validation environment.
            batch_size: Batch size to train neural network.
            lr: Policy neural network learning rate.
            action_noise: Noise parameter (between 0 and 1) to be applied
                during training.
            optimizer: Optimizer of neural network.
            device: Device where neural network is run.
        """
        self.policy = policy
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.validation_env = validation_env
        self.batch_size = batch_size
        self.lr = lr
        self.action_noise = action_noise
        self.optimizer_class = optimizer
        self.device = device
        self._setup_train(env, self.policy, self.batch_size, self.lr, self.optimizer_class)

    def _setup_train(self, env, policy, batch_size, lr, optimizer_class):
        """Initializes algorithm before training.

        Args:
            env: Training environment.
            policy: Policy architecture to be used.
            batch_size: Batch size to train neural network.
            lr: Policy neural network learning rate.
            optimizer_class: Optimizer of neural network.
        """
        # Environment
        self.train_env = env

        # Neural networks
        self.train_policy = policy(**self.policy_kwargs).to(self.device)
        self.train_optimizer = optimizer_class(self.train_policy.parameters(), lr=lr)

        # Replay buffer and portfolio vector memory
        self.train_batch_size = batch_size
        self.train_buffer = ReplayBuffer(capacity=batch_size)
        self.train_pvm = PVM(self.train_env.episode_length, env.portfolio_size)

        # Dataset and DataLoader
        dataset = RLDataset(self.train_buffer)
        self.train_dataloader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, pin_memory=True
        )
        self.scheduler = ReduceLROnPlateau(self.train_optimizer, mode='max', factor=0.1, patience=5, threshold=0.01)

    def train(self, episodes=100, callback=None, validation_freq=1, patience=10):
        """Training sequence with validation-based early stopping.

        Args:
            episodes: Number of episodes to simulate.
            callback: Callback function or object.
            validation_freq: Frequency (in episodes) to perform validation.
            patience: Number of validation evaluations with no improvement to wait before stopping.
        """
        best_validation_performance = -np.inf
        episodes_without_improvement = 0
        best_model_state = None  
        threshold = 1e-7  # Threshold to consider an improvement
        total_steps = 0
        for episode in tqdm(range(1, episodes + 1)):
            obs = self.train_env.reset()
            self.train_pvm.reset()
            done = False
            step_counter = 0

            while not done:
                step_counter += 1
                total_steps += 1
                last_action = self.train_pvm.retrieve()
                obs_batch = np.expand_dims(obs, axis=0)
                last_action_batch = np.expand_dims(last_action, axis=0)
                action = apply_portfolio_noise(
                    self.train_policy(obs_batch, last_action_batch), self.action_noise
                )
                self.train_pvm.add(action)

                next_obs, reward, done, info = self.train_env.step(action)

                exp = (obs, last_action, info["price_variation"], info.get("trf_mu", 1.0))
                self.train_buffer.append(exp)

                if len(self.train_buffer) == self.train_batch_size:
                    self._gradient_ascent()

                if callback is not None:
                    callback_info = {
                        'obs': obs,
                        'next_obs': next_obs,
                        'reward': reward,
                        'done': done,
                        'info': info,
                        'episode': episode,
                        'step': step_counter,
                        'total_steps': total_steps,
                    }
                    continue_training = callback(callback_info)
                    if not continue_training:
                        print(f"Early stopping triggered at episode {episode}, step {step_counter}")
                        return  

                obs = next_obs

            self._gradient_ascent()

            if self.validation_env and episode % validation_freq == 0:
                validation_performance = self.evaluate_on_validation()
                print(f"Validation performance at episode {episode}: {validation_performance}")
                self.scheduler.step(validation_performance)  

                if validation_performance > best_validation_performance + threshold:
                    best_validation_performance = validation_performance
                    episodes_without_improvement = 0
                    best_model_state = copy.deepcopy(self.train_policy.state_dict())
                else:
                    episodes_without_improvement += 1
                    print(f"No improvement for {episodes_without_improvement} validation(s).")
                    if episodes_without_improvement >= patience:
                        print("Early stopping due to no improvement on validation set.")
                        break 

        if best_model_state is not None:
            self.train_policy.load_state_dict(best_model_state)
            print("Loaded best model based on validation performance.")

    def evaluate_on_validation(self):
        """Evaluates the current policy on the validation environment."""
        obs = self.validation_env.reset()
        self.train_pvm.reset() 
        done = False
        total_reward = 0
        rewards = [] 

        while not done:
            last_action = self.train_pvm.retrieve()
            obs_batch = np.expand_dims(obs, axis=0)
            last_action_batch = np.expand_dims(last_action, axis=0)
            with torch.no_grad():
                action = self.train_policy(obs_batch, last_action_batch)
            self.train_pvm.add(action)

            obs, reward, done, info = self.validation_env.step(action)
            total_reward += reward
            rewards.append(reward)

        self.train_pvm.reset()

        # cumulative_reward = total_reward
        sharpe_ratio = np.mean(rewards) / (np.std(rewards) + 1e-8)

        return sharpe_ratio  # or cumulative_reward
    
    def _setup_test(self, env, policy, batch_size, lr, optimizer_class):
        """Initializes algorithm before testing.

        Args:
            env: Testing environment.
            policy: Policy architecture to be used.
            batch_size: Batch size to train neural network.
            lr: Policy neural network learning rate.
            optimizer_class: Optimizer of neural network.
        """
        # Environment
        self.test_env = env

        # Process None arguments
        policy = self.train_policy if policy is None else policy
        lr = self.lr if lr is None else lr
        optimizer_class = self.optimizer_class if optimizer_class is None else optimizer_class

        # Neural networks
        self.test_policy = copy.deepcopy(policy).to(self.device)
        self.test_optimizer = optimizer_class(self.test_policy.parameters(), lr=lr)

        # Replay buffer and portfolio vector memory
        self.test_buffer = ReplayBuffer(capacity=batch_size)
        self.test_pvm = PVM(self.test_env.episode_length, env.portfolio_size)

        # Dataset and DataLoader
        dataset = RLDataset(self.test_buffer)
        self.test_dataloader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False, pin_memory=True
        )

    def test(
        self, env, policy=None, online_training_period=10, lr=None, optimizer=None
    ):
        """Tests the policy with online learning.

        Args:
            env: Environment to be used in testing.
            policy: Policy architecture to be used. If None, it will use the training
                architecture.
            online_training_period: Period in which an online training will occur. To
                disable online learning, use a very big value.
            lr: Policy neural network learning rate. If None, it will use the training
                learning rate.
            optimizer: Optimizer of neural network. If None, it will use the training
                optimizer.

        Note:
            To disable online learning, set learning rate to 0 or a very big online
            training period.
        """
        self._setup_test(env, policy, self.batch_size, lr, optimizer)

        obs = self.test_env.reset()
        self.test_pvm.reset()
        done = False
        steps = 0

        while not done:
            steps += 1
            # Define last_action and action, update portfolio vector memory
            last_action = self.test_pvm.retrieve()
            obs_batch = np.expand_dims(obs, axis=0)
            last_action_batch = np.expand_dims(last_action, axis=0)
            with torch.no_grad():
                action = self.test_policy(obs_batch, last_action_batch)
            self.test_pvm.add(action)

            # Run simulation step
            next_obs, reward, done, info = self.test_env.step(action)

            # Add experience to replay buffer
            exp = (obs, last_action, info["price_variation"], info.get("trf_mu", 1.0))
            self.test_buffer.append(exp)

            # Update policy networks
            if steps % online_training_period == 0 and lr > 0:
                self._gradient_ascent(test=True)

            obs = next_obs

    def _gradient_ascent(self, test=False):
        """Performs the gradient ascent step in the policy gradient algorithm.

        Args:
            test: If true, it uses the test dataloader and policy.
        """
        # Get batch data from dataloader
        if test:
            dataloader = self.test_dataloader
            policy = self.test_policy
            optimizer = self.test_optimizer
        else:
            dataloader = self.train_dataloader
            policy = self.train_policy
            optimizer = self.train_optimizer

        obs, last_actions, price_variations, trf_mu = next(iter(dataloader))
        obs = obs.to(self.device)
        last_actions = last_actions.to(self.device)
        price_variations = price_variations.to(self.device)
        trf_mu = trf_mu.unsqueeze(1).to(self.device)

        # Define policy loss (negative for gradient ascent)
        mu = policy.mu(obs, last_actions)
        portfolio_returns = torch.sum(mu * price_variations * trf_mu, dim=1)
        policy_loss = -torch.mean(torch.log(portfolio_returns))

        # Update policy network
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
