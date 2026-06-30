import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal
import cv2
import gymnasium as gym
import concurrent.futures  
import os



device = torch.device("cpu")
torch.set_num_threads(1)

def init_weights(m, gain=1.0):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0.0)



class ReplayBuffer:
    """Buffer for Off-Policy Algorithms (DDPG, TD3)"""
    def __init__(self, state_dim, action_dim, max_size=1000000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.terminated = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, terminated):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.terminated[self.ptr] = terminated
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.state[ind], dtype=torch.float32),
            torch.as_tensor(self.action[ind], dtype=torch.float32),
            torch.as_tensor(self.reward[ind], dtype=torch.float32),
            torch.as_tensor(self.next_state[ind], dtype=torch.float32),
            torch.as_tensor(self.terminated[ind], dtype=torch.float32)
        )

    def __len__(self):
        return self.size

class RolloutBuffer:
    """Buffer for On-Policy Algorithms (PPO)"""
    def __init__(self, state_dim, action_dim, max_size):
        self.max_size = max_size
        self.ptr = 0
        
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.logprob = np.zeros((max_size, 1), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.state_value = np.zeros((max_size, 1), dtype=np.float32)
        self.terminated = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, logprob, reward, state_value, terminated):
        if self.ptr < self.max_size:
            self.state[self.ptr] = state
            self.action[self.ptr] = action
            self.logprob[self.ptr] = logprob
            self.reward[self.ptr] = reward
            self.state_value[self.ptr] = state_value
            self.terminated[self.ptr] = terminated
            self.ptr += 1

    def clear(self):
        self.ptr = 0

    def get_data(self):
        return (
            torch.as_tensor(self.state[:self.ptr], dtype=torch.float32).to(device),
            torch.as_tensor(self.action[:self.ptr], dtype=torch.float32).to(device),
            torch.as_tensor(self.logprob[:self.ptr], dtype=torch.float32).to(device),
            torch.as_tensor(self.reward[:self.ptr], dtype=torch.float32).to(device),
            torch.as_tensor(self.state_value[:self.ptr], dtype=torch.float32).to(device),
            torch.as_tensor(self.terminated[:self.ptr], dtype=torch.float32).to(device)
        )



from stable_baselines3.common.callbacks import BaseCallback

class RewardLoggerCallback(BaseCallback):
    def __init__(self, max_episodes):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = [] 
        self.max_episodes = max_episodes

    def _on_step(self):
        if 'episode' in self.locals['infos'][0]:
            self.episode_rewards.append(self.locals['infos'][0]['episode']['r'])
            self.episode_lengths.append(self.locals['infos'][0]['episode']['l']) 
            if len(self.episode_rewards) >= self.max_episodes:
                return False
        return True





class DDPG_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(DDPG_Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.layer2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.layer3 = nn.Linear(256, action_dim)

        torch.nn.init.uniform_(self.layer3.weight, a=-3e-3, b=3e-3)
        torch.nn.init.uniform_(self.layer3.bias, a=-3e-3, b=3e-3)
        self.max_action = max_action

    def forward(self, state):
        a = self.layer1(state)
        a = F.relu(self.ln1(a))
        a = self.layer2(a)
        a = F.relu(self.ln2(a))
        a = torch.tanh(self.layer3(a))
        return self.max_action * a

class DDPG_Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DDPG_Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.layer2 = nn.Linear(256 + action_dim, 256)
        self.layer3 = nn.Linear(256, 1)

        torch.nn.init.uniform_(self.layer3.weight, a=-3e-3, b=3e-3)
        torch.nn.init.uniform_(self.layer3.bias, a=-3e-3, b=3e-3)

    def forward(self, state, action):
        q = self.layer1(state)
        q = F.relu(self.ln1(q))
        q = torch.cat([q, action], dim=1)
        q = F.relu(self.layer2(q))
        q = self.layer3(q)
        return q

class DDPG:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = DDPG_Actor(state_dim, action_dim, max_action).to(device)
        self.critic = DDPG_Critic(state_dim, action_dim).to(device)

        self.actor_target = DDPG_Actor(state_dim, action_dim, max_action).to(device)
        self.critic_target = DDPG_Critic(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)

        self.max_action = max_action
        self.gamma = 0.99
        self.tau = 0.001

    def select_action(self, state, noise_std=0.1, noise_type='gauss', ou_noise=None):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        with torch.no_grad():
            action = self.actor(state).numpy()[0]

        if noise_type == 'gauss':
            noise = np.random.normal(0, max(self.max_action * noise_std, 0), size=action.shape)
            action = action + noise
        elif noise_type == 'ou' and ou_noise is not None:
            noise = ou_noise() * max(noise_std, 0)
            action = action + noise
        elif noise_type == 'epsilon_greedy':
            if random.random() < noise_std:
                action = np.random.uniform(-self.max_action, self.max_action, size=action.shape)

        return np.clip(action, -self.max_action, self.max_action)

    def train(self, replay_buffer, batch_size=128):
        state, action, reward, next_state, terminated = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + (1 - terminated) * self.gamma * target_Q

        current_Q = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item(), current_Q.mean().item()

def train_ddpg(seed, batch_size=64, episodes=10001, record_video=False, noise_type='gauss', reward_scale=0.1, learning_starts=2000):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    render_mode = "rgb_array" if record_video else None
    env = gym.make("BipedalWalker-v3", render_mode=render_mode)
    env.reset(seed=seed)

    out = None
    epizody_do_nagrania = [0, 100, 1000, 2000, 4000, 8000]

    if record_video:
        sample_frame = env.render()
        wysokosc, szerokosc, _ = sample_frame.shape
        out = cv2.VideoWriter(f'ewolucja_ddpg_seed_{seed}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (szerokosc, wysokosc))

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = DDPG(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    if noise_type == 'ou':
        from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
        ou_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(action_dim), sigma=np.ones(action_dim) * 0.2, theta=0.15)
    else:
        ou_noise = None

    historia_nagrod, historia_straty, historia_krokow = [], [], []

    start_noise, end_noise, exploration_fraction = 0.2, 0.05, 0.9
    noise_decay = (start_noise - end_noise) / int(episodes * exploration_fraction)

    for episode in range(episodes):
        if ou_noise is not None: ou_noise.reset()
        state, _ = env.reset()
        episode_reward, kroki_w_epizodzie = 0, 0
        straty_w_epizodzie = []
        terminated, truncated = False, False

        current_noise = max(end_noise, start_noise - episode * noise_decay)

        while not (terminated or truncated):
            kroki_w_epizodzie += 1 
            if len(replay_buffer) < learning_starts:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, noise_std=current_noise, noise_type=noise_type, ou_noise=ou_noise)

            next_state, reward, terminated, truncated, _ = env.step(action)
            scaled_reward = reward * reward_scale

            if record_video and (episode in epizody_do_nagrania):
                frame = env.render()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.putText(frame_bgr, f"DDPG Epizod: {episode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                out.write(frame_bgr)

            replay_buffer.add(state, action, scaled_reward, next_state, terminated)

            if len(replay_buffer) > max(batch_size, learning_starts):
                strata, _, _ = agent.train(replay_buffer, batch_size)
                straty_w_epizodzie.append(strata)

            state = next_state
            episode_reward += reward

        historia_nagrod.append(episode_reward)
        historia_krokow.append(kroki_w_epizodzie) 
        historia_straty.append(sum(straty_w_epizodzie) / len(straty_w_epizodzie) if straty_w_epizodzie else 0)

        if episode > 0 and episode % 100 == 0:
            print(f"[DDPG Własna | Seed {seed}] Epizod {episode}/{episodes}")

    if out is not None: out.release()
    env.close()
    return historia_nagrod, historia_straty, historia_krokow



# TD3


class TD3_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(TD3_Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.layer2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.layer3 = nn.Linear(256, action_dim)

        torch.nn.init.uniform_(self.layer3.weight, a=-3e-3, b=3e-3)
        torch.nn.init.uniform_(self.layer3.bias, a=-3e-3, b=3e-3)
        self.max_action = max_action

    def forward(self, state):
        a = self.layer1(state)
        a = F.relu(self.ln1(a))
        a = self.layer2(a)
        a = F.relu(self.ln2(a))
        a = torch.tanh(self.layer3(a))
        return self.max_action * a

class TD3_Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(TD3_Critic, self).__init__()
        # Krytyk 1
        self.layer1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.layer2 = nn.Linear(256 + action_dim, 256)
        self.layer3 = nn.Linear(256, 1)
        torch.nn.init.uniform_(self.layer3.weight, a=-3e-3, b=3e-3)
        torch.nn.init.uniform_(self.layer3.bias, a=-3e-3, b=3e-3)

        # Krytyk 2
        self.layer4 = nn.Linear(state_dim, 256)
        self.ln4 = nn.LayerNorm(256)
        self.layer5 = nn.Linear(256 + action_dim, 256)
        self.layer6 = nn.Linear(256, 1)
        torch.nn.init.uniform_(self.layer6.weight, a=-3e-3, b=3e-3)
        torch.nn.init.uniform_(self.layer6.bias, a=-3e-3, b=3e-3)

    def forward(self, state, action):
        q1 = self.layer1(state)
        q1 = F.relu(self.ln1(q1))
        q1 = torch.cat([q1, action], dim=1)
        q1 = F.relu(self.layer2(q1))
        q1 = self.layer3(q1)

        q2 = self.layer4(state)
        q2 = F.relu(self.ln4(q2))
        q2 = torch.cat([q2, action], dim=1)
        q2 = F.relu(self.layer5(q2))
        q2 = self.layer6(q2)
        return q1, q2
    
    def Q1(self, state, action):
        q1 = self.layer1(state)
        q1 = F.relu(self.ln1(q1))
        q1 = torch.cat([q1, action], dim=1)
        q1 = F.relu(self.layer2(q1))
        q1 = self.layer3(q1)
        return q1

class TD3:
    def __init__(self, state_dim, action_dim, max_action, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.actor = TD3_Actor(state_dim, action_dim, max_action).to(device)
        self.critic = TD3_Critic(state_dim, action_dim).to(device)
        self.actor_target = TD3_Actor(state_dim, action_dim, max_action).to(device)
        self.critic_target = TD3_Critic(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)

        self.max_action = max_action
        self.gamma = 0.99
        self.tau = 0.001
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

    def select_action(self, state, noise_std=0.1, noise_type='gauss', ou_noise=None):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        with torch.no_grad():
            action = self.actor(state).numpy()[0]

        if noise_type == 'gauss':
            noise = np.random.normal(0, max(self.max_action * noise_std, 0), size=action.shape)
            action = action + noise
        elif noise_type == 'ou' and ou_noise is not None:
            noise = ou_noise() * max(noise_std, 0)
            action = action + noise

        return np.clip(action, -self.max_action, self.max_action)

    def train(self, replay_buffer, batch_size=128):
        self.total_it += 1
        state, action, reward, next_state, terminated = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - terminated) * self.gamma * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_val = 0
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            actor_loss_val = actor_loss.item()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss_val, current_Q1.mean().item()

def train_td3(seed, batch_size=64, episodes=10001, record_video=False, noise_type='gauss', reward_scale=0.1, learning_starts=2000):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    render_mode = "rgb_array" if record_video else None
    env = gym.make("BipedalWalker-v3", render_mode=render_mode)
    env.reset(seed=seed)
    out = None

    if record_video:
        sample_frame = env.render()
        wysokosc, szerokosc, _ = sample_frame.shape
        out = cv2.VideoWriter(f'ewolucja_td3_seed_{seed}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (szerokosc, wysokosc))
        
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD3(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    ou_noise = None

    historia_nagrod, historia_straty, historia_krokow = [], [], []
    start_noise, end_noise, exploration_fraction = 0.2, 0.05, 0.9 
    noise_decay = (start_noise - end_noise) / int(episodes * exploration_fraction)
    epizody_do_nagrania = [0, 100, 1000, 2000, 4000, 8000]

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward, kroki_w_epizodzie = 0, 0
        straty_w_epizodzie = []
        terminated, truncated = False, False
        current_noise = max(end_noise, start_noise - episode * noise_decay)

        while not (terminated or truncated):
            kroki_w_epizodzie += 1 
            if len(replay_buffer) < learning_starts:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, noise_std=current_noise, noise_type=noise_type, ou_noise=ou_noise)

            next_state, reward, terminated, truncated, _ = env.step(action)
            replay_buffer.add(state, action, reward * reward_scale, next_state, terminated)

            if record_video and (episode in epizody_do_nagrania):
                frame = env.render()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.putText(frame_bgr, f"TD3 Epizod: {episode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                out.write(frame_bgr)

            if len(replay_buffer) > max(batch_size, learning_starts):
                strata, _, _ = agent.train(replay_buffer, batch_size)
                straty_w_epizodzie.append(strata)

            state = next_state
            episode_reward += reward

        historia_nagrod.append(episode_reward)
        historia_krokow.append(kroki_w_epizodzie) 
        historia_straty.append(sum(straty_w_epizodzie) / len(straty_w_epizodzie) if straty_w_epizodzie else 0)

        if episode > 0 and episode % 100 == 0:
            print(f"[TD3 Własna | Seed {seed}] Zakończono epizod {episode}/{episodes}")

    if out is not None: out.release()
    env.close()
    return historia_nagrod, historia_straty, historia_krokow



# PPO 


class PPO_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(PPO_Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.layer2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.layer3 = nn.Linear(256, action_dim)
        
        self.layer1.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        self.layer2.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        self.layer3.apply(lambda m: init_weights(m, gain=0.01))
        
        self.max_action = max_action
        self.action_var = nn.Parameter(torch.full((action_dim,), 0.0))

    def forward(self, state):
        a = self.layer1(state)
        a = F.relu(self.ln1(a))
        a = self.layer2(a)
        a = F.relu(self.ln2(a))
        mean = torch.tanh(self.layer3(a)) * self.max_action
        return mean

class PPO_Critic(nn.Module):
    def __init__(self, state_dim):
        super(PPO_Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)

        self.layer1.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        self.layer2.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
        self.layer3.apply(lambda m: init_weights(m, gain=1.0))

    def forward(self, state):
        v = self.layer1(state)
        v = F.relu(self.ln1(v))
        v = F.relu(self.layer2(v))
        v = self.layer3(v)
        return v

class PPO:
    def __init__(self, state_dim, action_dim, max_action, lr_actor=3e-4, lr_critic=3e-4, gamma=0.99, gae_lambda=0.95, K_epochs=10, eps_clip=0.2, batch_size=64, entropy_coef=0.0):
        self.actor = PPO_Actor(state_dim, action_dim, max_action).to(device)
        self.critic = PPO_Critic(state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.entropy_coef = entropy_coef

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        with torch.no_grad():
            mean = self.actor(state)
            cov_mat = torch.diag(self.actor.action_var.exp()).unsqueeze(dim=0)
            dist = MultivariateNormal(mean, cov_mat)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            state_value = self.critic(state)
        return action.cpu().numpy()[0], action_logprob.cpu().numpy()[0], state_value.cpu().numpy()[0]

    def evaluate(self, state, action):
        mean = self.actor(state)
        action_var = self.actor.action_var.exp()
        cov_mat = torch.diag_embed(action_var).expand(state.size(0), self.action_dim, self.action_dim)
        dist = MultivariateNormal(mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values.squeeze(), dist_entropy

    def train(self, rollout_buffer):
        old_states, old_actions, old_logprobs, rewards, state_values, terminated = rollout_buffer.get_data()
        rewards_np = rewards.cpu().numpy().flatten()
        values_np = state_values.cpu().numpy().flatten()
        terminated_np = terminated.cpu().numpy().flatten()
        
        advantages_np = np.zeros(len(rewards_np), dtype=np.float32)
        last_gae_lam = 0
        
        for t in reversed(range(len(rewards_np))):
            next_non_terminal = 1.0 - terminated_np[t]
            next_value = 0.0 if t == len(rewards_np) - 1 else values_np[t + 1]
            delta = rewards_np[t] + self.gamma * next_value * next_non_terminal - values_np[t]
            advantages_np[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            
        rewards_to_go = advantages_np + values_np
        advantages = torch.tensor(advantages_np, dtype=torch.float32).to(device)
        rewards_to_go = torch.tensor(rewards_to_go, dtype=torch.float32).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        critic_losses, actor_losses = [], []
        dataset_size = old_states.size(0)

        for _ in range(self.K_epochs):
            indices = torch.randperm(dataset_size)
            for start_idx in range(0, dataset_size, self.batch_size):
                batch_idx = indices[start_idx:start_idx + self.batch_size]
                b_states, b_actions = old_states[batch_idx], old_actions[batch_idx]
                b_logprobs = old_logprobs[batch_idx].squeeze()
                b_advantages, b_returns = advantages[batch_idx], rewards_to_go[batch_idx]

                logprobs, state_values_pred, dist_entropy = self.evaluate(b_states, b_actions)
                ratios = torch.exp(logprobs - b_logprobs)

                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * b_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * dist_entropy.mean()
                critic_loss = F.mse_loss(state_values_pred, b_returns)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                
                critic_losses.append(critic_loss.item())
                actor_losses.append(actor_loss.item())

        rollout_buffer.clear()
        return np.mean(critic_losses), np.mean(actor_losses)

def train_ppo(seed, update_timestep=2048, episodes=10001, record_video=False, reward_scale=0.1, entropy_coef=0.0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    render_mode = "rgb_array" if record_video else None
    env = gym.make("BipedalWalker-v3", render_mode=render_mode)
    env.reset(seed=seed)
    out = None

    if record_video:
        sample_frame = env.render()
        wysokosc, szerokosc, _ = sample_frame.shape
        out = cv2.VideoWriter(f'ewolucja_ppo_seed_{seed}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (szerokosc, wysokosc))

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = PPO(state_dim, action_dim, max_action, entropy_coef=entropy_coef)
    rollout_buffer = RolloutBuffer(state_dim, action_dim, max_size=update_timestep)

    historia_nagrod, historia_straty, historia_krokow = [], [], []
    time_step = 0
    epizody_do_nagrania = [0, 100, 1000, 2000, 4000, 8000]

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward, kroki_w_epizodzie = 0, 0
        straty_w_epizodzie = []
        terminated, truncated = False, False

        while not (terminated or truncated):
            action, action_logprob, state_value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            rollout_buffer.add(state, action, action_logprob, reward * reward_scale, state_value, terminated)
            
            if record_video and (episode in epizody_do_nagrania):
                frame = env.render()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.putText(frame_bgr, f"PPO Epizod: {episode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                out.write(frame_bgr)

            state = next_state
            episode_reward += reward
            time_step += 1
            kroki_w_epizodzie += 1 

            if time_step % update_timestep == 0:
                strata_krytyka, _ = agent.train(rollout_buffer)
                straty_w_epizodzie.append(strata_krytyka)

        historia_nagrod.append(episode_reward)
        historia_krokow.append(kroki_w_epizodzie) 
        historia_straty.append(sum(straty_w_epizodzie)/len(straty_w_epizodzie) if straty_w_epizodzie else 0)

        if episode > 0 and episode % 100 == 0:
            print(f"[PPO Własna | Seed {seed}] Zakończono epizod {episode}/{episodes}")

    if out is not None: out.release()
    env.close()
    return historia_nagrod, historia_straty, historia_krokow





def train_sb3_algorithm(algo_name, seed, episodes):
    from stable_baselines3 import DDPG, TD3, PPO
    from stable_baselines3.common.monitor import Monitor
    env = Monitor(gym.make("BipedalWalker-v3"))
    env.reset(seed=seed)
    
    logger = RewardLoggerCallback(max_episodes=episodes)
    
    if algo_name == "DDPG":
        model = DDPG("MlpPolicy", env, seed=seed, device='cpu')
    elif algo_name == "TD3":
        model = TD3("MlpPolicy", env, seed=seed, device='cpu')
    elif algo_name == "PPO":
        model = PPO("MlpPolicy", env, seed=seed, device='cpu')
        
    model.learn(total_timesteps=1600 * episodes, callback=logger)
    env.close()
    return logger.episode_rewards[:episodes], logger.episode_lengths[:episodes]




if __name__ == '__main__':
    seedy = [1, 42, 123, 1234, 999]
    liczba_epizodow = 8001
    domyslny_batch = 256
    domyslny_update = 2048
    testowy_seed = seedy[0]

    print("Rozpoczęcie wielowątkowego pipeline'u badań (DDPG -> TD3 -> PPO)")

    # 1. DDPG
    print("\n--- 1. Uruchamianie algorytmu DDPG ---")
    wyniki_ddpg_nagrody, wyniki_ddpg_straty, wyniki_ddpg_kroki = {}, {}, {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        f_ddpg = {executor.submit(train_ddpg, s, domyslny_batch, liczba_epizodow, s == testowy_seed): s for s in seedy}
        for future in concurrent.futures.as_completed(f_ddpg):
            seed = f_ddpg[future]
            nagrody, straty, kroki = future.result()
            np.savez(f'temp_ddpg_wlasna_seed_{seed}.npz', nagrody=nagrody, straty=straty, kroki=kroki)
            wyniki_ddpg_nagrody[f"Seed {seed}"] = nagrody
            wyniki_ddpg_straty[f"Seed {seed}"] = straty
            wyniki_ddpg_kroki[f"Seed {seed}"] = kroki
            print(f"[DDPG Własna] Zapisano dane dla seed: {seed}")
            
    np.savez('dane_ddpg_wlasna_seedy.npz', **wyniki_ddpg_nagrody)
    np.savez('dane_ddpg_wlasna_straty.npz', **wyniki_ddpg_straty)
    np.savez('dane_ddpg_wlasna_kroki.npz', **wyniki_ddpg_kroki)

    wyniki_sb3_ddpg_nagrody, wyniki_sb3_ddpg_kroki = {}, {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        f_sb3_ddpg = {executor.submit(train_sb3_algorithm, "DDPG", s, liczba_epizodow): s for s in seedy}
        for future in concurrent.futures.as_completed(f_sb3_ddpg):
            seed = f_sb3_ddpg[future]
            nagrody, kroki = future.result()
            np.savez(f'temp_ddpg_sb3_seed_{seed}.npz', nagrody=nagrody, kroki=kroki)
            wyniki_sb3_ddpg_nagrody[f"SB3 DDPG Seed {seed}"] = nagrody
            wyniki_sb3_ddpg_kroki[f"SB3 DDPG Seed {seed}"] = kroki
            print(f"[SB3 DDPG] Zapisano dane dla seed: {seed}")
            
    np.savez('dane_ddpg_sb3_seedy.npz', **wyniki_sb3_ddpg_nagrody)
    np.savez('dane_ddpg_sb3_kroki.npz', **wyniki_sb3_ddpg_kroki)

    # 2. TD3
    print("\n--- 2. Uruchamianie algorytmu TD3 ---")
    wyniki_td3_nagrody, wyniki_td3_straty, wyniki_td3_kroki = {}, {}, {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        f_td3 = {executor.submit(train_td3, s, domyslny_batch, liczba_epizodow, s == testowy_seed): s for s in seedy}
        for future in concurrent.futures.as_completed(f_td3):
            seed = f_td3[future]
            nagrody, straty, kroki = future.result()
            np.savez(f'temp_td3_wlasna_seed_{seed}.npz', nagrody=nagrody, straty=straty, kroki=kroki)
            wyniki_td3_nagrody[f"Seed {seed}"] = nagrody
            wyniki_td3_straty[f"Seed {seed}"] = straty
            wyniki_td3_kroki[f"Seed {seed}"] = kroki
            print(f"[TD3 Własna] Zapisano dane dla seed: {seed}")
            
    np.savez('dane_td3_wlasna_seedy.npz', **wyniki_td3_nagrody)
    np.savez('dane_td3_wlasna_straty.npz', **wyniki_td3_straty)
    np.savez('dane_td3_wlasna_kroki.npz', **wyniki_td3_kroki)

    wyniki_sb3_td3_nagrody, wyniki_sb3_td3_kroki = {}, {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        f_sb3_td3 = {executor.submit(train_sb3_algorithm, "TD3", s, liczba_epizodow): s for s in seedy}
        for future in concurrent.futures.as_completed(f_sb3_td3):
            seed = f_sb3_td3[future]
            nagrody, kroki = future.result()
            np.savez(f'temp_td3_sb3_seed_{seed}.npz', nagrody=nagrody, kroki=kroki)
            wyniki_sb3_td3_nagrody[f"SB3 TD3 Seed {seed}"] = nagrody
            wyniki_sb3_td3_kroki[f"SB3 TD3 Seed {seed}"] = kroki
            print(f"[SB3 TD3] Zapisano dane dla seed: {seed}")
            
    np.savez('dane_td3_sb3_seedy.npz', **wyniki_sb3_td3_nagrody)
    np.savez('dane_td3_sb3_kroki.npz', **wyniki_sb3_td3_kroki)

    # 3. PPO
    print("\n--- 3. Uruchamianie algorytmu PPO ---")
    wyniki_ppo_nagrody, wyniki_ppo_straty, wyniki_ppo_kroki = {}, {}, {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        f_ppo = {executor.submit(train_ppo, s, domyslny_update, liczba_epizodow, s == testowy_seed): s for s in seedy}
        for future in concurrent.futures.as_completed(f_ppo):
            seed = f_ppo[future]
            nagrody, straty, kroki = future.result()
            np.savez(f'temp_ppo_wlasna_seed_{seed}.npz', nagrody=nagrody, straty=straty, kroki=kroki)
            wyniki_ppo_nagrody[f"Seed {seed}"] = nagrody
            wyniki_ppo_straty[f"Seed {seed}"] = straty
            wyniki_ppo_kroki[f"Seed {seed}"] = kroki
            print(f"[PPO Własna] Zapisano dane dla seed: {seed}")
            
    np.savez('dane_ppo_wlasna_seedy.npz', **wyniki_ppo_nagrody)
    np.savez('dane_ppo_wlasna_straty.npz', **wyniki_ppo_straty)
    np.savez('dane_ppo_wlasna_kroki.npz', **wyniki_ppo_kroki)

    wyniki_sb3_ppo_nagrody, wyniki_sb3_ppo_kroki = {}, {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        f_sb3_ppo = {executor.submit(train_sb3_algorithm, "PPO", s, liczba_epizodow): s for s in seedy}
        for future in concurrent.futures.as_completed(f_sb3_ppo):
            seed = f_sb3_ppo[future]
            nagrody, kroki = future.result()
            np.savez(f'temp_ppo_sb3_seed_{seed}.npz', nagrody=nagrody, kroki=kroki)
            wyniki_sb3_ppo_nagrody[f"SB3 PPO Seed {seed}"] = nagrody
            wyniki_sb3_ppo_kroki[f"SB3 PPO Seed {seed}"] = kroki
            print(f"[SB3 PPO] Zapisano dane dla seed: {seed}")
            
    np.savez('dane_ppo_sb3_seedy.npz', **wyniki_sb3_ppo_nagrody)
    np.savez('dane_ppo_sb3_kroki.npz', **wyniki_sb3_ppo_kroki)

    print("\nZakończono pomyślnie cały pipeline.")