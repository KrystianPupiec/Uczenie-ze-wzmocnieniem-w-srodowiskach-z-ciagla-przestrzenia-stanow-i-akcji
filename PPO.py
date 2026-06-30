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

class RolloutBuffer:
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

def init_weights(m, gain=1.0):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0.0)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
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

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
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
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim).to(device)

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
            if t == len(rewards_np) - 1:
                next_non_terminal = 1.0 - terminated_np[t]
                next_value = 0.0
            else:
                next_non_terminal = 1.0 - terminated_np[t]
                next_value = values_np[t + 1]
                
            delta = rewards_np[t] + self.gamma * next_value * next_non_terminal - values_np[t]
            advantages_np[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            
        rewards_to_go = advantages_np + values_np
        
        advantages = torch.tensor(advantages_np, dtype=torch.float32).to(device)
        rewards_to_go = torch.tensor(rewards_to_go, dtype=torch.float32).to(device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        critic_losses = []
        actor_losses = []
        dataset_size = old_states.size(0)

        for _ in range(self.K_epochs):
            indices = torch.randperm(dataset_size)
            
            for start_idx in range(0, dataset_size, self.batch_size):
                batch_idx = indices[start_idx:start_idx + self.batch_size]
                
                b_states = old_states[batch_idx]
                b_actions = old_actions[batch_idx]
                b_logprobs = old_logprobs[batch_idx].squeeze()
                b_advantages = advantages[batch_idx]
                b_returns = rewards_to_go[batch_idx]

                logprobs, state_values_pred, dist_entropy = self.evaluate(b_states, b_actions)

                ratios = torch.exp(logprobs - b_logprobs)

                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * b_advantages
                
                # Użycie parametru entropii
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
    epizody_do_nagrania = [0, 1, 2, 3, 5, 10, 50, 100, 101, 102, 103, 105, 150, 155, 300, 400, 500, 600,700, 800, 900, 1000, 1001, 1002, 1003, 2000, 3000, 4000, 4999, 5000, 7000, 7050, 7080, 7900, 7995, 7996, 7997, 7998, 7999, 8000]

    if record_video:
        sample_frame = env.render()
        wysokosc, szerokosc, _ = sample_frame.shape
        frame_size = (szerokosc, wysokosc)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'ewolucja_ppo_seed_{seed}.mp4', fourcc, 30.0, frame_size)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = PPO(state_dim, action_dim, max_action, entropy_coef=entropy_coef)
    rollout_buffer = RolloutBuffer(state_dim, action_dim, max_size=update_timestep)

    historia_nagrod = []
    historia_straty = []
    historia_krokow = [] 

    time_step = 0

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        straty_w_epizodzie = []
        terminated = False
        truncated = False
        kroki_w_epizodzie = 0 

        while not (terminated or truncated):
            action, action_logprob, state_value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            scaled_reward = reward * reward_scale

            if record_video and (episode in epizody_do_nagrania):
                frame = env.render()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.putText(frame_bgr, f"Epizod: {episode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
                out.write(frame_bgr)

            rollout_buffer.add(state, action, action_logprob, scaled_reward, state_value, terminated)
            
            state = next_state
            episode_reward += reward
            time_step += 1
            kroki_w_epizodzie += 1 

            if time_step % update_timestep == 0:
                strata_krytyka, strata_aktora = agent.train(rollout_buffer)
                straty_w_epizodzie.append(strata_krytyka)

        historia_nagrod.append(episode_reward)
        historia_krokow.append(kroki_w_epizodzie) 

        if len(straty_w_epizodzie) > 0:
            srednia_straty = sum(straty_w_epizodzie) / len(straty_w_epizodzie)
            historia_straty.append(srednia_straty)
        else:
            historia_straty.append(0)

        if episode > 0 and episode % 100 == 0:
            print(f"[Proces: Seed {seed} | Update {update_timestep}] Zakończono epizod {episode}/{episodes}")

    if out is not None:
        out.release()
    env.close()

    return historia_nagrod, historia_straty, historia_krokow


def train_sb3_ppo(seed, episodes=10001):
    from stable_baselines3 import PPO as SB3_PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback

    class RewardLoggerCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episode_rewards = []
            self.episode_lengths = [] 

        def _on_step(self):
            if 'episode' in self.locals['infos'][0]:
                self.episode_rewards.append(self.locals['infos'][0]['episode']['r'])
                self.episode_lengths.append(self.locals['infos'][0]['episode']['l']) 
            return True

    env = gym.make("BipedalWalker-v3")
    env = Monitor(env)
    env.reset(seed=seed)

    model = SB3_PPO("MlpPolicy", env, seed=seed, learning_rate=3e-4, gamma=0.99, n_steps=2048, batch_size=64, device='cpu')
    logger = RewardLoggerCallback()
    total_steps = 200 * episodes
    model.learn(total_timesteps=total_steps, callback=logger)

    env.close()
    
    return logger.episode_rewards[:episodes], logger.episode_lengths[:episodes]


if __name__ == '__main__':
    seedy = [1, 42, 123, 1234, 999]
    liczba_epizodow = 8001
    domyslny_update = 2048
    testowy_seed = seedy[0]

    print("1. Zbieranie danych dla różnych seedów w wielu procesach")
    wyniki_seedy_nagrody = {}
    wyniki_seedy_straty = {}
    wyniki_seedy_kroki = {} 
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        przyszle_seedy = {executor.submit(train_ppo, seed, domyslny_update, liczba_epizodow, seed == testowy_seed): seed for seed in seedy}
        
        for future in concurrent.futures.as_completed(przyszle_seedy):
            seed = przyszle_seedy[future]
            nagrody, straty, kroki = future.result() 
            
            np.savez(f'temp_1_seed_{seed}_ppo.npz', nagrody=nagrody, straty=straty, kroki=kroki)
            
            wyniki_seedy_nagrody[f"Seed {seed}"] = nagrody
            wyniki_seedy_straty[f"Seed {seed}"] = straty
            wyniki_seedy_kroki[f"Seed {seed}"] = kroki
            print(f"Ukończono seed {seed}")

    np.savez('dane_seedy_ppo.npz', **wyniki_seedy_nagrody)
    np.savez('dane_seedy_straty_ppo.npz', **wyniki_seedy_straty)
    np.savez('dane_seedy_kroki_ppo.npz', **wyniki_seedy_kroki) 

    print("2. Zbieranie danych dla różnych timestep update (odpowiednik batch)")
    testowane_update = [512, 1024, 2048, 4096]
    wyniki_update = {}
    wyniki_update_straty = {}
    wyniki_update_kroki = {} 
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        przyszle_update = {executor.submit(train_ppo, testowy_seed, upd, liczba_epizodow, False): upd for upd in testowane_update}
        
        for future in concurrent.futures.as_completed(przyszle_update):
            upd = przyszle_update[future]
            nagrody, straty, kroki = future.result()
            
            np.savez(f'temp_2_update_{upd}_ppo.npz', nagrody=nagrody, straty=straty, kroki=kroki) 
            
            wyniki_update[f"Update {upd}"] = nagrody
            wyniki_update_straty[f"Update {upd}"] = straty
            wyniki_update_kroki[f"Update {upd}"] = kroki
            print(f"Ukończono update {upd}")
            
    np.savez('dane_update_ppo.npz', **wyniki_update)
    np.savez('dane_update_straty_ppo.npz', **wyniki_update_straty)
    np.savez('dane_update_kroki_ppo.npz', **wyniki_update_kroki) 

    print("3. Zbieranie danych dla różnych współczynników skalowania nagrody")
    skale_nagrody = [0.1, 1.0, 10.0]
    wyniki_skale = {}
    wyniki_skale_straty = {}
    wyniki_skale_kroki = {} 
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        przyszle_skale = {executor.submit(train_ppo, testowy_seed, domyslny_update, liczba_epizodow, False, skala): skala for skala in skale_nagrody}
        
        for future in concurrent.futures.as_completed(przyszle_skale):
            skala = przyszle_skale[future]
            nagrody, straty, kroki = future.result()
            
            np.savez(f'temp_3_skala_{skala}_ppo.npz', nagrody=nagrody, straty=straty, kroki=kroki) 
            
            wyniki_skale[f"Skala {skala}"] = nagrody
            wyniki_skale_straty[f"Skala {skala}"] = straty
            wyniki_skale_kroki[f"Skala {skala}"] = kroki
            print(f"Ukończono skalę {skala}")
            
    np.savez('dane_skale_ppo.npz', **wyniki_skale)
    np.savez('dane_skale_straty_ppo.npz', **wyniki_skale_straty)
    np.savez('dane_skale_kroki_ppo.npz', **wyniki_skale_kroki) 

    print("4. Zbieranie danych dla różnych współczynników entropii")
    wspolczynniki_entropii = [0.0, 0.01]
    wyniki_entropia = {}
    wyniki_entropia_straty = {}
    wyniki_entropia_kroki = {}
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        przyszle_entropia = {executor.submit(train_ppo, testowy_seed, domyslny_update, liczba_epizodow, False, 0.1, coef): coef for coef in wspolczynniki_entropii}
        
        for future in concurrent.futures.as_completed(przyszle_entropia):
            coef = przyszle_entropia[future]
            nagrody, straty, kroki = future.result()
            
            np.savez(f'temp_4_entropia_{coef}_ppo.npz', nagrody=nagrody, straty=straty, kroki=kroki) 
            
            wyniki_entropia[f"Entropia {coef}"] = nagrody
            wyniki_entropia_straty[f"Entropia {coef}"] = straty
            wyniki_entropia_kroki[f"Entropia {coef}"] = kroki
            print(f"Ukończono współczynnik entropii {coef}")
            
    np.savez('dane_entropia_ppo.npz', **wyniki_entropia)
    np.savez('dane_entropia_straty_ppo.npz', **wyniki_entropia_straty)
    np.savez('dane_entropia_kroki_ppo.npz', **wyniki_entropia_kroki) 

    print("5. Zbieranie danych ze Stable-Baselines3")
    sb3_nagrody, sb3_kroki = train_sb3_ppo(testowy_seed, episodes=liczba_epizodow)
    
    np.savez('temp_5_sb3_ppo.npz', nagrody=sb3_nagrody, kroki=sb3_kroki) 
    
    wyniki_sb3 = {
        "Własna implementacja PPO": wyniki_seedy_nagrody[f"Seed {testowy_seed}"],
        "SB3 PPO": sb3_nagrody
    }
    wyniki_sb3_kroki = {
        "Własna implementacja PPO": wyniki_seedy_kroki[f"Seed {testowy_seed}"],
        "SB3 PPO": sb3_kroki
    }
    np.savez('dane_sb3_ppo.npz', **wyniki_sb3)
    np.savez('dane_sb3_kroki_ppo.npz', **wyniki_sb3_kroki) 
    print("Zakończono")