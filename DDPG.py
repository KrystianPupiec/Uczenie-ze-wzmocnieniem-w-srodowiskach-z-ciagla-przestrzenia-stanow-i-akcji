import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import gymnasium as gym
# Do zrównoleglenia procesów
import concurrent.futures  
import os

# 1. Wymuszenie procesora i ograniczenie wątków na proces
device = torch.device("cpu")
torch.set_num_threads(1)

# 2.  bufor na prealokowanych tablicach numpy zamiast listy
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=1000000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        # Prealokacja pamięci
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.terminated = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, terminated):
        # Nadpisywanie starych danych w pętli
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.terminated[self.ptr] = terminated
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        # Szybkie losowanie indeksów
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

# Aktor
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
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

# Krytyk
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
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

# DDPG
class DDPG:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)

        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
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

    # 3. Zmiana renderowania środowiska - wymuszony tryb graficzny tylko jeśli seed jest nagrywany
    render_mode = "rgb_array" if record_video else None
    env = gym.make("BipedalWalker-v3", render_mode=render_mode)
    env.reset(seed=seed)

    out = None
    epizody_do_nagrania = [0, 1, 2, 3, 5, 10, 50, 100, 150, 1000, 2000, 3000, 4000, 4999, 5000, 7000, 7050, 7080, 7900, 7995, 7996, 7997, 7998, 7999, 8000]

    if record_video:
        sample_frame = env.render()
        wysokosc, szerokosc, _ = sample_frame.shape
        frame_size = (szerokosc, wysokosc)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'ewolucja_ddpg_seed_{seed}.mp4', fourcc, 30.0, frame_size)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = DDPG(state_dim, action_dim, max_action)
    
    # Przekazujemy wymiary do nowego bufora pamięci
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    if noise_type == 'ou':
        from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
        mean = np.zeros(action_dim)
        sigma = np.ones(action_dim) * 0.2
        ou_noise = OrnsteinUhlenbeckActionNoise(mean=mean, sigma=sigma, theta=0.15)
    else:
        ou_noise = None

    historia_nagrod = []
    historia_straty = []
    historia_krokow = [] 

    start_noise = 0.2
    end_noise = 0.05
    exploration_fraction = 0.9 
    decay_steps = int(episodes * exploration_fraction)
    noise_decay = (start_noise - end_noise) / decay_steps

    for episode in range(episodes):
        if ou_noise is not None:
            ou_noise.reset()

        state, _ = env.reset()
        episode_reward = 0
        straty_w_epizodzie = []
        terminated = False
        truncated = False
        
        kroki_w_epizodzie = 0 

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
                cv2.putText(frame_bgr, f"Epizod: {episode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
                out.write(frame_bgr)

            replay_buffer.add(state, action, scaled_reward, next_state, terminated)

            if len(replay_buffer) > max(batch_size, learning_starts):
                strata, _, _ = agent.train(replay_buffer, batch_size)
                straty_w_epizodzie.append(strata)

            state = next_state
            episode_reward += reward

        historia_nagrod.append(episode_reward)
        historia_krokow.append(kroki_w_epizodzie) 

        if len(straty_w_epizodzie) > 0:
            srednia_straty = sum(straty_w_epizodzie) / len(straty_w_epizodzie)
            historia_straty.append(srednia_straty)
        else:
            historia_straty.append(0)

        if episode > 0 and episode % 100 == 0:
            print(f"[Proces: Seed {seed} | Batch {batch_size} | Szum {noise_type}] Zakończono epizod {episode}/{episodes}")

    if out is not None:
        out.release()
    env.close()

    return historia_nagrod, historia_straty, historia_krokow


def train_sb3_ddpg(seed, episodes=10001):
    from stable_baselines3 import DDPG as SB3_DDPG
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

    # Nie włączamy tutaj renderowania graficznego
    env = gym.make("BipedalWalker-v3")
    env = Monitor(env)
    env.reset(seed=seed)

    # ustawienie argumentu `device='cpu'` dla SB3
    model = SB3_DDPG("MlpPolicy", env, seed=seed, learning_rate=1e-3, gamma=0.99, tau=0.001, batch_size=128, device='cpu')
    logger = RewardLoggerCallback()
    total_steps = 200 * episodes
    model.learn(total_timesteps=total_steps, callback=logger)

    env.close()
    
    return logger.episode_rewards[:episodes], logger.episode_lengths[:episodes]


# 4. Dodanie bloku __main__
if __name__ == '__main__':
    seedy = [1, 42, 123, 1234, 999]
    liczba_epizodow = 8001
    domyslny_batch = 256
    testowy_seed = seedy[0]

    # Zastosowanie ProcessPoolExecutor by wrzucać eksperymenty równolegle
    print("1A. Zbieranie danych dla różnych seedów w wielu procesach")
    wyniki_seedy_nagrody = {}
    wyniki_seedy_straty = {}
    wyniki_seedy_kroki = {} 
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        przyszle_seedy = {executor.submit(train_ddpg, seed, domyslny_batch, liczba_epizodow, seed == testowy_seed): seed for seed in seedy}
        
        for future in concurrent.futures.as_completed(przyszle_seedy):
            seed = przyszle_seedy[future]
            nagrody, straty, kroki = future.result() 
            
            # Zapis tymczasowy
            np.savez(f'temp_1a_seed_{seed}.npz', nagrody=nagrody, straty=straty, kroki=kroki)
            
            wyniki_seedy_nagrody[f"Seed {seed}"] = nagrody
            wyniki_seedy_straty[f"Seed {seed}"] = straty
            wyniki_seedy_kroki[f"Seed {seed}"] = kroki
            print(f"Ukończono seed {seed}")

    np.savez('dane_seedy.npz', **wyniki_seedy_nagrody)
    np.savez('dane_seedy_straty.npz', **wyniki_seedy_straty)
    np.savez('dane_seedy_kroki.npz', **wyniki_seedy_kroki) 


    # Zbieranie danych dla różnych seedów, ale z szumem OU
    print("1B. Zbieranie danych dla różnych seedów (Szum OU)")
    wyniki_seedy_nagrody_ou = {}
    wyniki_seedy_straty_ou = {}
    wyniki_seedy_kroki_ou = {} 
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        przyszle_seedy_ou = {
            executor.submit(train_ddpg, seed, domyslny_batch, liczba_epizodow, False, 'ou'): seed for seed in seedy}
        
        for future in concurrent.futures.as_completed(przyszle_seedy_ou):
            seed = przyszle_seedy_ou[future]
            nagrody, straty, kroki = future.result() 
            
            # Zapis tymczasowy
            np.savez(f'temp_1b_seed_{seed}_ou.npz', nagrody=nagrody, straty=straty, kroki=kroki) 
            
            wyniki_seedy_nagrody_ou[f"Seed {seed}"] = nagrody
            wyniki_seedy_straty_ou[f"Seed {seed}"] = straty
            wyniki_seedy_kroki_ou[f"Seed {seed}"] = kroki
            print(f"Ukończono seed {seed} z szumem OU")

    np.savez('dane_seedy_ou.npz', **wyniki_seedy_nagrody_ou)
    np.savez('dane_seedy_straty_ou.npz', **wyniki_seedy_straty_ou)
    np.savez('dane_seedy_kroki_ou.npz', **wyniki_seedy_kroki_ou) 


    print("2. Zbieranie danych dla różnych batch_size")
    testowane_batche = [32, 64, 128, 256, 512]
    wyniki_batche = {}
    wyniki_batche_kroki = {} 
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        przyszle_batche = {executor.submit(train_ddpg, testowy_seed, bs, liczba_epizodow, False): bs for bs in testowane_batche}
        
        for future in concurrent.futures.as_completed(przyszle_batche):
            bs = przyszle_batche[future]
            res = future.result()
            nagrody = res[0]
            straty = res[1] 
            kroki = res[2]  
            
            # Zapis tymczasowy
            np.savez(f'temp_2_batch_{bs}.npz', nagrody=nagrody, straty=straty, kroki=kroki) 
            
            wyniki_batche[f"Batch {bs}"] = nagrody
            wyniki_batche_kroki[f"Batch {bs}"] = kroki
            print(f"Ukończono batch {bs}")
            
    np.savez('dane_batche.npz', **wyniki_batche)
    np.savez('dane_batche_kroki.npz', **wyniki_batche_kroki) 


    print("3. Zbieranie danych dla różnych procesów szumu")
    rodzaje_szumu = ['gauss', 'ou', 'epsilon_greedy', 'none']
    wyniki_szumy = {}
    wyniki_szumy_kroki = {} 
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        przyszle_szumy = {executor.submit(train_ddpg, testowy_seed, domyslny_batch, liczba_epizodow, False, szum): szum for szum in rodzaje_szumu}
        
        for future in concurrent.futures.as_completed(przyszle_szumy):
            szum = przyszle_szumy[future]
            res = future.result()
            nagrody = res[0]
            straty = res[1] 
            kroki = res[2]  
            
            # Zapis tymczasowy
            np.savez(f'temp_3_szum_{szum}.npz', nagrody=nagrody, straty=straty, kroki=kroki) 
            
            wyniki_szumy[f"Szum {szum}"] = nagrody
            wyniki_szumy_kroki[f"Szum {szum}"] = kroki
            print(f"Ukończono szum {szum}")
            
    np.savez('dane_szumy.npz', **wyniki_szumy)
    np.savez('dane_szumy_kroki.npz', **wyniki_szumy_kroki) 


    print("4. Zbieranie danych dla różnych współczynników skalowania nagrody")
    skale_nagrody = [0.1, 1.0, 10.0]
    wyniki_skale = {}
    wyniki_skale_kroki = {} 
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        przyszle_skale = {executor.submit(train_ddpg, testowy_seed, domyslny_batch, liczba_epizodow, False, 'gauss', skala): skala for skala in skale_nagrody}
        
        for future in concurrent.futures.as_completed(przyszle_skale):
            skala = przyszle_skale[future]
            res = future.result()
            nagrody = res[0]
            straty = res[1] 
            kroki = res[2]  
            
            # Zapis tymczasowy
            np.savez(f'temp_4_skala_{skala}.npz', nagrody=nagrody, straty=straty, kroki=kroki) 
            
            wyniki_skale[f"Skala {skala}"] = nagrody
            wyniki_skale_kroki[f"Skala {skala}"] = kroki
            print(f"Ukończono skalę {skala}")
            
    np.savez('dane_skale.npz', **wyniki_skale)
    np.savez('dane_skale_kroki.npz', **wyniki_skale_kroki) 


    print("5. Zbieranie danych ze Stable-Baselines3")
    sb3_nagrody, sb3_kroki = train_sb3_ddpg(testowy_seed, episodes=liczba_epizodow)
    
    # Zapis tymczasowy dla SB3
    np.savez('temp_5_sb3.npz', nagrody=sb3_nagrody, kroki=sb3_kroki) 
    
    wyniki_sb3 = {
        "Własna implementacja DDPG": wyniki_seedy_nagrody[f"Seed {testowy_seed}"],
        "SB3 DDPG": sb3_nagrody
    }
    wyniki_sb3_kroki = {
        "Własna implementacja DDPG": wyniki_seedy_kroki[f"Seed {testowy_seed}"],
        "SB3 DDPG": sb3_kroki
    }
    np.savez('dane_sb3.npz', **wyniki_sb3)
    np.savez('dane_sb3_kroki.npz', **wyniki_sb3_kroki) 
    print("Ukończono SB3")


    print("6. Zbieranie danych dla różnych szumów bez rozgrzewki")
    wyniki_szumy_bez_rozgrzewki = {}
    wyniki_szumy_bez_rozgrzewki_kroki = {} 
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        przyszle_bez_rozgrzewki = {executor.submit(train_ddpg, testowy_seed, domyslny_batch, liczba_epizodow, False, szum, 0.1, 0): szum for szum in rodzaje_szumu}
        
        for future in concurrent.futures.as_completed(przyszle_bez_rozgrzewki):
            szum = przyszle_bez_rozgrzewki[future]
            res = future.result()
            nagrody = res[0]
            straty = res[1] 
            kroki = res[2]  
            
            # Zapis tymczasowy
            np.savez(f'temp_6_bezrozgrzewki_szum_{szum}.npz', nagrody=nagrody, straty=straty, kroki=kroki) 
            
            wyniki_szumy_bez_rozgrzewki[f"Szum {szum}"] = nagrody
            wyniki_szumy_bez_rozgrzewki_kroki[f"Szum {szum}"] = kroki
            print(f"Ukończono szum {szum} bez rozgrzewki")
            
    np.savez('dane_szumy_bez_rozgrzewki.npz', **wyniki_szumy_bez_rozgrzewki)
    np.savez('dane_szumy_bez_rozgrzewki_kroki.npz', **wyniki_szumy_bez_rozgrzewki_kroki) 

    print("Zakończono")