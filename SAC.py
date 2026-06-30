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

# 2. bufor na prealokowanych tablicach numpy zamiast listy
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
LOG_STD_MAX = 2
LOG_STD_MIN = -20

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.layer1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.layer2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        
        self.mean_linear = nn.Linear(256, action_dim)
        self.log_std_linear = nn.Linear(256, action_dim)

        self.max_action = max_action
        
        # Modyfikacja inicjalizacji z rozkładu jednostajnego [-3e-3, 3e-3]
        torch.nn.init.uniform_(self.mean_linear.weight, a=-3e-3, b=3e-3)
        torch.nn.init.uniform_(self.mean_linear.bias, a=-3e-3, b=3e-3)
        torch.nn.init.uniform_(self.log_std_linear.weight, a=-3e-3, b=3e-3)
        torch.nn.init.uniform_(self.log_std_linear.bias, a=-3e-3, b=3e-3)

    def forward(self, state):
        a = self.layer1(state)
        a = F.relu(self.ln1(a))
        a = self.layer2(a)
        a = F.relu(self.ln2(a))
        
        mean = self.mean_linear(a)
        log_std = self.log_std_linear(a)
        
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        
        x_t = normal.rsample()  
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        mean_action = torch.tanh(mean) * self.max_action
        
        return action, log_prob, mean_action


# Krytyk
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layer1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.layer2 = nn.Linear(256 + action_dim, 256)
        self.layer3 = nn.Linear(256, 1)

        self.layer4 = nn.Linear(state_dim, 256)
        self.ln4 = nn.LayerNorm(256)
        self.layer5 = nn.Linear(256 + action_dim, 256)
        self.layer6 = nn.Linear(256, 1)

        # Modyfikacja inicjalizacji z rozkładu jednostajnego [-3e-3, 3e-3]
        torch.nn.init.uniform_(self.layer3.weight, a=-3e-3, b=3e-3)
        torch.nn.init.uniform_(self.layer3.bias, a=-3e-3, b=3e-3)
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


# SAC
class SAC:
    def __init__(self, state_dim, action_dim, max_action, target_entropy=None):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.gamma = 0.99
        self.tau = 0.005

        if target_entropy is None:
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = float(target_entropy)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        with torch.no_grad():
            action, _, mean_action = self.actor.sample(state)
            
        if evaluate:
            return mean_action.numpy()[0]
        else:
            return action.numpy()[0]

    def train(self, replay_buffer, batch_size=256):
        state, action, reward, next_state, terminated = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            
            alpha = self.log_alpha.exp()
            target_Q = reward + (1 - terminated) * self.gamma * (target_Q - alpha * next_log_prob)

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        pi, log_prob, _ = self.actor.sample(state)
        
        q1_pi, q2_pi = self.critic(state, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        
        actor_loss = ((self.log_alpha.exp().detach() * log_prob) - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item(), current_Q1.mean().item()


def train_sac(seed, batch_size=256, episodes=10001, record_video=False, reward_scale=0.1, learning_starts=2000, target_entropy=None):
    random.seed(seed)
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
        out = cv2.VideoWriter(f'ewolucja_sac_seed_{seed}.mp4', fourcc, 30.0, frame_size)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = SAC(state_dim, action_dim, max_action, target_entropy=target_entropy)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    historia_nagrod = []
    historia_straty = []
    historia_krokow = [] 

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        straty_w_epizodzie = []
        terminated = False
        truncated = False
        kroki_w_epizodzie = 0 

        while not (terminated or truncated):
            kroki_w_epizodzie += 1 
            
            if len(replay_buffer) < learning_starts:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, evaluate=False)

            next_state, reward, terminated, truncated, _ = env.step(action)
            scaled_reward = reward * reward_scale

            if record_video and (episode in epizody_do_nagrania):
                frame = env.render()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.putText(frame_bgr, f"Epizod: {episode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
                out.write(frame_bgr)

            replay_buffer.add(state, action, scaled_reward, next_state, terminated)

            if len(replay_buffer) > batch_size:
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

        if episode > 0 and episode % 1000 == 0:
            print(f"[Proces: Seed {seed} | Batch {batch_size}] Zakończono epizod {episode}/{episodes}")

    if out is not None:
        out.release()
    env.close()

    return historia_nagrod, historia_straty, historia_krokow


# Klasa Callback wspólna dla wariantów SB3
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


def train_sb3_sac(seed, episodes=10001):
    from stable_baselines3 import SAC as SB3_SAC
    from stable_baselines3.common.monitor import Monitor

    env = gym.make("BipedalWalker-v3")
    env = Monitor(env)
    env.reset(seed=seed)

    model = SB3_SAC("MlpPolicy", env, seed=seed, learning_rate=3e-4, gamma=0.99, tau=0.005, batch_size=256, learning_starts=2000, device='cpu')
    logger = RewardLoggerCallback(max_episodes=episodes)
    total_steps = 1600 * episodes
    model.learn(total_timesteps=total_steps, callback=logger)

    env.close()
    return logger.episode_rewards[:episodes], logger.episode_lengths[:episodes]


def train_sb3_sac_matched(seed, episodes=10001):
    from stable_baselines3 import SAC as SB3_SAC
    from stable_baselines3.common.monitor import Monitor

    env = gym.make("BipedalWalker-v3")
    env = Monitor(env)
    env.reset(seed=seed)

    # Ujednolicenie architektury sieci (brak LayerNorm w domyślnym SB3, ale pasujące wymiary 256x256)
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))

    model = SB3_SAC(
        "MlpPolicy", 
        env, 
        seed=seed, 
        learning_rate=3e-4, 
        buffer_size=1000000,
        learning_starts=2000, # Zgodnie z własną implementacją
        batch_size=256,       # Zgodnie z własną implementacją
        tau=0.005, 
        gamma=0.99, 
        ent_coef='auto',
        policy_kwargs=policy_kwargs,
        device='cpu'
    )
    
    logger = RewardLoggerCallback(max_episodes=episodes)
    max_possible_steps = 1600 * episodes
    model.learn(total_timesteps=max_possible_steps, callback=logger)

    env.close()
    return logger.episode_rewards[:episodes], logger.episode_lengths[:episodes]


def train_sb3_sac_default(seed, episodes=10001):
    from stable_baselines3 import SAC as SB3_SAC
    from stable_baselines3.common.monitor import Monitor

    env = gym.make("BipedalWalker-v3")
    env = Monitor(env)
    env.reset(seed=seed)

    # Całkowicie domyślne parametry SB3 dla SAC (poza procesorem)
    model = SB3_SAC("MlpPolicy", env, seed=seed, device='cpu')
    
    logger = RewardLoggerCallback(max_episodes=episodes)
    max_possible_steps = 1600 * episodes
    model.learn(total_timesteps=max_possible_steps, callback=logger)

    env.close()
    return logger.episode_rewards[:episodes], logger.episode_lengths[:episodes]


if __name__ == '__main__':
    seedy = [1, 42, 123, 1234, 999]
    liczba_epizodow = 8001
    domyslny_batch = 256
    testowy_seed = seedy[0]
    testowane_batche = [32, 64, 128, 256, 512]
    skale_nagrody = [0.1, 1.0, 10.0]
    docelowe_entropie = [-2.0, -4.0, -8.0]

    # Przygotowanie słowników na wyniki
    wyniki_seedy_nagrody, wyniki_seedy_straty, wyniki_seedy_kroki = {}, {}, {}
    wyniki_batche_nagrody, wyniki_batche_straty, wyniki_batche_kroki = {}, {}, {}
    wyniki_skale_nagrody, wyniki_skale_straty, wyniki_skale_kroki = {}, {}, {}
    wyniki_entropia_nagrody, wyniki_entropia_straty, wyniki_entropia_kroki = {}, {}, {}
    sb3_nagrody, sb3_kroki = None, None
    
    # Nowe słowniki dla zrównoleglonego SB3
    wyniki_sb3_matched_nagrody, wyniki_sb3_matched_kroki = {}, {}
    wyniki_sb3_default_nagrody, wyniki_sb3_default_kroki = {}, {}

    print(f"Uruchamianie wszystkich eksperymentów w puli procesów")
    
    # max_workers=32
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
        przyszle_zadania = {}

        # 1. Zlecenie eksperymentów z Seedami (5 zadań)
        for seed in seedy:
            future = executor.submit(train_sac, seed, domyslny_batch, liczba_epizodow, seed == testowy_seed, 0.1, 2000, None)
            przyszle_zadania[future] = ('seed', seed)

        # 2. Zlecenie eksperymentów z Batchami (5 zadań)
        for bs in testowane_batche:
            future = executor.submit(train_sac, testowy_seed, bs, liczba_epizodow, False, 0.1, 2000, None)
            przyszle_zadania[future] = ('batch', bs)

        # 3. Zlecenie eksperymentów ze Skalą (3 zadania)
        for skala in skale_nagrody:
            future = executor.submit(train_sac, testowy_seed, domyslny_batch, liczba_epizodow, False, skala, 2000, None)
            przyszle_zadania[future] = ('skala', skala)

        # 4. Zlecenie eksperymentów z Entropią (3 zadania)
        for ent in docelowe_entropie:
            future = executor.submit(train_sac, testowy_seed, domyslny_batch, liczba_epizodow, False, 0.1, 2000, ent)
            przyszle_zadania[future] = ('entropia', ent)

        # 5. Zlecenie eksperymentu SB3 - STARA WERSJA DLA KOMPATYBILNOŚCI (1 zadanie)
        future_sb3 = executor.submit(train_sb3_sac, testowy_seed, liczba_epizodow)
        przyszle_zadania[future_sb3] = ('sb3_legacy', 'sb3_legacy')
        
        # 6. Zlecenie eksperymentów SB3 MATCHED na wszystkich seedach (5 zadań)
        for seed in seedy:
            future = executor.submit(train_sb3_sac_matched, seed, liczba_epizodow)
            przyszle_zadania[future] = ('sb3_matched', seed)
            
        # 7. Zlecenie eksperymentów SB3 DEFAULT na wszystkich seedach (5 zadań)
        for seed in seedy:
            future = executor.submit(train_sb3_sac_default, seed, liczba_epizodow)
            przyszle_zadania[future] = ('sb3_default', seed)

        # Wyłapywanie wyników w miarę ich kończenia (niezależnie od kolejności)
        for future in concurrent.futures.as_completed(przyszle_zadania):
            typ, wartosc = przyszle_zadania[future]

            if typ == 'sb3_legacy':
                sb3_nagrody, sb3_kroki = future.result()
                np.savez('temp_4_sb3_sac.npz', nagrody=sb3_nagrody, kroki=sb3_kroki)
                print("Ukończono eksperyment: SB3 Legacy")
                
            elif typ == 'sb3_matched':
                nagrody, kroki = future.result()
                np.savez(f'temp_sb3_matched_seed_{wartosc}_sac.npz', nagrody=nagrody, kroki=kroki)
                wyniki_sb3_matched_nagrody[f"Seed {wartosc}"] = nagrody
                wyniki_sb3_matched_kroki[f"Seed {wartosc}"] = kroki
                print(f"Ukończono eksperyment: SB3 Matched (Seed {wartosc})")
                
            elif typ == 'sb3_default':
                nagrody, kroki = future.result()
                np.savez(f'temp_sb3_default_seed_{wartosc}_sac.npz', nagrody=nagrody, kroki=kroki)
                wyniki_sb3_default_nagrody[f"Seed {wartosc}"] = nagrody
                wyniki_sb3_default_kroki[f"Seed {wartosc}"] = kroki
                print(f"Ukończono eksperyment: SB3 Default (Seed {wartosc})")
            
            else:
                nagrody, straty, kroki = future.result()

                if typ == 'seed':
                    np.savez(f'temp_1_seed_{wartosc}_sac.npz', nagrody=nagrody, straty=straty, kroki=kroki)
                    wyniki_seedy_nagrody[f"Seed {wartosc}"] = nagrody
                    wyniki_seedy_straty[f"Seed {wartosc}"] = straty
                    wyniki_seedy_kroki[f"Seed {wartosc}"] = kroki
                    print(f"Ukończono eksperyment: Seed {wartosc}")

                elif typ == 'batch':
                    np.savez(f'temp_2_batch_{wartosc}_sac.npz', nagrody=nagrody, straty=straty, kroki=kroki)
                    wyniki_batche_nagrody[f"Batch {wartosc}"] = nagrody
                    wyniki_batche_straty[f"Batch {wartosc}"] = straty
                    wyniki_batche_kroki[f"Batch {wartosc}"] = kroki
                    print(f"Ukończono eksperyment: Batch {wartosc}")

                elif typ == 'skala':
                    np.savez(f'temp_3_skala_{wartosc}_sac.npz', nagrody=nagrody, straty=straty, kroki=kroki)
                    wyniki_skale_nagrody[f"Skala {wartosc}"] = nagrody
                    wyniki_skale_straty[f"Skala {wartosc}"] = straty
                    wyniki_skale_kroki[f"Skala {wartosc}"] = kroki
                    print(f"Ukończono eksperyment: Skala {wartosc}")

                elif typ == 'entropia':
                    np.savez(f'temp_5_entropia_{wartosc}_sac.npz', nagrody=nagrody, straty=straty, kroki=kroki)
                    wyniki_entropia_nagrody[f"Entropia {wartosc}"] = nagrody
                    wyniki_entropia_straty[f"Entropia {wartosc}"] = straty
                    wyniki_entropia_kroki[f"Entropia {wartosc}"] = kroki
                    print(f"Ukończono eksperyment: Entropia {wartosc}")

    # ZAPIS ZBIORCZY PO ZAKOŃCZENIU WSZYSTKICH PROCESÓW 
    print("\nZapisywanie zbiorczych plików")
    
    np.savez('dane_seedy_sac.npz', **wyniki_seedy_nagrody)
    np.savez('dane_seedy_straty_sac.npz', **wyniki_seedy_straty)
    np.savez('dane_seedy_kroki_sac.npz', **wyniki_seedy_kroki) 

    np.savez('dane_batche_sac.npz', **wyniki_batche_nagrody)
    np.savez('dane_batche_straty_sac.npz', **wyniki_batche_straty)
    np.savez('dane_batche_kroki_sac.npz', **wyniki_batche_kroki) 

    np.savez('dane_skale_sac.npz', **wyniki_skale_nagrody)
    np.savez('dane_skale_straty_sac.npz', **wyniki_skale_straty)
    np.savez('dane_skale_kroki_sac.npz', **wyniki_skale_kroki) 

    np.savez('dane_entropia_sac.npz', **wyniki_entropia_nagrody)
    np.savez('dane_entropia_straty_sac.npz', **wyniki_entropia_straty)
    np.savez('dane_entropia_kroki_sac.npz', **wyniki_entropia_kroki)

    # Zapis Legacy SB3 (zgodnie z poprzednim zachowaniem)
    wyniki_sb3 = {
        "Własna implementacja SAC": wyniki_seedy_nagrody[f"Seed {testowy_seed}"],
        "SB3 SAC": sb3_nagrody
    }
    wyniki_sb3_kroki = {
        "Własna implementacja SAC": wyniki_seedy_kroki[f"Seed {testowy_seed}"],
        "SB3 SAC": sb3_kroki
    }
    np.savez('dane_sb3_sac.npz', **wyniki_sb3)
    np.savez('dane_sb3_kroki_sac.npz', **wyniki_sb3_kroki) 
    
    # Zapis nowych, zrównoleglonych wariantów SB3
    np.savez('dane_sb3_sac_matched_seedy.npz', **wyniki_sb3_matched_nagrody)
    np.savez('dane_sb3_sac_matched_kroki_seedy.npz', **wyniki_sb3_matched_kroki) 
    
    np.savez('dane_sb3_sac_default_seedy.npz', **wyniki_sb3_default_nagrody)
    np.savez('dane_sb3_sac_default_kroki_seedy.npz', **wyniki_sb3_default_kroki) 

    print("Zakończono pomyślnie.")