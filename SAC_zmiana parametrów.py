import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import gymnasium as gym
import concurrent.futures  
import os

# 1. Wymuszenie procesora i ograniczenie wątków na proces
device = torch.device("cpu")
torch.set_num_threads(1)

# 2. Bufor na prealokowanych tablicach numpy zamiast listy
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


# Aktor (Bez LayerNorm)
LOG_STD_MAX = 2
LOG_STD_MIN = -20

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        
        self.mean_linear = nn.Linear(256, action_dim)
        self.log_std_linear = nn.Linear(256, action_dim)

        self.max_action = max_action
        
        torch.nn.init.uniform_(self.mean_linear.weight, a=-3e-3, b=3e-3)
        torch.nn.init.uniform_(self.mean_linear.bias, a=-3e-3, b=3e-3)
        torch.nn.init.uniform_(self.log_std_linear.weight, a=-3e-3, b=3e-3)
        torch.nn.init.uniform_(self.log_std_linear.bias, a=-3e-3, b=3e-3)

    def forward(self, state):
        a = F.relu(self.layer1(state))
        a = F.relu(self.layer2(a))
        
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


# Krytyk (Złączenie stan + akcja na samym początku, brak LayerNorm)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Architektura Q1
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Architektura Q2
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

        torch.nn.init.uniform_(self.l3.weight, a=-3e-3, b=3e-3)
        torch.nn.init.uniform_(self.l3.bias, a=-3e-3, b=3e-3)
        torch.nn.init.uniform_(self.l6.weight, a=-3e-3, b=3e-3)
        torch.nn.init.uniform_(self.l6.bias, a=-3e-3, b=3e-3)

    def forward(self, state, action):
        # Łączymy stan i akcję OD RAZU
        sa = torch.cat([state, action], dim=1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        
        return q1, q2


# SAC (Zwiększone LR do standardowego 3e-4)
class SAC:
    def __init__(self, state_dim, action_dim, max_action, target_entropy=None):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Zmiana LR z 5e-5 na 3e-4
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.gamma = 0.99
        self.tau = 0.005

        if target_entropy is None:
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = float(target_entropy)

        self.log_alpha = torch.tensor([np.log(0.1)], dtype=torch.float32, requires_grad=True, device=device)
        # Zmiana LR dla alphy również na 3e-4
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


def train_sac(seed, batch_size=256, episodes=10001, record_video=False, reward_scale=1.0, learning_starts=2000, target_entropy=None):
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
                cv2.putText(frame_bgr, f"Epizod: {episode} | Kroki: {kroki_w_epizodzie}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
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


# Właściwa funkcja pomocnicza do omijania problemów z as_completed
def uruchom_eksperyment_seed(seed):
    print(f"Startuje proces dla Seeda: {seed}")
    nagrody, straty, kroki = train_sac(
        seed=seed, 
        batch_size=256, 
        episodes=8001, 
        record_video=True, 
        reward_scale=1.0, 
        learning_starts=2000, 
        target_entropy=None
    )
    print(f"Zakończono Seeda: {seed}")
    return seed, nagrody, straty, kroki


if __name__ == '__main__':
    seedy = [1, 42, 123, 1234, 999]
    
    wyniki_seedy_nagrody, wyniki_seedy_straty, wyniki_seedy_kroki = {}, {}, {}

    print(f"Uruchamianie wszystkich eksperymentów w puli procesów")
    
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
            
           
            wyniki = list(executor.map(uruchom_eksperyment_seed, seedy))

            for seed, nagrody, straty, kroki in wyniki:
                wyniki_seedy_nagrody[f'seed_{seed}'] = nagrody
                wyniki_seedy_straty[f'seed_{seed}'] = straty
                wyniki_seedy_kroki[f'seed_{seed}'] = kroki
                print(f"Zapisano dane do słownika dla Seeda: {seed}")
                
                np.savez(f'temp_seed_{seed}_sac.npz', nagrody=nagrody, straty=straty, kroki=kroki)

    except Exception as exc:
        print(f'Błąd krytyczny podczas działania puli procesów: {exc}')
        import traceback
        traceback.print_exc()

    print("\nZapisywanie zbiorczych plików .npz")
    
    if wyniki_seedy_nagrody: 
        try:
            np.savez('dane_seedy_sac.npz', **wyniki_seedy_nagrody)
            np.savez('dane_seedy_straty_sac.npz', **wyniki_seedy_straty)
            np.savez('dane_seedy_kroki_sac.npz', **wyniki_seedy_kroki) 
            print("Zakończono pomyślnie. Pliki zbiorcze zapisane na dysku.")
        except Exception as e:
            print(f"Wystąpił błąd podczas ostatecznego zapisu do pliku: {e}")
    else:
        print("Brak danych do zapisania")