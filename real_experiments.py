import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy

print("Starting REAL Experiments for EV Placement...")

# 1. Load Data
demand_avg = np.load('ev_placement/demand_avg.npy')
landuse_r1 = np.load('ev_placement/landuse_r1.npy')
stations_mask = np.load('ev_placement/stations_mask.npy')
stations_distance = np.load('ev_placement/stations_distance.npy')

# 2. Configurable Environment
class ConfigurableEVEnv:
    def __init__(self, demand, landuse, stations_mask, stations_distance, scenario_weights, grid_shape=(50, 50)):
        self.demand = demand
        self.landuse = landuse
        self.stations_mask = stations_mask.copy()
        self.initial_mask = stations_mask.copy()
        self.stations_distance = stations_distance
        self.grid_shape = grid_shape
        self.placements = []
        self.max_placements = 120
        self.current_step = 0
        self.max_steps = 1000
        self.weights = scenario_weights
        
        self.demand_norm = (demand - demand.min()) / (demand.max() - demand.min() + 1e-8)
        self.landuse_norm = (landuse - landuse.min()) / (landuse.max() - landuse.min() + 1e-8)
        
    def reset(self):
        self.placements = []
        self.current_step = 0
        self.stations_mask = self.initial_mask.copy()
        state = np.stack([self.demand_norm, self.landuse_norm, self.stations_mask], axis=0)
        return state
    
    def step(self, action):
        self.current_step += 1
        
        if isinstance(action, np.ndarray) and len(action) == 2:
            x, y = action
            x = int(np.clip(x, 0, self.grid_shape[1] - 1))
            y = int(np.clip(y, 0, self.grid_shape[0] - 1))
        else:
            x, y = 0, 0
            
        is_duplicate = (self.stations_mask[y, x] == 1)
        
        if not is_duplicate and len(self.placements) < self.max_placements:
            self.placements.append((x, y))
            self.stations_mask[y, x] = 1
            reward, comps = self._calc_reward(x, y)
        else:
            reward = -2.0
            comps = [0,0,0,0,0]
        
        done = (len(self.placements) >= self.max_placements or self.current_step >= self.max_steps)
        next_state = np.stack([self.demand_norm, self.landuse_norm, self.stations_mask], axis=0)
        
        info = {'comps': comps}
        return next_state, reward, done, info
    
    def _calc_reward(self, x, y):
        # R1: Demand
        r1 = self.demand_norm[y, x] if y < self.demand_norm.shape[0] else 0
        # R2: Landuse
        r2 = self.landuse_norm[y, x] if y < self.landuse_norm.shape[0] else 0
        # R3: Coverage
        r3 = 0
        for px, py in self.placements[:-1]:
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            if dist <= 10: r3 += 0.1
        # R4: Distance (Penalty -> Positive means less penalty here for plotting)
        r4 = 0
        for px, py in self.placements[:-1]:
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            if dist < 3: r4 += 0.2
        # R5: Placement
        r5 = 1.0
        
        w = self.weights
        # Formula: w1*R1 + w2*R2 + w3*R3 - w4*R4 + w5*R5
        total = (r1 * w[0]) + (r2 * w[1]) + (r3 * w[2]) - (r4 * w[3]) + (r5 * w[4])
        return total, [r1, r2, r3, r4, r5]

# 3. Time Wrapper
class TimeWrapper:
    def __init__(self, base_env):
        self.base_env = base_env
        self.current_time = 8
        self.current_day = 1
        self.time_step = 0
        
    def reset(self, t=8, d=1):
        self.current_time = t
        self.current_day = d
        self.time_step = 0
        return self.base_env.reset(), self._get_time()
        
    def step(self, action):
        ns, r, d, info = self.base_env.step(action)
        mult = 1.5 if (7<=self.current_time<=10 or 17<=self.current_time<=20) else (0.6 if (self.current_time>=22 or self.current_time<=6) else 1.0)
        sr = r * mult
        self.time_step += 1
        if self.time_step % 4 == 0: self.current_time = (self.current_time + 1) % 24
        return ns, self._get_time(), sr, d, info
        
    def _get_time(self):
        f = [np.sin(2*np.pi*self.current_time/24), np.cos(2*np.pi*self.current_time/24),
             np.sin(2*np.pi*self.current_day/7), np.cos(2*np.pi*self.current_day/7),
             1 if 7<=self.current_time<=10 else 0, 1 if 17<=self.current_time<=20 else 0,
             1 if self.current_time>=22 or self.current_time<=6 else 0, 1.0]
        return np.array(f, dtype=np.float32)

# 4. Agent
class RealA2C(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4))
        )
        self.temporal = nn.LSTM(8, 64, 1, batch_first=True)
        self.actor = nn.Sequential(nn.Linear(64*4*4 + 64, 128), nn.ReLU(), nn.Linear(128, 2))
        self.critic = nn.Sequential(nn.Linear(64*4*4 + 64, 128), nn.ReLU(), nn.Linear(128, 1))
        
    def forward(self, s, t, h=None):
        sf = self.spatial(s).view(s.size(0), -1)
        tf, (h, c) = self.temporal(t, h)
        tf = tf[:, -1, :]
        feat = torch.cat([sf, tf], dim=1)
        return self.actor(feat), self.critic(feat), (h, c)

# 5. Training Loop Tracker
def train_episode(agent, env, optimizer, gamma=0.99):
    ss, ts = env.reset(np.random.randint(0,24), np.random.randint(0,7))
    ss, ts = torch.FloatTensor(ss).unsqueeze(0), torch.FloatTensor(ts).unsqueeze(0).unsqueeze(0)
    
    h = None
    lps, vals, rews, comps = [], [], [], []
    
    # max steps 50 to match original training bounds
    for _ in range(50):
        mean, val, h = agent(ss, ts, h)
        std = torch.ones_like(mean) * 0.1
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        
        nss, nts, r, d, info = env.step(action.detach().numpy().squeeze())
        
        lps.append(dist.log_prob(action).sum())
        vals.append(val.squeeze())
        rews.append(r)
        comps.append(info['comps'])
        
        ss, ts = torch.FloatTensor(nss).unsqueeze(0), torch.FloatTensor(nts).unsqueeze(0).unsqueeze(0)
        if d: break
        
    if len(lps) < 2: return 0, 0, 0, [0]*5
    
    R = 0
    returns = []
    for r in reversed(rews):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.FloatTensor(returns)
    vals = torch.stack(vals)
    advs = returns - vals
    
    ploss = -(torch.stack(lps) * advs.detach()).mean()
    vloss = advs.pow(2).mean()
    loss = ploss + 0.5 * vloss
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
    optimizer.step()
    
    avg_comps = np.mean(comps, axis=0) if len(comps) > 0 else [0]*5
    return sum(rews), loss.item(), ploss.item(), vloss.item(), avg_comps

# 6. Scenarios (Paper equivalent ablations)
scenarios = {
    '1': [3.0, 2.0, 1.0, 1.0, 1.0], # R_all
    '2': [0.0, 2.0, 1.0, 1.0, 1.0], # R_all except Demand
    '3': [3.0, 0.0, 1.0, 1.0, 1.0], # R_all except Landuse
    '4': [3.0, 2.0, 0.0, 1.0, 1.0], # R_all except Coverage
    '5': [3.0, 2.0, 1.0, 0.0, 1.0], # R_all except Distance
    '6': [1.0, 1.0, 1.0, 1.0, 0.0]  # R_all except Placement
}

results = {}
episodes = 250 # Adjusted to 250 for realistic completion time within bounds

for sc_name, weights in scenarios.items():
    print(f"Training Scenario {sc_name} with weights {weights}...")
    b_env = ConfigurableEVEnv(demand_avg, landuse_r1, stations_mask, stations_distance, weights)
    env = TimeWrapper(b_env)
    agent = RealA2C()
    opt = torch.optim.Adam(agent.parameters(), lr=1e-3) # faster lr for quick learning
    
    sc_rews, sc_ploss, sc_vloss = [], [], []
    sc_comps = []
    
    for ep in range(episodes):
        R, L, PL, VL, C = train_episode(agent, env, opt)
        sc_rews.append(R)
        sc_ploss.append(PL)
        sc_vloss.append(VL)
        sc_comps.append(C)
        if ep % 50 == 0:
            print(f"  Ep {ep}: R={np.mean(sc_rews[-20:]):.2f}, L={L:.2f}")

    results[sc_name] = {
        'rewards': sc_rews,
        'actor_loss': sc_ploss,
        'critic_loss': sc_vloss,
        'components': np.array(sc_comps)
    }

print("Training completed. Saving real results...")

# 7. Generate Real Results
output_dir = 'ExpectedResult_Real'
os.makedirs(output_dir, exist_ok=True)

# Generate Real Table 6
# Normalize components to 'performance' % based on maximum achieved across all scenarios.
max_c = np.max([np.max(results[s]['components'], axis=0) for s in scenarios], axis=0) + 1e-5
table_rows = []
for s in scenarios:
    last_10 = np.mean(results[s]['components'][-50:], axis=0)
    perf = (last_10 / max_c) * 100
    perf = np.clip(perf, 0, 100)
    avg_total = np.mean(results[s]['rewards'][-50:])
    baseline = np.mean(results['1']['rewards'][-50:])
    overall = min(100.0, max(0.0, (avg_total/baseline)*100)) if baseline > 0 else 0
    table_rows.append({
        'Scenario': s,
        '$R_1$': f"{perf[0]:.2f} %",
        '$R_2$': f"{perf[1]:.2f} %",
        '$R_3$': f"{perf[2]:.2f} %",
        '$R_4$': f"{perf[3]:.2f} %",
        '$R_5$': f"{perf[4]:.2f} %",
        '$R_{all}$': f"{overall:.2f} %"
    })

df = pd.DataFrame(table_rows)
df.to_csv(os.path.join(output_dir, 'Table_6_Real.csv'), index=False)
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
table.scale(1, 2)
table.auto_set_font_size(False)
table.set_fontsize(12)
plt.title("Table 6 (REAL MODEL RESULTS)\nLearning performance for six training scenarios.", loc='left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Table_6_Real.png'), dpi=300)
plt.close()

# Generate Real Fig 5
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#B0C4DE', '#D2B48C', '#F0E68C', '#4CAF50', '#212121', '#808080']
for (s, d), c in zip(results.items(), colors):
    smoothed = pd.Series(d['rewards']).rolling(10, min_periods=1).mean()
    ax.plot(smoothed, label=f'Scenario {s}', color=c, lw=2, alpha=0.8)
ax.set_xlabel("Number of training episodes")
ax.set_ylabel("Training rewards")
ax.legend(loc='center right', bbox_to_anchor=(1.15, 0.5))
plt.title("Fig. 5. REAL Training results of six scenarios.")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Fig_5_Real.png'), dpi=300)
plt.close()

# Generate Real Fig 6 (Losses - from Scenario 1 as representative)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
aloss = pd.Series(results['1']['actor_loss']).rolling(5, min_periods=1).mean()
closs = pd.Series(results['1']['critic_loss']).rolling(5, min_periods=1).mean()
ax1.plot(aloss, color='green')
ax1.set_xlabel("Number of training episodes")
ax1.set_ylabel("Actor loss")
ax1.set_title("(a) REAL Actor Loss")

ax2.plot(closs, color='green')
ax2.set_xlabel("Number of training episodes")
ax2.set_ylabel("Critic loss")
ax2.set_title("(b) REAL Critic Loss")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Fig_6_Real.png'), dpi=300)
plt.close()

print(f"All real results saved to {output_dir}/")
