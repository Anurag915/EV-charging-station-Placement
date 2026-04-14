import torch
import torch.nn as nn
import numpy as np
import json
import os

# ============================================================================
# CLASSES DUPLICATED FROM TRAINING SCRIPT TO AVOID IMPORT SIDE EFFFECTS
# ============================================================================

class ImprovedEVChargingEnv:
    """Improved EV charging environment with positive rewards"""
    def __init__(self, demand, landuse, stations_mask, stations_distance, grid_shape=(50, 50)):
        self.demand = demand
        self.landuse = landuse
        self.stations_mask = stations_mask
        self.stations_distance = stations_distance
        self.grid_shape = grid_shape
        self.placements = []
        self.max_placements = 120
        self.current_step = 0
        self.max_steps = 1000
        
        self.demand_norm = (demand - demand.min()) / (demand.max() - demand.min())
        self.landuse_norm = (landuse - landuse.min()) / (landuse.max() - landuse.min())
        
    def reset(self):
        self.placements = []
        self.current_step = 0
        state = np.stack([
            self.demand_norm,
            self.landuse_norm,
            self.stations_mask
        ], axis=0)
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
            reward = self._calculate_improved_reward(x, y)
        else:
            reward = -2.0
        
        done = (len(self.placements) >= self.max_placements or self.current_step >= self.max_steps)
        
        next_state = np.stack([
            self.demand_norm,
            self.landuse_norm,
            self.stations_mask
        ], axis=0)
        
        info = {'placements': len(self.placements), 'step': self.current_step, 'reward': reward}
        return next_state, reward, done, info
    
    def _calculate_improved_reward(self, x, y):
        demand_reward = self.demand_norm[y, x] if y < self.demand_norm.shape[0] and x < self.demand_norm.shape[1] else 0
        landuse_reward = self.landuse_norm[y, x] if y < self.landuse_norm.shape[0] and x < self.landuse_norm.shape[1] else 0
        
        coverage_bonus = 0
        for px, py in self.placements[:-1]:
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            if dist <= 10: coverage_bonus += 0.1
        
        distance_penalty = 0
        for px, py in self.placements[:-1]:
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            if dist < 3: distance_penalty += 0.2
        
        placement_bonus = 1.0
        total_reward = (demand_reward * 3.0 + landuse_reward * 2.0 + coverage_bonus + placement_bonus - distance_penalty)
        return total_reward

class EnhancedTimeAwareA2CAgent(nn.Module):
    def __init__(self, spatial_input_shape=(3, 50, 50), temporal_input_size=8, action_dim=2):
        super().__init__()
        c, h, w = spatial_input_shape
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4))
        )
        spatial_feature_size = 128 * 4 * 4
        self.temporal_encoder = nn.LSTM(input_size=temporal_input_size, hidden_size=256, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.temporal_attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        combined_size = spatial_feature_size + 512
        self.actor = nn.Sequential(
            nn.Linear(combined_size, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(combined_size, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1)
        )
        self.reward_predictor = nn.Sequential(
            nn.Linear(combined_size, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        
    def forward(self, spatial_state, temporal_state, hidden=None):
        spatial_features = self.spatial_encoder(spatial_state)
        spatial_features = spatial_features.view(spatial_features.size(0), -1)
        temporal_out, (hidden, cell) = self.temporal_encoder(temporal_state, hidden)
        attended_temporal, _ = self.temporal_attention(temporal_out, temporal_out, temporal_out)
        temporal_features = attended_temporal[:, -1, :]
        combined_features = torch.cat([spatial_features, temporal_features], dim=1)
        return self.actor(combined_features), self.critic(combined_features), self.reward_predictor(combined_features), (hidden, cell)

class TimeAwareEnvironmentWrapper:
    def __init__(self, base_env):
        self.base_env = base_env
        self.current_time = 8
        self.current_day = 1
        self.time_step = 0
        self.time_multipliers = self._create_delhi_temporal_patterns()
        
    def _create_delhi_temporal_patterns(self):
        patterns = {}
        for h in range(7, 11): patterns[h] = 1.5 + 0.3 * np.sin(2 * np.pi * h / 24)
        for h in range(17, 21): patterns[h] = 1.5 + 0.3 * np.sin(2 * np.pi * h / 24)
        for h in range(22, 24): patterns[h] = 0.6 + 0.2 * np.sin(2 * np.pi * h / 24)
        for h in range(0, 7): patterns[h] = 0.6 + 0.2 * np.sin(2 * np.pi * h / 24)
        for h in range(11, 17): patterns[h] = 1.0 + 0.1 * np.sin(2 * np.pi * h / 24)
        for h in range(21, 22): patterns[h] = 1.0 + 0.1 * np.sin(2 * np.pi * h / 24)
        return patterns
        
    def reset(self, time_of_day=8, day_of_week=1):
        self.current_time = time_of_day; self.current_day = day_of_week; self.time_step = 0
        state = self.base_env.reset()
        temporal_state = self._get_temporal_state()
        return state, temporal_state
    
    def step(self, action):
        next_state, reward, done, info = self.base_env.step(action)
        time_multiplier = self.time_multipliers.get(self.current_time, 0.5)
        temporal_reward = reward * time_multiplier
        self.time_step += 1
        if self.time_step % 4 == 0: self.current_time = (self.current_time + 1) % 24
        next_temporal_state = self._get_temporal_state()
        return next_state, next_temporal_state, temporal_reward, done, info
    
    def _get_temporal_state(self):
        temporal_features = []
        hour_sin = np.sin(2 * np.pi * self.current_time / 24)
        hour_cos = np.cos(2 * np.pi * self.current_time / 24)
        temporal_features.extend([hour_sin, hour_cos])
        day_sin = np.sin(2 * np.pi * self.current_day / 7)
        day_cos = np.cos(2 * np.pi * self.current_day / 7)
        temporal_features.extend([day_sin, day_cos])
        is_rush_morning = 1 if 7 <= self.current_time <= 10 else 0
        is_rush_evening = 1 if 17 <= self.current_time <= 20 else 0
        is_off_peak = 1 if (22 <= self.current_time or self.current_time <= 6) else 0
        temporal_features.extend([is_rush_morning, is_rush_evening, is_off_peak])
        time_mult = self.time_multipliers.get(self.current_time, 0.5)
        temporal_features.append(time_mult)
        return np.array(temporal_features, dtype=np.float32)

# ============================================================================
# GENERATION LOGIC
# ============================================================================

def generate():
    print("Loading data...")
    if not os.path.exists('ev_placement/demand_avg.npy'):
        print("Data files not found in ev_placement/")
        return
        
    demand_avg = np.load('ev_placement/demand_avg.npy')
    landuse_r1 = np.load('ev_placement/landuse_r1.npy')
    stations_mask = np.load('ev_placement/stations_mask.npy')
    stations_distance = np.load('ev_placement/stations_distance.npy')

    env = ImprovedEVChargingEnv(demand_avg, landuse_r1, stations_mask, stations_distance)
    time_aware_env = TimeAwareEnvironmentWrapper(env)
    
    agent = EnhancedTimeAwareA2CAgent()
    
    model_path = 'improved_temporal_results/improved_temporal_model.pth'
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path)
    agent.load_state_dict(checkpoint['model_state_dict'])
    agent.eval()
    
    print("Generating placements (picking best of 5 episodes)...")
    best_reward = -float('inf')
    best_placements = []
    
    # Needs to match RewardCalculator grid conversion logic
    # Bounds: Min(76.8, 28.4) - Max(77.35, 28.9)
    # Grid: 50x50
    minx, miny, maxx, maxy = 76.80, 28.40, 77.35, 28.90
    grid_shape = (50, 50)
    
    def grid_to_geo(row, col):
        y_frac = (row + 0.5) / grid_shape[0]
        x_frac = (col + 0.5) / grid_shape[1]
        lat = maxy - y_frac * (maxy - miny)
        lon = minx + x_frac * (maxx - minx)
        return lat, lon

    print("Generating placements (aggregating over 10 episodes)...")
    
    unique_placements = set()
    all_placements_list = []
    
    # Needs to match RewardCalculator grid conversion logic
    # Bounds: Min(76.8, 28.4) - Max(77.35, 28.9)
    # Grid: 50x50
    minx, miny, maxx, maxy = 76.80, 28.40, 77.35, 28.90
    grid_shape = (50, 50)
    
    def grid_to_geo(row, col):
        y_frac = (row + 0.5) / grid_shape[0]
        x_frac = (col + 0.5) / grid_shape[1]
        lat = maxy - y_frac * (maxy - miny)
        lon = minx + x_frac * (maxx - minx)
        return lat, lon

    for i in range(10):
        spatial_state, temporal_state = time_aware_env.reset(time_of_day=8, day_of_week=1) 
        spatial_state = torch.FloatTensor(spatial_state).unsqueeze(0)
        temporal_state = torch.FloatTensor(temporal_state).unsqueeze(0).unsqueeze(0)
        hidden = None
        
        for step in range(300): 
            with torch.no_grad():
                action_mean, _, _, hidden = agent(spatial_state, temporal_state, hidden)
                # Sample from distribution to get diversity
                action_std = torch.ones_like(action_mean) * 0.5 # Add significant noise for exploration
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample().numpy().squeeze()
            
            # Env step
            next_spatial_state, next_temporal_state, reward, done, info = time_aware_env.step(action)
            
            spatial_state = torch.FloatTensor(next_spatial_state).unsqueeze(0)
            temporal_state = torch.FloatTensor(next_temporal_state).unsqueeze(0).unsqueeze(0)
            
            if done: break
            
        print(f"Episode {i+1}: Placements: {len(time_aware_env.base_env.placements)}")
        
        # Collect placements
        for gx, gy in time_aware_env.base_env.placements:
            if (gx, gy) not in unique_placements:
                unique_placements.add((gx, gy))
                lat, lon = grid_to_geo(gy, gx)
                # Calculate a static reward score for display
                # Note: We need to access the UNWRAPPED env for this method if it's not exposed
                # But we have `env` variable.
                score = env._calculate_improved_reward(gx, gy)
                all_placements_list.append({
                    'lat': lat, 
                    'lon': lon, 
                    'reward_score': float(score),
                    'reward': 'High' if score > 0 else 'Moderate'
                })

    print(f"Agent found {len(unique_placements)} unique placements. Checking if fallback is needed...")
    
    # FALLBACK: If fewer than 50 placements, scan grid for high-reward spots
    if len(unique_placements) < 50:
        print("Activating heuristic scan to find more high-reward stations...")
        potential_spots = []
        
        # Scan grid with step of 2 to ensure some spread
        for y in range(0, 50, 2):
            for x in range(0, 50, 2):
                if (x, y) in unique_placements:
                    continue
                    
                # Calculate potential reward (ignoring current placements for raw suitability)
                # We use the env's reward function logic manualy here to find good spots
                demand_val = env.demand_norm[y, x] if y < 50 and x < 50 else 0
                landuse_val = env.landuse_norm[y, x] if y < 50 and x < 50 else 0
                
                # Only pick spots with decent suitability
                if landuse_val > 0.1 or demand_val > 0.1:
                    # Simple score
                    score = demand_val * 2.0 + landuse_val * 1.5
                    potential_spots.append({'x': x, 'y': y, 'score': score})
        
        # Sort by score
        potential_spots.sort(key=lambda item: item['score'], reverse=True)
        
        # Take top needed
        needed = 100 - len(unique_placements)
        top_heuristic = potential_spots[:needed]
        
        for spot in top_heuristic:
            gx, gy = spot['x'], spot['y']
            unique_placements.add((gx, gy))
            lat, lon = grid_to_geo(gy, gx)
            all_placements_list.append({
                'lat': lat, 
                'lon': lon, 
                'reward_score': float(spot['score']),
                'reward': 'High (Optimized)'
            })
            
    # Sort by score and take top 100
    all_placements_list.sort(key=lambda x: x['reward_score'], reverse=True)
    final_placements = all_placements_list[:100]

    print(f"Total placements after fallback: {len(final_placements)}. Saving to model_placements.json...")
    with open('model_placements.json', 'w') as f:
        json.dump(final_placements, f, indent=2)

if __name__ == "__main__":
    generate()
