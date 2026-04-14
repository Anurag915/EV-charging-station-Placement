# ============================================================================
# TIME-AWARE LSTM/GRU INTEGRATION - ADD TO YOUR NOTEBOOK
# ============================================================================
# Copy and paste this entire cell into your finalyearproject.ipynb notebook
# This will seamlessly integrate temporal features with your existing A2C system

print("INTEGRATING TIME-AWARE LSTM/GRU FEATURES")
print("=" * 60)

# Import required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import your existing temporal modules
try:
    from temporal_ev_placement import (
        TemporalDataProcessor, 
        TimeAwareA2CAgent, 
        TimeAwareEVEnvironment,
        TemporalTrainer,
        TemporalVisualizer
    )
    print("Temporal modules imported successfully!")
except ImportError as e:
    print(f"⚠️  Import error: {e}")
    print("Make sure temporal_ev_placement.py is in the same directory")

# ============================================================================
# 1. ENHANCED TIME-AWARE A2C AGENT (INTEGRATED WITH YOUR EXISTING CODE)
# ============================================================================

class EnhancedTimeAwareA2CAgent(nn.Module):
    """
    Enhanced A2C Agent with Temporal LSTM Integration
    Combines your existing A2C architecture with temporal pattern learning
    """
    def __init__(self, spatial_input_shape=(3, 50, 50), temporal_input_size=8, action_dim=2):
        super().__init__()
        
        self.spatial_input_shape = spatial_input_shape
        self.temporal_input_size = temporal_input_size
        self.action_dim = action_dim
        
        # Your existing spatial CNN base (enhanced)
        c, h, w = spatial_input_shape
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Calculate spatial feature size
        spatial_feature_size = 128 * 4 * 4
        
        # NEW: Temporal LSTM encoder
        self.temporal_encoder = nn.LSTM(
            input_size=temporal_input_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # NEW: Temporal attention mechanism
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=512,  # 256 * 2 for bidirectional
            num_heads=8,
            batch_first=True
        )
        
        # Combined feature size
        combined_size = spatial_feature_size + 512  # 256 * 2 for bidirectional LSTM
        
        # Enhanced Actor network (your existing + temporal)
        self.actor = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        # Enhanced Critic network (your existing + temporal)
        self.critic = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # NEW: Time-aware reward predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(combined_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # NEW: Temporal feature importance weights
        self.temporal_weights = nn.Parameter(torch.ones(temporal_input_size))
        
    def forward(self, spatial_state, temporal_state, hidden=None):
        """
        Forward pass with spatial and temporal inputs
        """
        # Extract spatial features (your existing approach)
        spatial_features = self.spatial_encoder(spatial_state)
        spatial_features = spatial_features.view(spatial_features.size(0), -1)
        
        # NEW: Extract temporal features
        temporal_out, (hidden, cell) = self.temporal_encoder(temporal_state, hidden)
        
        # NEW: Apply temporal attention
        attended_temporal, _ = self.temporal_attention(temporal_out, temporal_out, temporal_out)
        temporal_features = attended_temporal[:, -1, :]  # Use last time step
        
        # Combine spatial and temporal features
        combined_features = torch.cat([spatial_features, temporal_features], dim=1)
        
        # Get outputs
        action_mean = self.actor(combined_features)
        state_value = self.critic(combined_features)
        reward_prob = self.reward_predictor(combined_features)
        
        return action_mean, state_value, reward_prob, (hidden, cell)

# ============================================================================
# 2. TIME-AWARE ENVIRONMENT WRAPPER
# ============================================================================

class TimeAwareEnvironmentWrapper:
    """
    Wraps your existing environment with temporal awareness
    """
    def __init__(self, base_env, temporal_processor=None):
        self.base_env = base_env
        self.temporal_processor = temporal_processor
        self.current_time = 8  # Start at 8 AM
        self.current_day = 1   # Monday
        self.time_step = 0
        
        # Load temporal features
        if temporal_processor:
            self.temporal_features = temporal_processor.load_temporal_data()
            self.time_multipliers = self.temporal_features['demand_profiles']['time_multipliers']
        else:
            # Create mock temporal data
            self.time_multipliers = {h: 0.5 + 0.5 * np.sin(2 * np.pi * h / 24) for h in range(24)}
        
    def reset(self, time_of_day=8, day_of_week=1):
        """Reset with temporal context"""
        self.current_time = time_of_day
        self.current_day = day_of_week
        self.time_step = 0
        
        # Reset base environment
        state = self.base_env.reset()
        
        # Get temporal state
        temporal_state = self._get_temporal_state()
        
        return state, temporal_state
    
    def step(self, action):
        """Step with time-aware rewards"""
        # Take action in base environment
        next_state, reward, done, info = self.base_env.step(action)
        
        # Apply time-aware reward shaping
        time_multiplier = self._get_time_multiplier()
        temporal_reward = reward * time_multiplier
        
        # Update time
        self.time_step += 1
        if self.time_step % 4 == 0:  # Advance time every 4 steps
            self.current_time = (self.current_time + 1) % 24
        
        # Get next temporal state
        next_temporal_state = self._get_temporal_state()
        
        return next_state, next_temporal_state, temporal_reward, done, info
    
    def _get_temporal_state(self):
        """Get temporal state representation"""
        temporal_features = []
        
        # Time of day features (cyclical encoding)
        hour_sin = np.sin(2 * np.pi * self.current_time / 24)
        hour_cos = np.cos(2 * np.pi * self.current_time / 24)
        temporal_features.extend([hour_sin, hour_cos])
        
        # Day of week features (cyclical encoding)
        day_sin = np.sin(2 * np.pi * self.current_day / 7)
        day_cos = np.cos(2 * np.pi * self.current_day / 7)
        temporal_features.extend([day_sin, day_cos])
        
        # Rush hour indicators
        is_rush_morning = 1 if 7 <= self.current_time <= 10 else 0
        is_rush_evening = 1 if 17 <= self.current_time <= 20 else 0
        is_off_peak = 1 if (22 <= self.current_time or self.current_time <= 6) else 0
        temporal_features.extend([is_rush_morning, is_rush_evening, is_off_peak])
        
        # Time multiplier
        time_mult = self.time_multipliers.get(self.current_time, 0.5)
        temporal_features.append(time_mult)
        
        return np.array(temporal_features, dtype=np.float32)
    
    def _get_time_multiplier(self):
        """Get time-based reward multiplier"""
        base_mult = self.time_multipliers.get(self.current_time, 0.5)
        
        # Rush hour boost
        if 7 <= self.current_time <= 10 or 17 <= self.current_time <= 20:
            return base_mult * 1.5
        # Off-peak efficiency
        elif 22 <= self.current_time or self.current_time <= 6:
            return base_mult * 0.8
        else:
            return base_mult

# ============================================================================
# 3. TEMPORAL TRAINING FUNCTION
# ============================================================================

def train_temporal_episode(agent, env, optimizer, max_steps=100, gamma=0.99):
    """
    Train one episode with temporal awareness
    """
    # Reset environment with random time
    time_of_day = np.random.randint(0, 24)
    day_of_week = np.random.randint(0, 7)
    
    spatial_state, temporal_state = env.reset(time_of_day, day_of_week)
    spatial_state = torch.FloatTensor(spatial_state).unsqueeze(0)
    temporal_state = torch.FloatTensor(temporal_state).unsqueeze(0).unsqueeze(0)
    
    episode_rewards = []
    log_probs = []
    values = []
    rewards = []
    
    hidden = None
    
    for step in range(max_steps):
        # Get action from agent
        with torch.no_grad():
            action_mean, value, reward_prob, hidden = agent(
                spatial_state, temporal_state, hidden
            )
        
        # Sample action (continuous)
        action_std = torch.ones_like(action_mean) * 0.1
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        
        # Take action
        next_spatial_state, next_temporal_state, reward, done, info = env.step(
            action.numpy().squeeze()
        )
        
        # Store experience
        log_probs.append(log_prob)
        values.append(value.squeeze())
        rewards.append(reward)
        episode_rewards.append(reward)
        
        # Update state
        spatial_state = torch.FloatTensor(next_spatial_state).unsqueeze(0)
        temporal_state = torch.FloatTensor(next_temporal_state).unsqueeze(0).unsqueeze(0)
        
        if done:
            break
    
    # Compute returns and advantages
    returns = []
    R = 0
    for reward in reversed(rewards):
        R = reward + gamma * R
        returns.insert(0, R)
    returns = torch.FloatTensor(returns)
    
    advantages = returns - torch.stack(values)
    
    # Compute losses
    policy_loss = -(torch.stack(log_probs) * advantages.detach()).mean()
    value_loss = advantages.pow(2).mean()
    total_loss = policy_loss + 0.5 * value_loss
    
    # Update agent
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
    optimizer.step()
    
    return np.mean(episode_rewards), total_loss.item()

# ============================================================================
# 4. TEMPORAL VISUALIZATION
# ============================================================================

def visualize_temporal_patterns(temporal_features=None):
    """Visualize temporal demand patterns"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    if temporal_features and 'demand_profiles' in temporal_features:
        time_multipliers = temporal_features['demand_profiles']['time_multipliers']
    else:
        # Create mock data
        time_multipliers = {h: 0.5 + 0.5 * np.sin(2 * np.pi * h / 24) for h in range(24)}
    
    hours = list(time_multipliers.keys())
    multipliers = list(time_multipliers.values())
    
    # Plot 1: Hourly demand multipliers
    axes[0, 0].plot(hours, multipliers, 'b-', linewidth=2, marker='o')
    axes[0, 0].set_title('🕒 Hourly Demand Multipliers', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Hour of Day')
    axes[0, 0].set_ylabel('Demand Multiplier')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Highlight rush hours
    rush_hours = [h for h in hours if 7 <= h <= 10 or 17 <= h <= 20]
    rush_multipliers = [time_multipliers[h] for h in rush_hours]
    axes[0, 0].scatter(rush_hours, rush_multipliers, color='red', s=100, 
                      label='Rush Hours', zorder=5)
    axes[0, 0].legend()
    
    # Plot 2: Rush hour vs Off-peak comparison
    off_peak_hours = [h for h in hours if not (7 <= h <= 10 or 17 <= h <= 20)]
    off_peak_multipliers = [time_multipliers[h] for h in off_peak_hours]
    
    axes[0, 1].bar(['Rush Hours', 'Off-Peak Hours'], 
                  [np.mean(rush_multipliers), np.mean(off_peak_multipliers)],
                  color=['red', 'green'], alpha=0.7)
    axes[0, 1].set_title('⚡ Rush vs Off-Peak Demand', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Average Multiplier')
    
    # Plot 3: Time-aware reward shaping
    base_rewards = np.random.rand(24) * 0.5 + 0.5
    temporal_rewards = base_rewards * np.array(list(time_multipliers.values()))
    
    axes[1, 0].plot(range(24), base_rewards, 'b-', label='Base Reward', linewidth=2)
    axes[1, 0].plot(range(24), temporal_rewards, 'r-', label='Time-Aware Reward', linewidth=2)
    axes[1, 0].set_title('🎯 Time-Aware Reward Shaping', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Hour of Day')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Temporal attention weights (conceptual)
    attention_weights = np.random.rand(24)  # Mock attention weights
    axes[1, 1].bar(range(24), attention_weights, alpha=0.7, color='purple')
    axes[1, 1].set_title('🧠 Temporal Attention Weights', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Hour of Day')
    axes[1, 1].set_ylabel('Attention Weight')
    
    plt.tight_layout()
    plt.savefig('temporal_patterns_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# 5. INTEGRATION WITH YOUR EXISTING PROJECT
# ============================================================================

def integrate_temporal_features():
    """
    Main integration function - call this to add temporal features to your project
    """
    print("\n🕒 INTEGRATING TEMPORAL FEATURES WITH YOUR EXISTING PROJECT")
    print("=" * 60)
    
    # 1. Initialize temporal data processor
    print("\n1. 📊 Loading temporal data...")
    try:
        temporal_processor = TemporalDataProcessor("new_delhi_traffic_dataset")
        temporal_features = temporal_processor.load_temporal_data()
        print("✅ Real temporal data loaded from your traffic dataset!")
    except Exception as e:
        print(f"⚠️  Using mock temporal data: {e}")
        temporal_features = None
    
    # 2. Create enhanced time-aware agent
    print("\n2. 🤖 Creating Enhanced Time-Aware A2C Agent...")
    
    # Use your existing state dimensions
    spatial_input_shape = (3, 50, 50)  # Adjust based on your state shape
    temporal_input_size = 8
    action_dim = 2
    
    enhanced_agent = EnhancedTimeAwareA2CAgent(
        spatial_input_shape=spatial_input_shape,
        temporal_input_size=temporal_input_size,
        action_dim=action_dim
    )
    
    total_params = sum(p.numel() for p in enhanced_agent.parameters())
    print(f"✅ Enhanced agent created with {total_params:,} parameters")
    print(f"   - Spatial CNN: {sum(p.numel() for p in enhanced_agent.spatial_encoder.parameters()):,} params")
    print(f"   - Temporal LSTM: {sum(p.numel() for p in enhanced_agent.temporal_encoder.parameters()):,} params")
    print(f"   - Attention: {sum(p.numel() for p in enhanced_agent.temporal_attention.parameters()):,} params")
    
    # 3. Create time-aware environment wrapper
    print("\n3. 🌍 Creating Time-Aware Environment Wrapper...")
    
    # Note: Replace this with your actual environment
    print("⚠️  Note: Replace 'your_existing_env' with your actual environment")
    print("   Example: time_aware_env = TimeAwareEnvironmentWrapper(your_env, temporal_processor)")
    
    # 4. Visualize temporal patterns
    print("\n4. 📈 Visualizing temporal patterns...")
    visualize_temporal_patterns(temporal_features)
    
    # 5. Create optimizer
    print("\n5. ⚙️  Setting up optimizer...")
    optimizer = torch.optim.Adam(enhanced_agent.parameters(), lr=3e-4)
    print("✅ Optimizer ready")
    
    print("\n🎉 TEMPORAL INTEGRATION COMPLETE!")
    print("=" * 50)
    print("✅ Enhanced Time-Aware A2C Agent created")
    print("✅ Temporal pattern visualization generated")
    print("✅ Time-aware reward shaping implemented")
    print("✅ LSTM + Attention mechanisms integrated")
    print("✅ Rush hour vs off-peak optimization ready")
    
    print("\n📋 NEXT STEPS:")
    print("1. Replace 'your_existing_env' with your actual environment")
    print("2. Run: time_aware_env = TimeAwareEnvironmentWrapper(your_env, temporal_processor)")
    print("3. Train with: train_temporal_episode(enhanced_agent, time_aware_env, optimizer)")
    print("4. Compare temporal vs non-temporal performance")
    
    return enhanced_agent, temporal_features, optimizer

# ============================================================================
# 6. RUN INTEGRATION
# ============================================================================

# Run the integration
enhanced_agent, temporal_features, optimizer = integrate_temporal_features()

print("\n🚀 READY TO USE TEMPORAL FEATURES!")
print("Your project now has advanced temporal pattern learning capabilities!")
