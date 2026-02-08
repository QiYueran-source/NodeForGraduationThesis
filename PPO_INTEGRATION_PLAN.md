# PPO Integration Implementation Plan

## Executive Summary

This document outlines the technical implementation plan for completing the PPO (Proximal Policy Optimization) integration with the custom MLP feature extractor in the Node project. The goal is to achieve end-to-end training where the MLP model learns to generate portfolio weights through reinforcement learning.

---

## 1. Current State Analysis

### 1.1 PPO.py Current Implementation
**File**: `src/worker/agent/rl/PPO.py`

**Status**: Incomplete skeleton
- Only contains imports and an empty PPO initialization
- Missing: policy configuration, training loop, model integration
- Missing: connection to NET_ADAPTER and MLP model

### 1.2 MLP Model (`src/worker/agent/net/mlp.py`)
**Status**: ✅ Complete
- Input: `(..., n, m, mask_len)` tensor
- Output: `(..., n+1)` softmax-normalized weights
- PyTorch `nn.Module` implementation
- Already handles device placement (CUDA/CPU)

### 1.3 NetAdapter (`src/worker/agent/net/adapter.py`)
**Status**: ✅ Complete
- Wraps MLP model
- Provides `act()` method for inference
- Handles device management
- **Issue**: `act()` expects torch.Tensor, but environment returns numpy arrays

### 1.4 RollingEnv (`src/worker/agent/env/rolling_env.py`)
**Status**: ✅ Complete
- Gymnasium-compatible environment
- Observation space: `Box(shape=(n, m, mask_len), dtype=np.float32)`
- Action space: `Box(shape=(n+1), low=0.0, high=1.0, dtype=np.float32)`
- Returns numpy arrays (standard Gymnasium format)

### 1.5 Reward System (`src/worker/agent/env/reward.py`)
**Status**: ✅ Complete
- Calculates portfolio performance metrics
- Computes normalized rewards
- Stores rewards in `_record` dictionary
- Environment queries rewards via `REWARD_MANAGER._record`

---

## 2. Technical Challenges & Solutions

### Challenge 1: Custom Policy Integration
**Problem**: Stable-baselines3 PPO requires a policy network that follows its architecture conventions. The MLP is a standalone PyTorch model.

**Solution**: Create a custom policy class that:
- Inherits from `stable_baselines3.common.policies.BasePolicy` or `ActorCriticPolicy`
- Wraps the existing MLP model as the actor network
- Adds a value network (critic) for PPO's value function estimation
- Handles observation normalization/preprocessing

### Challenge 2: Observation/Action Format Conversion
**Problem**: 
- Environment returns numpy arrays
- MLP expects torch.Tensor
- Stable-baselines3 handles conversions internally, but we need to ensure compatibility

**Solution**: 
- Use stable-baselines3's built-in conversion mechanisms
- Ensure observation space matches exactly: `(n, m, mask_len)`
- Action space is continuous, which PPO supports natively

### Challenge 3: End-to-End Training Flow
**Problem**: Need to connect:
- Environment → PPO → MLP → Actions → Rewards → Training

**Solution**: 
- Use stable-baselines3's `PPO.learn()` method
- Configure callbacks for checkpointing and logging
- Integrate with existing reward calculation system

### Challenge 4: Model Checkpointing
**Problem**: Need to save/load MLP weights during training

**Solution**: 
- Use `NetAdapter.get_checkpoint()` for saving
- Integrate with existing `SAVER` system
- Use PPO's built-in checkpointing for RL state

---

## 3. Implementation Steps

### Step 1: Create Custom Policy Class
**File**: `src/worker/agent/rl/policy.py` (NEW)

**Purpose**: Bridge between stable-baselines3 and custom MLP

**Key Components**:
```python
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn

class MLPFeatureExtractor(BaseFeaturesExtractor):
    """Wraps the custom MLP as a feature extractor"""
    def __init__(self, observation_space, features_dim=None):
        # Initialize MLP from NET_ADAPTER
        # Extract features from (n, m, mask_len) input
        pass

class MLPActorCriticPolicy(ActorCriticPolicy):
    """Custom policy using MLP as actor"""
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            features_extractor_class=MLPFeatureExtractor,
            features_extractor_kwargs={},
            **kwargs
        )
```

**Technical Details**:
- The MLP outputs `(n+1)` weights directly (action probabilities)
- For continuous actions, we need to wrap MLP output appropriately
- May need to adjust: MLP currently outputs softmax (discrete-like), but action space is continuous Box
- **Decision needed**: Use MLP output as mean of a distribution, or convert to discrete actions?

### Step 2: Update PPO.py Implementation
**File**: `src/worker/agent/rl/PPO.py`

**Required Changes**:
1. Import custom policy class
2. Configure PPO with proper hyperparameters from `train_config`
3. Initialize PPO with custom policy and environment
4. Implement training loop
5. Add model saving/loading integration

**Key Configuration**:
```python
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from src.worker.agent.rl.policy import MLPActorCriticPolicy
from src.worker.agent.env.rolling_env import ROLLING_ENV
from src.worker.cache import DATA_CACHE_POOL

# Get RL config from train_config
rl_config = DATA_CACHE_POOL.get_train_config().get('reinforcement_config', {})
learning_rate = rl_config.get('lr', 3e-4)
n_steps = rl_config.get('n_steps', 2048)
batch_size = rl_config.get('batch_size', 64)
n_epochs = rl_config.get('n_epochs', 10)
gamma = rl_config.get('gamma', 0.99)
gae_lambda = rl_config.get('gae_lambda', 0.95)
clip_range = rl_config.get('clip_range', 0.2)
ent_coef = rl_config.get('ent_coef', 0.01)
vf_coef = rl_config.get('vf_coef', 0.5)
max_grad_norm = rl_config.get('clip_grad_norm', 0.5)

ppo_model = PPO(
    policy=MLPActorCriticPolicy,
    env=ROLLING_ENV,
    learning_rate=learning_rate,
    n_steps=n_steps,
    batch_size=batch_size,
    n_epochs=n_epochs,
    gamma=gamma,
    gae_lambda=gae_lambda,
    clip_range=clip_range,
    ent_coef=ent_coef,
    vf_coef=vf_coef,
    max_grad_norm=max_grad_norm,
    verbose=1,
    tensorboard_log="./logs/tensorboard/",
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
```

### Step 3: Handle Action Space Mismatch
**Issue**: MLP outputs softmax (discrete-like), but action space is continuous Box

**Options**:
- **Option A**: Keep continuous action space, use MLP output as mean of Normal distribution
  - Modify MLP to output mean and (learned) std for each action dimension
  - More flexible but requires MLP architecture change
  
- **Option B**: Convert to discrete action space
  - Change action space to `Discrete(n+1)` 
  - Use MLP output directly as action probabilities
  - Simpler but less flexible
  
- **Option C**: Use MLP output as deterministic action (no exploration)
  - Not recommended for RL
  
- **Option D**: Use MLP output as mean, add fixed std for exploration
  - MLP outputs mean weights, add Gaussian noise during training
  - Best balance of simplicity and exploration

**Recommendation**: Option D - Use MLP output as mean of action distribution with learnable or fixed std.

### Step 4: Implement Training Loop
**File**: `src/worker/agent/rl/PPO.py` (continued)

**Training Function**:
```python
def train_ppo(total_timesteps=None):
    """
    Main training loop for PPO
    """
    # Get training config
    train_config = DATA_CACHE_POOL.get_train_config() or {}
    rl_config = train_config.get('reinforcement_config', {})
    
    # Total timesteps from config or default
    total_timesteps = total_timesteps or rl_config.get('total_timesteps', 1_000_000)
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=rl_config.get('save_freq', 10000),
        save_path='./logs/checkpoints/',
        name_prefix='ppo_model'
    )
    
    # Train
    ppo_model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        log_interval=rl_config.get('log_interval', 10)
    )
    
    return ppo_model
```

### Step 5: Integrate with Worker Training Flow
**File**: `worker.py`

**Update `train()` function**:
```python
def train():
    """使用NET_ADAPTER,ENV和RL_ADAPTER进行训练"""
    from src.worker.agent.rl.PPO import train_ppo, ppo_model
    
    logger.info("开始PPO训练")
    train_ppo()
    logger.info("PPO训练完成")
    
    # Save final model
    # Integration with SAVER.save_model() if needed
```

### Step 6: Model Saving/Loading Integration
**Files**: 
- `src/worker/agent/rl/PPO.py`
- `src/worker/save/saver.py` (may need updates)

**Save PPO Model**:
- Use `ppo_model.save(path)` for full PPO state
- Use `NET_ADAPTER.get_checkpoint()` for MLP weights only
- Integrate with existing `SAVER.save_model()` if needed

**Load PPO Model**:
- Use `PPO.load(path)` for full state restoration
- Ensure NET_ADAPTER is initialized with same config

---

## 4. Files to Create/Modify

### New Files:
1. **`src/worker/agent/rl/policy.py`**
   - Custom policy class wrapping MLP
   - Feature extractor adapter
   - Actor-critic architecture

### Files to Modify:
1. **`src/worker/agent/rl/PPO.py`**
   - Complete PPO initialization
   - Training loop implementation
   - Model save/load functions
   - Integration with config system

2. **`worker.py`**
   - Update `train()` function to call PPO training
   - Ensure proper initialization order

3. **`src/worker/agent/rl/adapter.py`** (optional)
   - May need to add RL-specific adapter methods
   - Currently just a placeholder

4. **`src/worker/agent/net/adapter.py`** (minor)
   - May need to add methods for policy integration
   - Ensure compatibility with stable-baselines3

5. **`src/worker/agent/__init__.py`** (minor)
   - Export PPO model/adapter if needed

---

## 5. Configuration Structure

### Required in `train_config` (from `pool.py`):
```python
train_config = {
    'reinforcement_config': {
        'lr': 3e-4,                    # Learning rate
        'n_steps': 2048,                # Steps per update
        'batch_size': 64,               # Batch size
        'n_epochs': 10,                 # Optimization epochs per update
        'gamma': 0.99,                  # Discount factor
        'gae_lambda': 0.95,             # GAE lambda
        'clip_range': 0.2,              # PPO clip range
        'ent_coef': 0.01,               # Entropy coefficient
        'vf_coef': 0.5,                 # Value function coefficient
        'clip_grad_norm': 0.5,          # Gradient clipping
        'total_timesteps': 1_000_000,   # Total training steps
        'save_freq': 10000,             # Checkpoint frequency
        'log_interval': 10,             # Logging frequency
    },
    # ... existing config ...
}
```

---

## 6. Critical Design Decisions

### Decision 1: Action Space Type
**Current**: Continuous Box `(n+1,)` with values in [0, 1]
**MLP Output**: Softmax probabilities (discrete-like)

**Recommendation**: 
- Keep continuous action space
- Use MLP output as mean of action distribution
- Add learnable or fixed standard deviation for exploration
- Sample actions from Normal(mean=MLP_output, std=learned_or_fixed)

### Decision 2: Value Function Architecture
**Options**:
- Separate value network (recommended)
- Shared feature extractor + separate value head
- Use MLP features + separate linear value head

**Recommendation**: Shared MLP feature extractor + separate value head

### Decision 3: Observation Preprocessing
**Current**: Raw factor values `(n, m, mask_len)`

**Considerations**:
- May need normalization
- Stable-baselines3 can handle this via `VecNormalize`
- Or handle in custom feature extractor

**Recommendation**: Use stable-baselines3's `VecNormalize` wrapper if needed

---

## 7. Testing Strategy

### Unit Tests:
1. Policy forward pass with sample observations
2. Action sampling from policy
3. Value function estimation
4. PPO update step

### Integration Tests:
1. Environment → Policy → Action flow
2. Training loop execution
3. Model save/load
4. Reward calculation integration

### End-to-End Test:
1. Full training run with small timesteps
2. Verify model improves (reward increases)
3. Verify checkpoints are saved correctly

---

## 8. Implementation Priority

### Phase 1: Core Integration (High Priority)
1. Create custom policy class
2. Implement basic PPO initialization
3. Test forward pass and action sampling

### Phase 2: Training Loop (High Priority)
1. Implement training function
2. Add callbacks
3. Integrate with worker.py

### Phase 3: Model Persistence (Medium Priority)
1. Save/load integration
2. Checkpoint management
3. Resume training capability

### Phase 4: Optimization (Low Priority)
1. Hyperparameter tuning
2. Performance optimization
3. Advanced callbacks (e.g., early stopping)

---

## 9. Potential Issues & Mitigations

### Issue 1: Dimension Mismatches
**Risk**: Observation/action shapes don't match between components
**Mitigation**: 
- Add shape validation in policy initialization
- Use assertions in forward pass
- Test with actual environment observations

### Issue 2: Device Mismatches
**Risk**: Model on GPU, environment on CPU (or vice versa)
**Mitigation**:
- Ensure consistent device usage
- Use `device` parameter in PPO initialization
- Convert tensors appropriately

### Issue 3: Reward Timing
**Risk**: Rewards not available when PPO expects them
**Mitigation**:
- Ensure reward calculation happens before PPO update
- Use environment's reward system correctly
- May need to adjust reward calculation timing

### Issue 4: Memory Issues
**Risk**: Large observation/action spaces cause OOM
**Mitigation**:
- Monitor memory usage
- Adjust batch_size and n_steps
- Use gradient accumulation if needed

---

## 10. Next Steps

1. **Review and approve this plan**
2. **Clarify action space decision** (continuous vs discrete)
3. **Create `policy.py`** with custom policy class
4. **Complete `PPO.py`** implementation
5. **Test with small training run**
6. **Iterate based on results**

---

## Appendix: Stable-Baselines3 Integration Reference

### Key Classes:
- `stable_baselines3.PPO`: Main PPO algorithm
- `stable_baselines3.common.policies.ActorCriticPolicy`: Base policy class
- `stable_baselines3.common.torch_layers.BaseFeaturesExtractor`: Feature extractor base
- `stable_baselines3.common.callbacks.CheckpointCallback`: Checkpointing

### Key Methods:
- `PPO.learn(total_timesteps)`: Train the model
- `PPO.predict(obs)`: Get action from observation
- `PPO.save(path)`: Save model
- `PPO.load(path)`: Load model

### Observation/Action Handling:
- Observations: Automatically converted numpy → torch
- Actions: Policy outputs torch → converted to numpy for environment
- Continuous actions: Use `SquashedDiagGaussianDistribution` or similar
