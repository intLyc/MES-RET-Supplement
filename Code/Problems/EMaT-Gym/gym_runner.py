import gymnasium as gym
import numpy as np
import multiprocessing as mp

ENV_NAMES = [
    "MountainCarContinuous-v0", "MountainCar-v0",
    "Pendulum-v1", "CartPole-v1", "Acrobot-v1", 
    "LunarLander-v3",
    "BipedalWalker-v3", 
    "InvertedPendulum-v5", "InvertedDoublePendulum-v5", 
    "Reacher-v5", "Pusher-v5",
    "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5",
    "Swimmer-v5", "Ant-v5", 
    "Humanoid-v5", "HumanoidStandup-v5"
]

env_normalizers = {}

class RunningNormalizer:
    def __init__(self, obs_dim, eps=1e-8):
        self.mean = np.zeros(obs_dim, dtype=np.float32)
        self.std = np.ones(obs_dim, dtype=np.float32)
        self.count = eps
        self.eps = eps
        self.M2 = np.zeros(obs_dim, dtype=np.float32)

    def update(self, obs_batch):
        batch_mean = np.mean(obs_batch, axis=0)
        batch_count = obs_batch.shape[0]
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = (self.count * self.mean + batch_count * batch_mean) / total_count
        m_a = self.M2
        m_b = np.sum((obs_batch - batch_mean)**2, axis=0)
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        
        self.M2 = M2
        self.mean = new_mean
        self.count = total_count
        self.std = np.sqrt(M2 / self.count) + self.eps

    def merge_stats(self, batch_mean, batch_M2, batch_count):
        if batch_count == 0:
            return
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = (self.count * self.mean + batch_count * batch_mean) / total_count
        
        # Welford
        term = delta**2 * self.count * batch_count / total_count
        self.M2 = self.M2 + batch_M2 + term
        
        self.mean = new_mean
        self.count = total_count
        self.std = np.sqrt(self.M2 / self.count) + self.eps

def reset_global_normalizer():
    global env_normalizers
    env_normalizers = {}
    
def get_normalizer(env_name):
    global env_normalizers
    if env_name not in env_normalizers:
        env = gym.make(env_name)
        obs_space = env.observation_space
        obs_is_discrete = isinstance(obs_space, gym.spaces.Discrete)
        obs_dim = obs_space.n if obs_is_discrete else obs_space.shape[0]
        env.close()
        env_normalizers[env_name] = RunningNormalizer(obs_dim)
    return env_normalizers[env_name]

def get_normalizer_stats(env_name):
    global env_normalizers
    if env_name in env_normalizers:
        normalizer = env_normalizers[env_name]
        return {'mean': normalizer.mean, 'std': normalizer.std}
    else:
        return None

def run_episode_single(env_idx, params, hidden_size, normalizer=None):
    env_name = ENV_NAMES[env_idx]
    # env = gym.make(env_name) 
    env = gym.vector.SyncVectorEnv([
        (lambda env_=env_name: lambda: gym.make(env_))()
    ])
    obs_space = env.single_observation_space
    act_space = env.single_action_space
    obs_is_discrete = isinstance(obs_space, gym.spaces.Discrete)
    act_is_discrete = isinstance(act_space, gym.spaces.Discrete)
    obs_dim = obs_space.n if obs_is_discrete else obs_space.shape[0]
    act_dim = act_space.n if act_is_discrete else act_space.shape[0]
    max_steps = getattr(env.spec, 'max_episode_steps', None) or 1000
    
    offset = 0
    # Shared parameters cross all environments
    b1 = params[offset: offset + hidden_size].reshape(1, hidden_size)
    offset += hidden_size
    W2 = params[offset: offset + hidden_size*hidden_size].reshape(1, hidden_size, hidden_size)
    offset += hidden_size*hidden_size
    b2 = params[offset: offset + hidden_size].reshape(1, hidden_size)
    offset += hidden_size
    # Environment-specific parameters
    W1 = params[offset: offset + obs_dim*hidden_size].reshape(1, obs_dim, hidden_size);
    offset += obs_dim*hidden_size
    W3 = params[offset: offset + hidden_size*act_dim].reshape(1, hidden_size, act_dim)
    offset += hidden_size*act_dim
    b3 = params[offset: offset + act_dim].reshape(1, act_dim)
    
    if not act_is_discrete:
        scale = (act_space.high - act_space.low) / 2
        offset_a = (act_space.high + act_space.low) / 2
        
    normalizer = get_normalizer(env_name) if normalizer is None else normalizer

    obs, _ = env.reset()
    
    obs_batch = []
    total_reward = 0.0
    for _ in range(max_steps):
        if obs_is_discrete:
            obs_vec = np.eye(obs_dim)[obs].reshape(1, obs_dim)
        else:
            # normalizer.update(obs.reshape(1, -1))
            obs_batch.append(obs.copy())
            obs_vec = (obs.reshape(1, obs_dim) - normalizer.mean) / (normalizer.std + 1e-8)
            
        h1 = np.tanh(np.einsum('ni,nih->nh', obs_vec, W1) + b1)
        h2 = np.tanh(np.einsum('nh,nhh->nh', h1, W2) + b2)
        logits = np.einsum('nh,nha->na', h2, W3) + b3
        
        if act_is_discrete:
            # action = np.argmax(logits)
            logits -= np.max(logits, axis=1, keepdims=True)  # Log-sum-exp trick for numerical stability
            prob = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
            action = [np.random.choice(act_dim, p=prob[0])]
        else:
            a = np.tanh(logits)
            action = np.clip(a * scale + offset_a, act_space.low, act_space.high)
            # action = action.flatten()
            
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward[0]
        if terminated or truncated:
            break
        
    batch_mean, batch_M2, batch_count = None, None, 0
    if not obs_is_discrete and len(obs_batch) > 0:
        obs_batch = np.array(obs_batch)
        batch_count = obs_batch.shape[0]
        batch_mean = np.mean(obs_batch, axis=0)
        batch_M2 = np.sum((obs_batch - batch_mean)**2, axis=0)
    
    env.close()
    return total_reward, (batch_mean, batch_M2, batch_count)

def run_episode_seq(env_idx, params, hidden_size, num_rollouts):
    env_name = ENV_NAMES[env_idx]
    normalizer = get_normalizer(env_name)
    
    n_params = params.shape[0]
    avg_rewards = np.zeros(n_params, dtype=float)
    
    for i in range(n_params):
        total_reward = 0.0
        for _ in range(num_rollouts):
            reward, stats = run_episode_single(env_idx, params[i], hidden_size, normalizer)
            total_reward += reward
            batch_mean, batch_M2, batch_count = stats
            if batch_count > 0:
                normalizer.merge_stats(batch_mean, batch_M2, batch_count)
        avg_rewards[i] = total_reward / num_rollouts
    return avg_rewards.tolist()

def _run_single_wrapper(args):
    env_idx, param, hidden_size, normalizer_state = args
    
    normalizer = RunningNormalizer(obs_dim=normalizer_state['mean'].shape[0])
    normalizer.mean = normalizer_state['mean'].copy()
    normalizer.std = normalizer_state['std'].copy()
    normalizer.count = normalizer_state['count']
    normalizer.M2 = normalizer_state['M2'].copy()
    
    reward, stats = run_episode_single(env_idx, param, hidden_size, normalizer)
    return reward, stats

def run_episode_mp(env_idx, params, hidden_size, num_rollouts):
    env_name = ENV_NAMES[env_idx]
    normalizer = get_normalizer(env_name)
    
    normalizer_state = {
        'mean': normalizer.mean,
        'std': normalizer.std,
        'count': normalizer.count,
        'M2': normalizer.M2
    }
    
    n_params = params.shape[0]
    args_list = [
        (env_idx, params[i], hidden_size, normalizer_state)
        for i in range(n_params)
        for _ in range(num_rollouts)
    ]
    
    with mp.Pool(min(mp.cpu_count(), len(args_list))) as pool:
        results = pool.map(_run_single_wrapper, args_list)
    
    rewards = []
    all_stats = []
    for reward, stats in results:
        rewards.append(reward)
        batch_mean, batch_M2, batch_count = stats
        if batch_count > 0:
            all_stats.append((batch_mean, batch_M2, batch_count))
    
    for stats in all_stats:
        normalizer.merge_stats(*stats)
    
    rewards = np.array(rewards).reshape(n_params, num_rollouts)
    return np.mean(rewards, axis=1).tolist()

def run_episode_vec(env_idx, params, hidden_size, num_rollouts):
    env_name = ENV_NAMES[env_idx]
    
    n_params = params.shape[0]
    repeated = np.repeat(params, num_rollouts, axis=0)
    n_envs = repeated.shape[0]

    env = gym.vector.SyncVectorEnv([
        (lambda env_=env_name: lambda: gym.make(env_))()
        for _ in range(n_envs)
    ])

    obs_space = env.single_observation_space
    act_space = env.single_action_space
    obs_is_discrete = isinstance(obs_space, gym.spaces.Discrete)
    act_is_discrete = isinstance(act_space, gym.spaces.Discrete)
    obs_dim = obs_space.n if obs_is_discrete else obs_space.shape[0]
    act_dim = act_space.n if act_is_discrete else act_space.shape[0]
    max_steps = getattr(env.spec, 'max_episode_steps', None) or 1000
    
    normalizer = get_normalizer(env_name)

    offset = 0
    # Shared parameters cross all environments
    b1 = repeated[:, offset:offset + hidden_size]
    offset += hidden_size
    W2 = repeated[:, offset:offset + hidden_size*hidden_size].reshape(n_envs, hidden_size, hidden_size)
    offset += hidden_size*hidden_size
    b2 = repeated[:, offset:offset + hidden_size]
    offset += hidden_size
    # Environment-specific parameters
    W1 = repeated[:, offset:offset + obs_dim*hidden_size].reshape(n_envs, obs_dim, hidden_size)
    offset += obs_dim*hidden_size
    W3 = repeated[:, offset:offset + hidden_size*act_dim].reshape(n_envs, hidden_size, act_dim)
    offset += hidden_size*act_dim
    b3 = repeated[:, offset:offset + act_dim]
    
    obs, _ = env.reset()
    
    total_rewards = np.zeros(n_envs, dtype=float)
    done_flags = np.zeros(n_envs, dtype=bool)
    
    if not act_is_discrete:
        scale = (act_space.high - act_space.low) / 2
        offset_a = (act_space.high + act_space.low) / 2

    obs_batch = []
    for _ in range(max_steps):
        if np.all(done_flags):
            break
        
        active_mask = ~done_flags

        if obs_is_discrete:
            obs_vec = np.zeros((n_envs, obs_dim))
            obs_vec[active_mask] = np.eye(obs_dim)[obs[active_mask]]
        else:
            obs_batch.append(obs[active_mask].copy())
            # normalizer.update(obs[active_mask])
            obs_vec = np.zeros_like(obs)
            obs_vec[active_mask] = (obs[active_mask] - normalizer.mean) / (normalizer.std + 1e-8)
            obs_vec[active_mask] = obs[active_mask]

        h1 = np.tanh(np.einsum('ni,nih->nh', obs_vec, W1) + b1)
        h2 = np.tanh(np.einsum('nh,nhh->nh', h1, W2) + b2)
        logits = np.einsum('nh,nha->na', h2, W3) + b3

        if act_is_discrete:
            # actions = np.argmax(logits, axis=1)
            logits -= np.max(logits, axis=1, keepdims=True)  # Log-sum-exp trick for numerical stability
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
            actions = np.array([
                np.random.choice(act_dim, p=probs[i])
                if active_mask[i] else 0
                for i in range(n_envs)
            ])
        else:
            a = np.tanh(logits)
            actions = a * scale + offset_a
            actions = np.clip(actions, act_space.low, act_space.high)
            actions[~active_mask] = 0.0
        
        obs_next, rewards, terminateds, truncateds, _ = env.step(actions)
        total_rewards += rewards * active_mask
        done_flags = done_flags | terminateds | truncateds
        obs = obs_next

    normalizer.update(np.concatenate(obs_batch, axis=0)) if not obs_is_discrete else None
    env.close()
    grouped = total_rewards.reshape(n_params, num_rollouts)
    return np.mean(grouped, axis=1)

def run_episode(env_idx, params, hidden_size, num_rollouts, multi_process=False):
    if multi_process:
        return run_episode_mp(env_idx, params, hidden_size, num_rollouts)
    elif env_idx in [4, 5, 6, 12, 13, 15]:
        return run_episode_seq(env_idx, params, hidden_size, num_rollouts)
    elif env_idx in [0, 1, 2, 3, 7, 8, 9, 10, 11, 14, 16, 17]:
        return run_episode_vec(env_idx, params, hidden_size, num_rollouts)
    else:
        raise ValueError(f"Invalid environment index: {env_idx}")
    
if __name__ == '__main__':
    import time
    from scipy.io import loadmat

    def override_env_normalizer(env_name, mean, std):
        global env_normalizers
        env_normalizers[env_name] = RunningNormalizer(mean.shape[0])
        env_normalizers[env_name].mean = mean
        env_normalizers[env_name].std = std
        env_normalizers[env_name].count = 100000

    dec_path = 'Dec.mat'
    norm_path = 'Normalizer.mat'

    dec_mat = loadmat(dec_path)['Dec']
    norm_mat = loadmat(norm_path)['normalizer']

    hidden_size = 16
    num_rollouts = 3
    print(f"{'Env':<35} | {'Sequential':<8} | {'Vectorized':<8} | {'Multi-Process':<8} | {'Selected':<8}")
    print('-'*85)

    for idx, name in enumerate(ENV_NAMES):
        env = gym.make(name);
        obs_space = env.observation_space
        act_space = env.action_space
        obs_dim = obs_space.n if isinstance(obs_space, gym.spaces.Discrete) else obs_space.shape[0]
        act_dim = act_space.n if isinstance(act_space, gym.spaces.Discrete) else act_space.shape[0]
        
        param_size = hidden_size + hidden_size*hidden_size + hidden_size + obs_dim*hidden_size + hidden_size*act_dim + act_dim
        base_param = np.array(dec_mat[idx], dtype=np.float32)
        base_param = base_param[0:param_size]

        population_size = 500
        noise_std = 0.001
        params = base_param[None, :] + np.random.normal(0, noise_std, size=(population_size, param_size)).astype(np.float32)

        reset_global_normalizer()
        norm_entry = norm_mat[0, idx]
        mean = norm_entry['mean'][0, 0].flatten()
        std = norm_entry['std'][0, 0].flatten()
        override_env_normalizer(name, mean, std)

        t0 = time.perf_counter()
        rewards = run_episode_seq(idx, params, hidden_size, num_rollouts)
        t_seq = time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = run_episode_vec(idx, params, hidden_size, num_rollouts)
        t_vec = time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = run_episode_mp(idx, params, hidden_size, num_rollouts)
        t_async = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        # _ = run_episode(idx, params, hidden_size, num_rollouts)
        t_sel = time.perf_counter() - t0

        print(f"{idx:<2} | {name:<30} | {t_seq:>8.4f} | {t_vec:>8.4f} | {t_async:>8.4f} | {t_sel:>8.4f}")
        