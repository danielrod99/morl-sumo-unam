import gymnasium as gym
from gymnasium import spaces
import sumo_rl
import numpy as np
import matplotlib.pyplot as plt
from morl_baselines.multi_policy.pareto_q_learning.pql import PQL
from morl_baselines.common.pareto import get_non_dominated
from Observacion import ObservacionCopilco
from Recompensa import recompensa

AGENT_NAME = 'J26'

class CompatibleMOEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        
        self.phase_bins = [0, 1, 2, 3, 4]
        self.min_bins = [0, 0.5, 1]
        self.density_bins = np.linspace(0, 1, 5)
        self.queue_bins = np.linspace(0, 1, 5)  
        
        n_phases = len(self.phase_bins) - 1
        n_min = len(self.min_bins) - 1
        n_density = len(self.density_bins) - 1
        n_queue = len(self.queue_bins) - 1
        self.n_states = n_phases * n_min * (n_density ** 4) * (n_queue ** 4)
        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(4)
        self.env.unwrapped.reward_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        
    def reset(self, **kwargs):
       obs = self.env.reset(**kwargs)
       return {AGENT_NAME: self._discretize_obs(obs[AGENT_NAME])}, {}

    def step(self, action_dict):
        raw_obs, raw_reward, raw_terminated, info = self.env.step(action_dict)

        obs = {AGENT_NAME: self._discretize_obs(raw_obs[AGENT_NAME])}
        reward = {AGENT_NAME: raw_reward[AGENT_NAME]}
        terminated = {AGENT_NAME: raw_terminated[AGENT_NAME], "__all__": raw_terminated["__all__"]}

        return obs, reward, terminated, info

    def _discretize_obs(self, obs):
        agent_obs = obs
        phase = np.digitize(agent_obs[0], self.phase_bins) - 1
        min_met = np.digitize(agent_obs[1], self.min_bins) - 1
        densities = [np.digitize(d, self.density_bins) - 1 for d in agent_obs[2:6]]
        queues = [np.digitize(q, self.queue_bins) - 1 for q in agent_obs[6:10]]
        
        discrete_state = phase
        discrete_state = discrete_state * len(self.min_bins) + min_met
        for d in densities:
            discrete_state = discrete_state * len(self.density_bins) + d
        for q in queues:
            discrete_state = discrete_state * len(self.queue_bins) + q
            
        return discrete_state

# Crear el entorno
env = sumo_rl.environment.env.SumoEnvironment(
    net_file='./Mapas/copilco/copilco2.net.xml',
    route_file='./Mapas/copilco/copilco2.rou.xml',
    use_gui=False,
    num_seconds=1800,
    reward_fn=recompensa,
    observation_class=ObservacionCopilco
)

discrete_env = CompatibleMOEnv(env)

ref_point = np.array([0, 0, 0])
agent = PQL(
    env=discrete_env,
    ref_point=ref_point,
    gamma=0.95,
    initial_epsilon=1.0,
    epsilon_decay_steps=10000,
    final_epsilon=0.05,
    log=False
)

def weighted_sum(reward_vector, weights=np.array([1.0, 1.0, 1.0])):
    return np.dot(reward_vector, weights)

def update_pql(agent, state, action, reward_vector, next_state, done):
    s = state
    a = action
    r = reward_vector
    s_prime = next_state

    # Update count and average reward
    agent.counts[s, a] += 1
    alpha = 1 / agent.counts[s, a]
    agent.avg_reward[s, a] = (1 - alpha) * agent.avg_reward[s, a] + alpha * r

    # Q set para s,a
    future_set = agent.calc_non_dominated(s_prime) if not done else {tuple(np.zeros(agent.num_objectives))}
    new_vectors = [tuple(agent.avg_reward[s, a] + agent.gamma * np.array(v)) for v in future_set]

    # Actualizar conjunto no dominado
    current_set = agent.non_dominated[s][a]
    combined_set = current_set.union(new_vectors)
    agent.non_dominated[s][a] = get_non_dominated(combined_set)

num_episodes = 1
episode_rewards = []
pareto_front_history = []

for episode in range(num_episodes):
    [state, _ ]= discrete_env.reset()
    done = False
    total_rewards = np.zeros(3)

    while not done:
        action = agent.select_action(state[AGENT_NAME], score_func=weighted_sum)
        next_obs, reward, terminated, info = discrete_env.step({AGENT_NAME: action})
        
        done = terminated["__all__"]
        reward_vector = np.array(reward[AGENT_NAME])
        
        update_pql(agent, state[AGENT_NAME], action, reward_vector, next_obs[AGENT_NAME], done)
        total_rewards += reward_vector
        state = next_obs

    episode_rewards.append(total_rewards)
    
    current_pareto = agent.get_local_pcs()
    pareto_front_history.append(current_pareto)
    
    print(f"Episodio {episode + 1}, Recompensas: {total_rewards}")
    print(f"Frente de Pareto actual: {current_pareto}")

# Gráfica de recompensas por episodio
plt.figure(figsize=(15, 5))

r1 = [r[0] for r in episode_rewards]
r2 = [r[1] for r in episode_rewards]
r3 = [r[2] for r in episode_rewards]

plt.subplot(1, 3, 1)
plt.plot(r1)
plt.title('Velocidad')
plt.grid()

plt.subplot(1, 3, 2)
plt.plot(r2, color='orange')
plt.title('Cola Normalizada')
plt.grid()

plt.subplot(1, 3, 3)
plt.plot(r3, color='green')
plt.title('Tiempo de Espera Normalizado')
plt.grid()

plt.tight_layout()
plt.savefig('pql_sumo_results.png')

# Gráfica 3D del frente de Pareto
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

pareto_points = pareto_front_history[-1]
if pareto_points:
    x = [p[0] for p in pareto_points]
    y = [p[1] for p in pareto_points]
    z = [p[2] for p in pareto_points]

    ax.scatter(x, y, z, c='red', marker='o')
    ax.set_xlabel('Velocidad')
    ax.set_ylabel('Cola Norm')
    ax.set_zlabel('Espera Norm')
    ax.set_title('Frente de Pareto')
    plt.savefig('pareto_front_3d.png')

# Mostrar políticas óptimas
print("\nPolíticas óptimas encontradas:")
for i, policy in enumerate(pareto_front_history[-1]):
    print(f"Política {i+1}: Velocidad={policy[0]:.2f}, Cola={policy[1]:.2f}, Espera={policy[2]:.2f}")
