import sumo_rl
from Recompensa import recompensa
from Observacion import ObservacionCopilco
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

AGENT_NAME = 'J26'

env = sumo_rl.environment.env.SumoEnvironment(
    net_file='./Mapas/copilco/copilco2.net.xml',
    route_file='./Mapas/copilco/copilco2.rou.xml',
    use_gui=False,
    num_seconds=800,
    reward_fn=recompensa,
    observation_class=ObservacionCopilco
)

num_episodes = 20

recompensas_episodio = []
recompensa1 = []
recompensa2 = []
recompensa3 = []

for episode in range(num_episodes):
        observations = env.reset()
        done = False
        orden = 0
        
        total_rewards = np.zeros(3) 
        
        while not done:
            action = orden
            observations, rewards, terminations, info = env.step({AGENT_NAME: action})

            cambio = observations['J26'][1]
            if cambio ==1:
                orden+=1
                if orden>=4:
                    orden = 0

            done = terminations['__all__']

            reward_vector = [
                rewards[AGENT_NAME][0],
                rewards[AGENT_NAME][1],
                rewards[AGENT_NAME][2]
            ]
            total_rewards += np.array(reward_vector)
            

        recompensas_episodio.append(total_rewards)
        recompensa1.append(total_rewards[0])
        recompensa2.append(total_rewards[1])
        recompensa3.append(total_rewards[2])
        
        print(f"\n\nEpisodio {episode + 1}, Recompensas totales: {total_rewards}\n")
    
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(recompensa1, label='Recompensa 1')
plt.title('Convergencia de Recompensa 1')
plt.xlabel('Episodio')
plt.ylabel('Recompensa')
plt.grid()
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(recompensa2, label='Recompensa 2', color='orange')
plt.title('Convergencia de Recompensa 2')
plt.xlabel('Episodio')
plt.ylabel('Recompensa')
plt.grid()
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(recompensa3, label='Recompensa 3', color='green')
plt.title('Convergencia de Recompensa 3')
plt.xlabel('Episodio')
plt.ylabel('Recompensa')
plt.grid()
plt.legend()

plt.tight_layout()
plt.savefig('clasico.png')

desc=pd.DataFrame(recompensas_episodio)
desc.to_csv('./clasico.csv')
print(desc.describe())