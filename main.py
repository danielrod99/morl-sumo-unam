import sumo_rl
from ParetoQLearning import ParetoQLearningSUMO
from Recompensa import recompensa
from Observacion import ObservacionCopilco
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dill as pickle
import os

def guardar_agente(agente,nombre_archivo='modelo_paretoql.pkl'):
        with open(nombre_archivo, 'wb') as f:
            pickle.dump(agente, f)
        print(f"Modelo guardado en: {nombre_archivo}")

def cargar_agente(nombre_archivo='modelo_paretoql.pkl'):
    if os.path.exists(nombre_archivo):
        with open(nombre_archivo, 'rb') as f:
            agente = pickle.load(f)
        print(f"Modelo cargado: {nombre_archivo}")
        return agente
    else:
        print(f"El archivo {nombre_archivo} no existe")
        return None

AGENT_NAME = 'J26'

env = sumo_rl.environment.env.SumoEnvironment(
    net_file='./Mapas/copilco/copilco2.net.xml',
    route_file='./Mapas/copilco/copilco2.rou.xml',
    use_gui=False,
    num_seconds=900,
    delta_time=3,
    reward_fn=recompensa,
    observation_class=ObservacionCopilco
)

def graficas_y_modelo(recompensas_episodio,recompensa1,recompensa2,recompensa3,nombre):
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
    plt.savefig('./graficas_modelo/'+nombre+'_entrenamiento.png')
    guardar_agente(agent,nombre_archivo='./graficas_modelo/'+nombre+'_modelo.pkl')
    
    desc = pd.DataFrame(recompensas_episodio)
    print(desc.describe())
    desc.to_csv('./graficas_modelo/'+nombre+'_valores.csv')

agent = ParetoQLearningSUMO()

def train(num_episodes=100):
    recompensas_episodio = []
    recompensa1 = []
    recompensa2 = []
    recompensa3 = []
    global_step = 0
    
    for episode in range(num_episodes):
        observations = env.reset()
        done = False
        episode_step = 0
        total_rewards = np.zeros(3) 
        times = [0,0,0,0]
        while not done:
            obs = observations[AGENT_NAME]
            
            action = agent.accion(obs)
            times[action] +=1
            observations, rewards, terminations, info = env.step({AGENT_NAME: action})
            episode_step+=env.delta_time
            done = terminations['__all__']
            global_step += 1
            reward_vector = [
                rewards[AGENT_NAME][0],
                rewards[AGENT_NAME][1],
                rewards[AGENT_NAME][2]
            ]
            total_rewards += np.array(reward_vector)
            
            next_obs = observations[AGENT_NAME]
            agent.actualizarQ(obs, action, reward_vector, next_obs, done)

            if episode > 2:
                agent.actualizar_epsilon_manual(max(agent.min_epsilon, agent.epsilon * np.exp(-agent.decay * global_step)))
                

        recompensas_episodio.append(total_rewards)
        recompensa1.append(total_rewards[0])
        recompensa2.append(total_rewards[1])
        recompensa3.append(total_rewards[2])
        
        print(f"\n\nEpisodio {episode + 1}, Recompensas totales: {total_rewards}\n")
        print(f"Epsilon actual: {agent.epsilon:.4f}")
        print('Tiempo en verde promedio: ')
        for i in range(0,4):
            print(i, times[i],(times[i]/4)*env.delta_time)

        if (episode>10 and episode % 10 == 0) or episode==num_episodes-1:
            graficas_y_modelo(recompensas_episodio,recompensa1,recompensa2,recompensa3,str(episode+1))

    
def test(num_episodes=5):
    #agent = cargar_agente()
    recompensas_episodio = []
    recompensa1 = []
    recompensa2 = []
    recompensa3 = []
    for episode in range(num_episodes):
        observations = env.reset()
        done = False
        total_rewards = np.zeros(3) 
        while not done:
            obs = observations[AGENT_NAME]
            action = agent.mejor_accion(obs)
            observations, rewards, terminations, _ = env.step({AGENT_NAME: action})
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
    plt.title('Recompensa 1')
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa')
    plt.grid()
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(recompensa2, label='Recompensa 2', color='orange')
    plt.title('Recompensa 2')
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa')
    plt.grid()
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(recompensa3, label='Recompensa 3', color='green')
    plt.title('Recompensa 3')
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa')
    plt.grid()
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('prueba.png')
    desc = pd.DataFrame(recompensas_episodio)
    desc.to_csv('./prueba.csv')
    print(desc.describe())

train()

test(num_episodes=20)

env.close()