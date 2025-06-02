from collections import defaultdict
import numpy as np
import random

class ParetoQLearningSUMO():
    def __init__(self, alpha= 0.1, gamma=0.99,epsilon=0.1,epsilon_decaimiento=0.995, min_epsilon=0.01):
        self.alpha=alpha
        self.gamma=gamma
        self.epsilon = epsilon
        self.epsilon_decaimiento=epsilon_decaimiento
        self.decay = 1e-5
        self.min_epsilon = min_epsilon

        self.acciones = [0,1,2,3]
        self.objetivos = 3
        self.q = defaultdict(lambda: np.zeros((len(self.acciones), self.objetivos))) 
        self.frente_pareto = defaultdict(list)
        self.hv_por_estado = defaultdict(float)
    
    def actualiza_epsilon(self):
        if self.epsilon>self.min_epsilon:
            self.epsilon*=self.epsilon_decaimiento
    
    def actualizar_epsilon_manual(self,epsilon):
        self.epsilon = epsilon

    def parse_observacion(self,observacion):
        fase_actual = int(observacion[0])
        min_tiempo_cumplido = int(observacion[1])
        densidades = observacion[2:6]
        colas = observacion[6:10]     

        densidad = np.mean(densidades)
        cola = np.mean(colas)

        if densidad <= 1/2:
            dens = -1
        else:
            dens =1
        
        if cola <= 1/2:
            cl = -1
        else:
            cl=1
        estado = (
            fase_actual,
            dens,
            cl
        )
        return estado

    def soluciones_pareto_optimas(self, estado):
        if estado not in self.q:
            return list(range(len(self.acciones)))
        valores_q = self.q[estado]
        pareto_optimo = []

        for i in range(len(valores_q)):
            pareto = True
            for j in range(len(valores_q)):
                if i == j:
                    continue
                if all(valores_q[j] >= valores_q[i]) and any(valores_q[j] > valores_q[i]):
                    pareto = False
                    break
            if pareto:
                pareto_optimo.append(i)
        return pareto_optimo if pareto_optimo else list(range(len(self.acciones)))

    def seleccionar_mejor_accion(self, puntos):
        num_obj = puntos.shape[1]
        num_puntos = puntos.shape[0]
        dist = np.zeros(num_puntos)
        for m in range(num_obj):
            orden = np.argsort(puntos[:, m])
            dist[orden[0]] = dist[orden[-1]] = float('inf')
            min_val, max_val = puntos[orden[0], m], puntos[orden[-1], m]
            if max_val - min_val == 0:
                continue
            for i in range(1, num_puntos - 1):
                dist[orden[i]] += (puntos[orden[i+1], m] - puntos[orden[i-1], m]) / (max_val - min_val)
        return dist

    def calcular_hipervolumen(self, puntos, ref):
        dominated = np.array(puntos)
        dominated = np.clip(dominated, None, ref)
        sorted_points = dominated[dominated[:, 0].argsort()[::-1]]
        hv = 0.0
        prev = ref[1]
        for point in sorted_points:
            width = ref[0] - point[0]
            height = prev - point[1]
            area = width * height
            hv += area
            prev = point[1]
        return hv

    def accion(self,state, exploracion=True):
        estado = self.parse_observacion(state)
        if exploracion and random.random() < self.epsilon:
            return random.choice(self.acciones)
        else:
            indices = self.soluciones_pareto_optimas(estado)
            acciones = [self.acciones[i] for i in indices]
            valores = np.array([self.q[estado][i] for i in indices])
            crowd = self.seleccionar_mejor_accion(valores)
            max_crowd = np.max(crowd)
            indices_max = [i for i, c in enumerate(crowd) if c == max_crowd]
            return acciones[random.choice(indices_max)]

    def actualizar_frente_pareto(self, estado):
        if estado not in self.q:
            return
        q = self.q[estado]
        frente = []
        for i in range(len(q)):
            pareto = True
            for j in range(len(q)):
                if i == j:
                    continue
                if all(q[j] >= q[i]) and any(q[j] > q[i]):
                    pareto = False
                    break
            if pareto:
                frente.append((self.acciones[i], q[i]))
        self.frente_pareto[estado] = frente

        if frente:
            puntos = np.array([v[:2] for _, v in frente])
            ref = np.max(puntos, axis=0) + 1
            self.hv_por_estado[estado] = self.calcular_hipervolumen(puntos, ref)

    def actualizarQ(self, state, action, recompensa, next_state, done):
        estado = self.parse_observacion(state)
        sig_estado = self.parse_observacion(next_state)
        accion = self.acciones[action]
        q = self.q[estado][accion].copy()

        if done:
            target_q = np.array(recompensa)
        else:
            indices = self.soluciones_pareto_optimas(sig_estado)
            sig_q = self.q[sig_estado][indices]
            sig_max_q = np.max(sig_q, axis=0)
            target_q = np.array(recompensa) + self.gamma * sig_max_q

        self.q[estado][accion] = (1 - self.alpha) * q + self.alpha * target_q
        self.actualizar_frente_pareto(estado)

    def mejor_accion(self,state):
        estado = self.parse_observacion(state)
        if estado not in self.frente_pareto or not self.frente_pareto[estado]:
            return random.choice(self.acciones)
        acciones = [a for a, _ in self.frente_pareto[estado]]
        valores = np.array([v for _, v in self.frente_pareto[estado]])
        crowd = self.seleccionar_mejor_accion(valores)
        max_crowd = np.max(crowd)
        indices_max = [i for i, c in enumerate(crowd) if c == max_crowd]
        return acciones[random.choice(indices_max)]
