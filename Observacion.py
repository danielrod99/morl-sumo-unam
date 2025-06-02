import numpy as np
from gymnasium import spaces
from sumo_rl.environment.observations import ObservationFunction

class ObservacionCopilco(ObservationFunction):
    
    def __init__(self, ts):
        super().__init__(ts)
        
        self.phase_lanes = {
            0: ['ENRIQUEZ_EO_0', 'ENRIQUEZ_EO_1', 'ENRIQUEZ_EO_2', 'ENRIQUEZ_EO_3', 'ASTURIAS_1'],
            1: ['COPILCO_OE_0', 'COPILCO_OE_1', 'COPILCO_OE_2', 'COPILCO_OE_3', 'ASTURIAS_0'],
            2: ['CERRO_DEL_AGUA_NS_0', 'CERRO_DEL_AGUA_NS_1', 'AGUA_SN_0', 'AGUA_SN_1', 'AGUA_SN_2'],
            3: ['CERRO_DEL_AGUA_NS_2', 'AGUA_SN_3', 'ASTURIAS_0']
        }

        self.lane_indices = None
        
    def __call__(self) -> np.ndarray:
        if self.lane_indices is None:
            lanes = list(dict.fromkeys(self.ts.sumo.trafficlight.getControlledLanes(self.ts.id)))
            self.lane_indices = {lane: idx for idx, lane in enumerate(lanes)}
        
        # 1. Fase actual (0-3)
        phase = np.array([self.ts.green_phase], dtype=np.float32)
        
        # 2. Indicador de si se cumplió el tiempo mínimo (0 o 1)
        min_green = np.array([0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1], 
                            dtype=np.float32)
        

        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        
        # 3. Densidad agregada por fase
        phase_density = []
        for phase_num in range(self.ts.num_green_phases):
            lanes_in_phase = self.phase_lanes[phase_num]
            indices = [self.lane_indices[lane] for lane in lanes_in_phase if lane in self.lane_indices]
            if indices:
                phase_density.append(np.mean([density[i] for i in indices]))
            else:
                phase_density.append(0.0)
        
        # 4. Cola agregada por fase
        phase_queue = []
        for phase_num in range(self.ts.num_green_phases):
            lanes_in_phase = self.phase_lanes[phase_num]
            indices = [self.lane_indices[lane] for lane in lanes_in_phase if lane in self.lane_indices]
            if indices:
                phase_queue.append(np.mean([queue[i] for i in indices]))
            else:
                phase_queue.append(0.0)
        
        observation = np.concatenate([
            phase, 
            min_green, 
            np.array(phase_density, dtype=np.float32),
            np.array(phase_queue, dtype=np.float32)
        ])
        return observation
    
    def observation_space(self) -> spaces.Box:
        # 1 valor para fase (0-3), 1 para min_green, num_green_phases para densidad, num_green_phases para cola
        low = np.zeros(2 + 2 * self.ts.num_green_phases, dtype=np.float32)
        high = np.ones(2 + 2 * self.ts.num_green_phases, dtype=np.float32)
        high[0] = self.ts.num_green_phases - 1
        
        return spaces.Box(low=low, high=high, dtype=np.float32)