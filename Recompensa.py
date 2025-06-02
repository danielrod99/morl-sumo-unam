def reward_average_speed(traffic_signal):
    return traffic_signal.get_average_speed()

def reward_queue_length(traffic_signal):
    return -traffic_signal.get_total_queued()

def queue_normalizado(traffic_signal): # a menor vehiculos -> 1, a mas vehiculos -> 0 
    max_queue=200
    normalized = 1.0 - min(traffic_signal.get_total_queued() / max_queue, 1.0)
    return normalized

def reward_diff_waiting_time(traffic_signal):
    """Diferencia de tiempo de espera respecto a la última acción."""
    ts_wait = sum(traffic_signal.get_accumulated_waiting_time_per_lane()) / 100.0
    reward = traffic_signal._diff_waiting_time_reward() - ts_wait
    traffic_signal.last_ts_waiting_time = ts_wait
    return reward

def diff_waiting_normalizado(traffic_signal): # a menor tiempo -> 1, a mas tiempo 0
    max_waiting=368

    ts_wait = sum(traffic_signal.get_accumulated_waiting_time_per_lane()) / 100.0
    reward = traffic_signal._diff_waiting_time_reward() - ts_wait
    traffic_signal.last_ts_waiting_time = ts_wait

    normalized = reward / max_waiting
    normalized = max(min(normalized, 0.0), -1.0)
    return normalized


def recompensa(traffic_signal):
    return [reward_average_speed(traffic_signal),queue_normalizado(traffic_signal),diff_waiting_normalizado(traffic_signal)]