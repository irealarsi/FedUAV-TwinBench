
import numpy as np

def priority_aware_greedy(devices, uavs, max_capacity=5):
    """
    Assigns devices to UAVs using priority-aware greedy logic.

    Args:
        devices (list of dict): [{'id': int, 'priority': float, 'location': (x, y)}]
        uavs (list of dict): [{'id': int, 'location': (x, y), 'assigned': []}]
        max_capacity (int): Maximum number of devices a UAV can handle.

    Returns:
        dict: UAV ID → List of assigned device IDs
    """
    assignment = {uav['id']: [] for uav in uavs}

    # Sort devices by priority (high to low)
    sorted_devices = sorted(devices, key=lambda x: -x['priority'])

    for device in sorted_devices:
        best_uav = None
        best_distance = float('inf')

        for uav in uavs:
            if len(assignment[uav['id']]) >= max_capacity:
                continue

            dist = np.linalg.norm(np.array(device['location']) - np.array(uav['location']))
            if dist < best_distance:
                best_distance = dist
                best_uav = uav

        if best_uav:
            assignment[best_uav['id']].append(device['id'])

    return assignment


# Example usage
if __name__ == "__main__":
    devices = [
        {'id': 1, 'priority': 0.9, 'location': (2, 3)},
        {'id': 2, 'priority': 0.6, 'location': (5, 6)},
        {'id': 3, 'priority': 0.8, 'location': (1, 2)}
    ]
    uavs = [
        {'id': 'UAV1', 'location': (0, 0)},
        {'id': 'UAV2', 'location': (6, 6)}
    ]
    result = priority_aware_greedy(devices, uavs)
    print("UAV Assignments:", result)
