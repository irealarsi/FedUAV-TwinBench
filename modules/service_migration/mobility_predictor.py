
import numpy as np

def predict_next_location(past_locations, steps=1):
    """
    Predict the next UAV location using simple linear trend extrapolation.

    Args:
        past_locations (list of tuples): Recent (x, y) coordinates.
        steps (int): How many steps ahead to predict.

    Returns:
        tuple: Predicted (x, y) coordinate.
    """
    if len(past_locations) < 2:
        raise ValueError("Need at least 2 past locations to predict.")

    x_vals = [loc[0] for loc in past_locations]
    y_vals = [loc[1] for loc in past_locations]

    # Simple delta calculation
    dx = np.mean(np.diff(x_vals))
    dy = np.mean(np.diff(y_vals))

    last_x, last_y = past_locations[-1]
    predicted_x = last_x + dx * steps
    predicted_y = last_y + dy * steps

    return round(predicted_x, 2), round(predicted_y, 2)


# Example usage
if __name__ == "__main__":
    trajectory = [(5, 5), (6, 5.2), (7, 5.4), (8, 5.6)]
    next_pos = predict_next_location(trajectory, steps=1)
    print("Predicted next UAV location:", next_pos)
