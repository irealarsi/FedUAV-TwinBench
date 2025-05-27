import numpy as np

def compute_semantic_fidelity(features, predicted_delay, predicted_energy, object_type):
    """
    Computes semantic fidelity based on:
    - Feature richness (L2 norm of feature vector)
    - Predicted delay (from DT)
    - Predicted energy (from DT)
    - Object type (importance weight)

    Returns a score in [0, 1], where higher is more semantically important.
    """
    # Semantic richness from feature vector
    feature_strength = np.linalg.norm(features) / 100.0  # normalize to [0, ~12]
    feature_strength = min(feature_strength, 1.0)

    # Delay and energy are treated as penalties
    delay_penalty = 1.0 - min(predicted_delay, 1.0)
    energy_penalty = 1.0 - min(predicted_energy, 1.0)

    # Object priority weight (higher is more important)
    priority_map = {
        "person": 1.0,
        "car": 0.9,
        "bicycle": 0.8,
        "animal": 0.7,
        "other": 0.5
    }
    object_weight = priority_map.get(object_type.lower(), 0.5)

    # Final score: weighted harmonic mean
    score = (feature_strength * 0.4 + delay_penalty * 0.3 + energy_penalty * 0.2 + object_weight * 0.1)
    return round(score, 3)


# Example usage
if __name__ == "__main__":
    dummy_vec = np.random.rand(1280)
    fidelity = compute_semantic_fidelity(
        features=dummy_vec,
        predicted_delay=0.18,
        predicted_energy=0.12,
        object_type="person"
    )
    print("Semantic Fidelity Score:", fidelity)
