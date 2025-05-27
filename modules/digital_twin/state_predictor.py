import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

class DigitalTwinPredictor:
    def __init__(self):
        self.energy_model = RandomForestRegressor(n_estimators=50)
        self.delay_model = LinearRegression()
        self.queue_model = LinearRegression()
        self.trained = False

    def train(self, df):
        """
        Trains the DT predictor from environment logs.
        Assumes df contains columns: ['rssi', 'cpu_load', 'task_size', 'queue_length', 'delay', 'energy']
        """
        # Drop rows with NaNs
        df = df.dropna()

        # Features and targets
        X = df[['rssi', 'cpu_load', 'task_size', 'queue_length']]
        y_delay = df['delay']
        y_energy = df['energy']
        y_queue = df['queue_length'].shift(-1).fillna(method='ffill')  # predict next queue length

        # Train models
        self.delay_model.fit(X, y_delay)
        self.energy_model.fit(X, y_energy)
        self.queue_model.fit(X, y_queue)

        self.trained = True
        print("✅ Digital Twin models trained.")

    def predict(self, rssi, cpu_load, task_size, queue_length):
        if not self.trained:
            raise ValueError("Digital Twin models not trained yet!")

        x = np.array([[rssi, cpu_load, task_size, queue_length]])
        delay = self.delay_model.predict(x)[0]
        energy = self.energy_model.predict(x)[0]
        next_queue = self.queue_model.predict(x)[0]

        return {
            "predicted_delay": round(delay, 4),
            "predicted_energy": round(energy, 4),
            "predicted_queue": round(next_queue, 2)
        }

if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("datasets/env_sensors/processed/env_normalized.csv")
    print("📋 Original Columns:", df.columns.tolist())

    # Rename columns to match what DT predictor expects
    df = df.rename(columns={
        '0': 'rssi',
        '1': 'cpu_load',
        '2': 'task_size',
        '3': 'queue_length'
    })

    # Add dummy delay and energy values for testing
    df['delay'] = 0.05 + df['rssi'] * 0.1 + df['cpu_load'] * 0.1  # synthetic
    df['energy'] = 0.02 + df['task_size'] * 0.2 + df['queue_length'] * 0.1

    dt = DigitalTwinPredictor()
    dt.train(df)

    # Run one prediction
    result = dt.predict(rssi=0.5, cpu_load=0.3, task_size=0.2, queue_length=0.1)
    print("🔮 Prediction:", result)