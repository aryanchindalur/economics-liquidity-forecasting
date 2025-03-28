# economics-liquidity-forecasting
# ECO Liquidity Forecasting

This project uses **PPO (Proximal Policy Optimization)** and **LSTM (Long Short-Term Memory)** models to predict Forex market liquidity. 

##  Technologies Used:
- **PPO (Reinforcement Learning)** from `Stable-Baselines3`
- **LSTM (Deep Learning)** using `TensorFlow/Keras`
- `Gym` for custom environments
- `Matplotlib` for visualization

---

##  PPO Model (Reinforcement Learning)
This code trains a PPO agent to predict forex liquidity.

```python
import numpy as np 
import matplotlib.pyplot as plt
import gym
from stable_baselines3 import PPO
from gym import spaces

# Create synthetic liquidity data (simulate forex market conditions)
np.random.seed(42)
time_steps = 500
liquidity = np.cumsum(np.random.randn(time_steps) * 5 + 100)  # Trending liquidity data

# Define a custom gym environment for PPO
class LiquidityEnv(gym.Env):
    def __init__(self, lookback=10):
        super(LiquidityEnv, self).__init__()
        self.lookback = lookback
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(lookback,), dtype=np.float32)
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(1,), dtype=np.float32)  # Reduced range
        self.current_step = lookback  

    def reset(self):
        self.current_step = self.lookback
        return np.array(liquidity[self.current_step - self.lookback:self.current_step], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(liquidity) - 1:
            return self._get_observation(), 0, True, {}

        predicted_liquidity = liquidity[self.current_step - 1] + action[0] * 2
        actual_liquidity = liquidity[self.current_step]
        reward = -abs(predicted_liquidity - actual_liquidity)

        return self._get_observation(), reward, False, {}

    def _get_observation(self):
        return np.array(liquidity[self.current_step - self.lookback:self.current_step], dtype=np.float32)

# Initialize the PPO model
env = LiquidityEnv()
model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=50000)

# Predict future liquidity using PPO
predicted_liquidity = []
obs = env.reset()
for _ in range(100):  
    action, _ = model.predict(obs)
    obs, _, done, _ = env.step(action)
    predicted_liquidity.append(obs[-1])  
    if done:
        break

# Plot actual vs PPO-predicted liquidity
plt.figure(figsize=(12, 6))
plt.plot(range(len(liquidity[-100:])), liquidity[-100:], label="Actual Liquidity (Last 100)", color="blue", linewidth=2)
plt.plot(range(len(predicted_liquidity)), predicted_liquidity, label="PPO Predicted Liquidity", linestyle="dashed", color="red", linewidth=2)
plt.xlabel("Time Steps")
plt.ylabel("Liquidity")
plt.title("Forex Liquidity Prediction using PPO")
plt.legend()
plt.grid()
plt.show()
``` 

##  LSTM Model (Deep Learning)
This code trains an LSTM neural network to predict forex liquidity based on historical data. It learns patterns from past liquidity levels and forecasts future values using sequential modeling.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

np.random.seed(42)

# Generate synthetic forex liquidity data
time_steps = 500
liquidity = np.cumsum(np.random.randn(time_steps)) + 50  

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
liquidity_scaled = scaler.fit_transform(liquidity.reshape(-1, 1))

# Prepare dataset
lookback = 10
X, y = [], []
for i in range(len(liquidity_scaled) - lookback):
    X.append(liquidity_scaled[i:i+lookback])
    y.append(liquidity_scaled[i+lookback])
X, y = np.array(X), np.array(y)

# Split data
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
    LSTM(50, return_sequences=False),
    Dense(25, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(y_test_actual, label="Actual Liquidity", color="blue")
plt.plot(predictions, label="Predicted Liquidity", color="red", linestyle="dashed")
plt.xlabel("Time Steps")
plt.ylabel("Liquidity Level")
plt.title("LSTM Liquidity Prediction in Forex Trading")
plt.legend()
plt.grid()
plt.show()
