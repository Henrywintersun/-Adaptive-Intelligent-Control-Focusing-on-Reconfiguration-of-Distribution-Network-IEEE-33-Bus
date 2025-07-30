import numpy as np
import pandas as pd
import gym
import torch
import torch.nn as nn
import cvxpy as cp
from pysindy import SINDy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
from gym import spaces

# BIBC/BIBV Power Flow Calculation
def calculate_bibc_bibv(network_data, power_injections):
    sending_bus = network_data['Sending Bus'].values
    receiving_bus = network_data['Receiving Bus'].values
    resistance = network_data['R (Ω)'].values
    reactance = network_data['X (Ω)'].values
    num_buses = max(max(sending_bus), max(receiving_bus))

    # Convert resistance and reactance to per unit system
    V_base = 12.66  # kV
    S_base = 100  # MVA
    Z_base = (V_base ** 2) / S_base
    resistance_pu = resistance / Z_base
    reactance_pu = reactance / Z_base

    # Create BIBC and BIBV matrices
    BIBC = np.zeros((num_buses, num_buses))
    BIBV = np.zeros((num_buses, num_buses), dtype=complex)

    for i in range(len(sending_bus)):
        BIBC[receiving_bus[i] - 1][sending_bus[i] - 1] = 1
        impedance = resistance_pu[i] + 1j * reactance_pu[i]
        BIBV[receiving_bus[i] - 1][sending_bus[i] - 1] = impedance

    # Convert real and reactive power injections to complex power (S = P + jQ)
    real_power_injections = network_data['P (kW)'].values / (S_base * 1000)  # Convert to per unit
    reactive_power_injections = network_data['Q (kVAR)'].values / (S_base * 1000)  # Convert to per unit
    complex_power_injections = real_power_injections + 1j * reactive_power_injections

    # Extend power injections to cover all buses
    extended_real_power = np.zeros(num_buses)
    extended_reactive_power = np.zeros(num_buses)
    extended_real_power[:len(real_power_injections)] = real_power_injections
    extended_reactive_power[:len(reactive_power_injections)] = reactive_power_injections
    extended_complex_power = extended_real_power + 1j * extended_reactive_power

    # Recalculate branch currents and bus voltages
    branch_currents = BIBC @ extended_complex_power
    bus_voltages = BIBV @ extended_complex_power

    return branch_currents, bus_voltages

# SINDy for discovering network dynamics
def apply_sindy(data):
    model = SINDy()
    model.fit(data)
    return model

# Lagrangian Neural Network (LNN)
class LNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LNN, self).__init__()
        self.l1 = nn.Linear(input_dim * 2, hidden_dim)  # Adjusted input size due to concatenation of q and q_dot
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, input_dim)

    def forward(self, q, q_dot):
        # Concatenate state (q) and velocity (q_dot)
        x = torch.cat([q, q_dot], dim=0)  # Concatenating along the first dimension (dim=0)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return self.out(x)  # Predict next state

# MPC Controller
class MPCController:
    def __init__(self, horizon, grid_size):
        self.horizon = horizon
        self.grid_size = grid_size

    def solve_mpc(self, q, q_dot, power_injections):
        # Separate real parts or use magnitude for optimization
        q_real = np.real(q)
        q_dot_real = np.real(q_dot)

        # MPC Optimization Problem (using real part or magnitude)
        u = cp.Variable((self.horizon, self.grid_size))
        constraints = [0.95 <= q_real + u[0], q_real + u[0] <= 1.05]  # Voltage constraints (real part)
        cost = cp.norm(q_real + u[0] - 1) + cp.norm(q_dot_real + u[0])

        # Solve optimization
        objective = cp.Minimize(cost)
        problem = cp.Problem(objective, constraints)
        problem.solve()

        return u.value[0]

# PPO Custom Environment for Grid Reconfiguration
class PowerGridEnv(gym.Env):
    def __init__(self, network_data, power_injections, sindy_model, lnn):
        super(PowerGridEnv, self).__init__()
        self.network_data = network_data
        self.power_injections = power_injections
        self.sindy_model = sindy_model  # Discovered dynamics from SINDy
        self.lnn = lnn  # Lagrangian Neural Network
        self.mpc = MPCController(horizon=10, grid_size=len(power_injections))  # MPC Controller

        # Calculate the initial state of the grid
        self.q, self.q_dot = self._calculate_initial_state()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(2)  # Example: 2 actions (open/close switch)
        
        # Observation space: Concatenated bus voltages and branch currents
        num_states = len(self.q) + len(self.q_dot)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_states,), dtype=np.float32)

    def _calculate_initial_state(self):
        # Initial BIBC/BIBV power flow calculation
        branch_currents, bus_voltages = calculate_bibc_bibv(self.network_data, self.power_injections)
        return bus_voltages, branch_currents

    def step(self, action):
        # Update grid state using BIBC/BIBV
        self.q, self.q_dot = self._calculate_initial_state()

        # Predict future state using LNN and optimize with MPC
        lnn_prediction = self.lnn.forward(torch.tensor(self.q, dtype=torch.float32), torch.tensor(self.q_dot, dtype=torch.float32))
        mpc_action = self.mpc.solve_mpc(self.q, self.q_dot, self.power_injections)

        # Calculate the reward
        reward = self.calculate_reward(action, mpc_action)
        done = False  # Termination logic (can be improved)
        return np.concatenate((self.q, self.q_dot)), reward, done, {}

    def reset(self):
        # Reset grid state
        self.q, self.q_dot = self._calculate_initial_state()
        return np.concatenate((self.q, self.q_dot))

    def calculate_reward(self, action, mpc_action):
        # Reward function based on multiple objectives
        voltage_penalty = np.abs(self.q - 1).sum()
        power_loss_penalty = np.abs(self.q_dot).sum()
        load_balance_penalty = np.std(self.q)
        return -(0.7 * voltage_penalty + 0.2 * power_loss_penalty + 0.1 * load_balance_penalty)

# Load the new network data for the 30-bus system
network_data = pd.read_csv('30_bus_EDN_network_data_fixed.csv')

# Placeholder for power injections
power_injections = np.random.rand(30)  # Assuming 30 buses, adjust based on actual data

# Apply SINDy to discover dynamics
sindy_model = apply_sindy(power_injections)

# Create LNN model
lnn = LNN(input_dim=30, hidden_dim=64)  # Adjust input_dim for the number of buses (30 in this case)

# Create the environment and train the PPO agent
env = DummyVecEnv([lambda: PowerGridEnv(network_data, power_injections, sindy_model, lnn)])
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Test the trained agent and collect rewards
rewards = []
obs = env.reset()
for _ in range(100):  # Test for 100 steps
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    rewards.append(reward)

# Visualization Functions

# Plot discovered dynamics by SINDy
def plot_sindy_dynamics(sindy_model, data):
    print("SINDy discovered dynamics:")
    sindy_model.print()

# Plot power flow (bus voltages and branch currents)
def plot_power_flow(branch_currents, bus_voltages):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(np.abs(bus_voltages), label='Bus Voltages')
    plt.title('Bus Voltages')
    plt.subplot(2, 1, 2)
    plt.plot(np.abs(branch_currents), label='Branch Currents')

   # Plot improvement in rewards over time
def plot_rewards(rewards):
    plt.figure()
    plt.plot(rewards)
    plt.title("Rewards over Time")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.show()

# Plot the results
branch_currents, bus_voltages = calculate_bibc_bibv(network_data, power_injections)
plot_sindy_dynamics(sindy_model, power_injections)
plot_power_flow(branch_currents, bus_voltages)
plot_rewards(rewards)


