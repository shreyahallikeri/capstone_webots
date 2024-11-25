import torch
import numpy as np
import os
from typing import List, Tuple
from collections import deque
from loguru import logger
import random
from datetime import datetime
import pandas as pd
import torch.nn.functional as F
import cv2  # OpenCV for resizing
from mavic import Mavic
from policy_network import PolicyNetwork
from controller import Robot  # type: ignore
from constants import *

# Ensure only one Robot instance is created
try:
    mavic = Mavic(Robot())
    logger.info("Mavic initialized.")
except RuntimeError as e:
    logger.error("Only one Robot instance can be created per controller process.")
    exit(1)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Initialize the policy network
policy_net = PolicyNetwork(
    INPUT_DIM, HIDDEN_DIMS, OUTPUT_DIM, "policy_net_checkpoints"
).to(DEVICE)
policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
logger.info("Policy network initialized.")

# Create directories for checkpoints and reward histories
os.makedirs("policy_net_checkpoints", exist_ok=True)
os.makedirs("reward_histories", exist_ok=True)

# Utility function to clamp values within a specified range
def clamp(value: float, low: float, high: float) -> float:
    return max(min(value, high), low)

# Depth Model
class DepthModel(torch.nn.Module):
    def __init__(self):
        super(DepthModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        
        # Calculate the output size after convolutions for a 64x64 input image
        # After two 3x3 convolutions with stride 1, the output size is reduced
        conv_output_size = (64 - 3 + 1 - 3 + 1)  # Output after conv1 and conv2 with kernel size 3
        self.fc1 = torch.nn.Linear(32 * conv_output_size * conv_output_size, 128)
        self.fc2 = torch.nn.Linear(128, 1)  # For regression tasks (depth estimation)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten before passing to fully connected layers
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Load the pre-trained depth estimation model weights
depth_model = DepthModel().to(DEVICE)

try:
    checkpoint = torch.load("/home/shreya/Downloads/mavic-webots/controllers/policy_net/Depth-Model.pth", map_location=DEVICE, weights_only=True)
    if 'state_dict' in checkpoint:
        depth_model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        depth_model.load_state_dict(checkpoint, strict=False)
    logger.info("Depth model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading DepthModel: {e}")
    exit(1)

# Function to process depth image
def process_depth_image():
    depth_data = mavic.get_depth_data()
    # Resize the depth image to a fixed size (e.g., 64x64) to ensure compatibility with the model
    depth_image_resized = np.array(depth_data)  # Ensure it's a numpy array
    depth_image_resized = cv2.resize(depth_image_resized, (64, 64))  # Resize to 64x64

    # Convert to tensor and add batch and channel dimensions
    depth_image_tensor = torch.tensor(depth_image_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    depth_features = depth_model(depth_image_tensor).cpu().detach().numpy()  # Extract features

    return depth_features.flatten()

# Function to get the state from the MAVIC drone, including depth image data
def get_state():
    # Get the individual components
    imu_values = mavic.get_imu_values()  # Get IMU values (typically 3 values: roll, pitch, yaw)
    gps_values = mavic.get_gps_values()  # Get GPS values (typically 3 values: x, y, z)
    gyro_values = mavic.get_gyro_values()  # Get Gyro values (typically 3 values: roll_rate, pitch_rate, yaw_rate)
    depth_features = process_depth_image()  # Get processed depth image features

    # Concatenate IMU, GPS, Gyro, and depth features
    state_vector = np.concatenate([imu_values, gps_values, gyro_values, depth_features])

    # Log the shape of the state vector for debugging
    logger.info(f"State vector shape: {state_vector.shape}")

    # Since the expected total length is 9, let's just take the first 9 values.
    if state_vector.shape[0] >= 9:
        # Unpack based on the known structure (first 9 values only)
        roll, pitch, yaw = imu_values  # 3 values: roll, pitch, yaw
        x, y, z = gps_values  # 3 values: x, y, z
        roll_rate, pitch_rate, yaw_rate = gyro_values  # 3 values: roll_rate, pitch_rate, yaw_rate
        depth_data = depth_features[:1]  # Use only the first element of depth features if needed

        # Log each component to verify
        logger.info(f"Roll, Pitch, Yaw: {roll}, {pitch}, {yaw}")
        logger.info(f"X, Y, Z: {x}, {y}, {z}")
        logger.info(f"Roll rate, Pitch rate, Yaw rate: {roll_rate}, {pitch_rate}, {yaw_rate}")
        logger.info(f"Depth features length: {len(depth_data)}")

        # Return the first 9 values of the state vector
        return np.concatenate([imu_values, gps_values, gyro_values, depth_data])

    else:
        logger.error(f"State vector length mismatch: expected at least 9 values, but got {state_vector.shape[0]}")
        raise ValueError("State vector length mismatch.")


# PID controller for altitude control
def altitude_PID(target, current, timestep, integral, prev_error):
    error = target - current
    integral += error * timestep
    derivative = (error - prev_error) / timestep
    output = Kp * error + Ki * integral + Kd * derivative
    return output, integral, error

# Reward calculation based on state and action
def calculate_reward(state_buffer: deque, action_buffer: deque) -> Tuple[float, bool]:
    position_error = np.linalg.norm(np.array(DESIRED_STATE) - np.array(state_buffer[1]))
    orientation_error = np.linalg.norm(np.array(DESIRED_STATE[0:4]) - np.array(state_buffer[1][0:4]))
    state_error = position_error + orientation_error
    state_reward = 1 / np.power((state_error + 1e-3), 0.2)

    gps0 = state_buffer[0][3:6]
    gps1 = state_buffer[1][3:6]
    gps_desired = DESIRED_STATE[3:6]

    d_vec = np.array(gps_desired) - np.array(gps0)
    v_vec = np.array(gps1) - np.array(gps0)

    d_dot_v = np.dot(d_vec, v_vec)
    vector_separation = 2 * d_dot_v / ((np.linalg.norm(d_vec) * np.linalg.norm(v_vec)) + 1e-12) - 1

    opposite_actions = {1: 0, 0: 1, 3: 2, 2: 3, 5: 4, 4: 5, 7: 6, 6: 7}
    control_effort_penalty = 0

    for i in range(1, len(action_buffer)):
        if action_buffer[i] == opposite_actions[action_buffer[i - 1]]:
            control_effort_penalty += 1
        else:
            control_effort_penalty -= 1

    control_effort_penalty /= NUM_STEPS - 1

    return float(REWARD_SCALE * (state_reward * vector_separation * control_effort_penalty)), state_error < 0.1

# Compute returns for rewards
def calculate_returns(rewards: List[np.float32]) -> List[np.float32]:
    returns = []
    R = 0
    for r in rewards[::-1]:
        R = r + GAMMA * R
        returns.insert(0, R)
    return returns

# Loss calculation based on log probabilities and rewards
def calculate_loss(log_probs: List[torch.Tensor], returns: List[np.float32]) -> torch.Tensor:
    policy_loss = []
    for log_prob, R in zip(log_probs, returns):
        policy_loss.append(-log_prob * R)
    return torch.stack(policy_loss).sum()

# Main function for running the training or evaluation
def main() -> None:
    if TRAIN_MODE:
        reward_history = []
        best_reward = float("-inf")
        logger.info("Starting in Training mode.")

        policy_net.train()

        for episode in range(NUM_EPISODES):
            mavic.reset()

            done = True
            state_buffer = deque(maxlen=2)
            action_buffer = deque(maxlen=NUM_STEPS)
            action_buffer.append(0)
            state_buffer.append(np.array(INITIAL_STATE))

            log_probs = []
            rewards = []
            episode_reward = 0

            current_step = 0

            altitude_integral_error = 0
            altitude_previous_error = 0

            while mavic.step_robot() != -1 and current_step < NUM_STEPS:
                target_altitude = 1
                roll_disturbance = 0
                pitch_disturbance = 0
                yaw_disturbance = 0

                state_vector = get_state()  # Now includes depth image data
                state_buffer.append(state_vector)

                roll, pitch, yaw, x, y, z, roll_rate, pitch_rate, yaw_rate = state_vector

                if current_step % ERROR_RESET == 0:
                    altitude_integral_error = 0
                    altitude_previous_error = 0

                action_probs = policy_net(
                    torch.tensor(np.concatenate((state_vector, (action_buffer[-1],))), dtype=torch.float32).to(DEVICE)
                )
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                log_probs.append(action_dist.log_prob(action))
                action_buffer.append(action.item())
                action = action.item()

                if action == 0:
                    target_altitude += 0.1
                elif action == 1:
                    target_altitude -= 0.1
                elif action in {2, 3}:
                    roll_disturbance = 1 if action == 2 else -1
                elif action in {4, 5}:
                    pitch_disturbance = 2 if action == 4 else -2
                elif action in {6, 7}:
                    yaw_disturbance = 1.3 if action == 6 else -1.3
                else:
                    raise ValueError(f"Invalid action: {action}. Expected values are 0 to 7.")

                control_effort, altitude_integral_error, altitude_previous_error = altitude_PID(
                    target_altitude, z, mavic.timestep / 1e3, altitude_integral_error, altitude_previous_error
                )

                roll_input = Kroll * np.clip(roll, -1, 1) + roll_rate + roll_disturbance
                pitch_input = Kpitch * np.clip(pitch, -1, 1) + pitch_rate + pitch_disturbance
                yaw_input = yaw_disturbance

                front_left_motor_input = np.clip(
                    MIN_THRUST + control_effort - roll_input + pitch_input - yaw_input, -MAX_THRUST, MAX_THRUST
                )
                front_right_motor_input = np.clip(
                    MIN_THRUST + control_effort + roll_input + pitch_input + yaw_input, -MAX_THRUST, MAX_THRUST
                )
                rear_left_motor_input = np.clip(
                    MIN_THRUST + control_effort - roll_input - pitch_input + yaw_input, -MAX_THRUST, MAX_THRUST
                )
                rear_right_motor_input = np.clip(
                    MIN_THRUST + control_effort + roll_input - pitch_input - yaw_input, -MAX_THRUST, MAX_THRUST
                )

                mavic.set_motor_speeds(front_left_motor_input, front_right_motor_input, rear_left_motor_input, rear_right_motor_input)

                reward, done = calculate_reward(state_buffer, action_buffer)
                rewards.append(reward)
                episode_reward += reward
                if done:
                    break

                current_step += 1

            logger.info(f"Episode {episode + 1}/{NUM_EPISODES}, Reward: {episode_reward}")
            reward_history.append(episode_reward)

            returns = calculate_returns(rewards)

            policy_optimizer.zero_grad()

            loss = calculate_loss(log_probs, returns)
            loss.backward()

            policy_optimizer.step()

            if episode_reward > best_reward:
                best_reward = episode_reward
                policy_net.save_checkpoint(f"policy_net_checkpoints/episode_{episode + 1}")

            if episode % 100 == 0:
                df = pd.DataFrame(reward_history)
                df.to_csv(f"reward_histories/reward_history_{episode + 1}.csv")

    else:
        logger.info("Starting in Evaluation mode.")
        policy_net.eval()
        mavic.reset()

        state_buffer = deque(maxlen=2)
        state_buffer.append(np.array(INITIAL_STATE))

        current_step = 0
        while mavic.step_robot() != -1:
            state_vector = get_state()
            state_buffer.append(state_vector)

            action_probs = policy_net(
                torch.tensor(np.concatenate((state_vector, (action_buffer[-1],))), dtype=torch.float32).to(DEVICE)
            )

            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()

            action = action.item()

            reward, done = calculate_reward(state_buffer, action_buffer)

            if done:
                break

            current_step += 1

if __name__ == "__main__":
    main()

