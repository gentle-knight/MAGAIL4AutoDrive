import numpy as np
import math

class InverseDynamics:
    def __init__(self, max_steering=0.7, max_acc=6.0, length=4.5):
        """
        :param max_steering: Max steering angle in radians (approx 40 degrees)
        :param max_acc: Max acceleration in m/s^2
        :param length: Vehicle length in meters (Waymo default approx 4.5m)
        """
        self.max_steering = max_steering
        self.max_acc = max_acc
        self.wheelbase = 0.7 * length  # Approximation as per request

    def compute_action(self, current_state, next_state, dt=0.1):
        """
        Compute action [steering, acceleration] from current and next state.
        State format: dictionary or object with keys/attrs: position (x, y), heading, velocity (v_x, v_y)
        or numpy array [x, y, vx, vy, heading]
        
        Using Bicycle Model:
        delta = arctan(L * theta_dot / v)
        acc = (v_next - v_curr) / dt
        """
        
        # Extract state
        # Assume state is dict-like for now, can adapt if needed
        # We need: velocity (scalar), heading
        
        # Helper to get speed
        def get_speed(vel):
            return np.linalg.norm(vel)

        v_curr = get_speed(current_state['velocity'])
        v_next = get_speed(next_state['velocity'])
        
        # 1. Acceleration (longitudinal)
        acc = (v_next - v_curr) / dt
        
        # 2. Steering (lateral)
        # theta_dot = (theta_next - theta_curr) / dt
        theta_curr = current_state['heading']
        theta_next = next_state['heading']
        
        # Handle angle wrapping [-pi, pi]
        diff_theta = theta_next - theta_curr
        if diff_theta > np.pi:
            diff_theta -= 2 * np.pi
        elif diff_theta < -np.pi:
            diff_theta += 2 * np.pi
            
        theta_dot = diff_theta / dt
        
        # Avoid division by zero for stationary vehicles
        if v_curr < 0.1:
            steering = 0.0
        else:
            # delta = arctan(L * theta_dot / v)
            steering = np.arctan(self.wheelbase * theta_dot / v_curr)

        # Normalize actions to [-1, 1]
        norm_acc = np.clip(acc / self.max_acc, -1.0, 1.0)
        norm_steering = np.clip(steering / self.max_steering, -1.0, 1.0)
        
        return np.array([norm_steering, norm_acc]), {'raw_acc': acc, 'raw_steering': steering}
