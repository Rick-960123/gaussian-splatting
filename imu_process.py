import numpy as np

class IMUPreintegration:
    def __init__(self, acc_bias, gyro_bias, acc_noise, gyro_noise):
        self.acc_bias = acc_bias
        self.gyro_bias = gyro_bias
        self.acc_noise = acc_noise
        self.gyro_noise = gyro_noise
        
        # Preintegrated values
        self.delta_p = np.zeros(3)
        self.delta_v = np.zeros(3)
        self.delta_R = np.eye(3)
        
        # Jacobians
        self.J_p_ba = np.zeros((3, 3))
        self.J_p_bg = np.zeros((3, 3))
        self.J_v_ba = np.zeros((3, 3))
        self.J_v_bg = np.zeros((3, 3))
        self.J_R_bg = np.zeros((3, 3))
        
        # Noise covariance
        self.P = np.zeros((9, 9))  # Covariance matrix
        self.imu_inited = False
        self.acc_list = []
        self.gyro_list = []
        
        self.gravity = np.array([0, 0, 9.81])
        self.gravity_norm = np.linalg.norm(self.gravity)
        self.acc_threshold = 0.1
        self.gyro_threshold = 0.01
        self.imu_num = 200

    def is_stationary(self, acc, gyro):
        acc_norm = np.linalg.norm(acc)
        is_acc_stationary = np.abs(acc_norm - 9.81) < self.acc_threshold
        gyro_norm = np.linalg.norm(gyro)
        is_gyro_stationary = gyro_norm < self.gyro_threshold
        return is_acc_stationary and is_gyro_stationary
    
    def initImu(self, acc, gyro):
        if self.is_stationary(acc, gyro):
            self.acc_list.append(acc)
            self.gyro_list.append(gyro)
            if(len(self.acc_list) == self.imu_num):
                acc_sum = np.zeros((3, 1))
                gyro_sum = np.zeros((3, 1))
                for acc_, gyro in  self.acc_list, self.gyro_list:
                    acc_sum += acc_
                    gyro_sum += gyro_sum
                acc_mean = acc_sum / self.imu_num
                gyro_mean = gyro_sum / self.imu_num

                self.gravity = (acc_mean / np.linalg.norm(acc_mean)) * self.gravity_norm
                self.acc_bias = acc_mean - self.gravity
                self.gyro_bias = gyro_mean
                self.initImu = True
        else:
            self.acc_list = []
            self.gyro_list = []

    def integrate(self, dt, acc, gyro):
        if not self.imu_inited:
            self.initImu(acc, gyro)

        acc_unbiased = acc - self.acc_bias
        gyro_unbiased = gyro - self.gyro_bias
        
        theta = gyro_unbiased * dt
        theta_norm = np.linalg.norm(theta)
        
        if theta_norm > 1e-5:
            theta_cross = np.array([[0, -theta[2], theta[1]],
                                    [theta[2], 0, -theta[0]],
                                    [-theta[1], theta[0], 0]])
            self.delta_R = self.delta_R @ (np.eye(3) + np.sin(theta_norm) / theta_norm * theta_cross +
                                           (1 - np.cos(theta_norm)) / theta_norm**2 * theta_cross @ theta_cross)
        
        self.delta_v += self.delta_R @ acc_unbiased * dt
        
        self.delta_p += self.delta_v * dt + 0.5 * self.delta_R @ acc_unbiased * dt**2
        
        acc_cross = np.array([[0, -acc_unbiased[2], acc_unbiased[1]],
                              [acc_unbiased[2], 0, -acc_unbiased[0]],
                              [-acc_unbiased[1], acc_unbiased[0], 0]])
        
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:9] = -self.delta_R @ acc_cross * dt
        F[6:9, 6:9] = np.eye(3) - np.linalg.norm(theta) * dt / 2 * theta_cross
        
        V = np.zeros((9, 6))
        V[3:6, 0:3] = self.delta_R * dt
        V[6:9, 3:6] = np.eye(3) * dt

        Q = np.zeros((6, 6))
        Q[0:3, 0:3] = self.acc_noise**2 * dt**2 * np.eye(3)
        Q[3:6, 3:6] = self.gyro_noise**2 * dt**2 * np.eye(3)
        
        self.P = F @ self.P @ F.T + V @ Q @ V.T
        
    def update_biases(self, new_acc_bias, new_gyro_bias):
        # Calculate the difference in biases
        delta_acc_bias = new_acc_bias - self.acc_bias
        delta_gyro_bias = new_gyro_bias - self.gyro_bias
        
        # Update the biases
        self.acc_bias = new_acc_bias
        self.gyro_bias = new_gyro_bias
        
        # Update the preintegrated values using the Jacobians
        self.delta_p += self.J_p_ba @ delta_acc_bias + self.J_p_bg @ delta_gyro_bias
        self.delta_v += self.J_v_ba @ delta_acc_bias + self.J_v_bg @ delta_gyro_bias
        self.delta_R = self.delta_R @ self.so3_exp(self.J_R_bg @ delta_gyro_bias)
    
    def so3_exp(self, omega):
        theta = np.linalg.norm(omega)
        if theta < 1e-5:
            return np.eye(3)
        axis = omega / theta
        axis_cross = np.array([[0, -axis[2], axis[1]],
                               [axis[2], 0, -axis[0]],
                               [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(theta) * axis_cross + (1 - np.cos(theta)) * axis_cross @ axis_cross


# 示例用法
acc_bias = np.array([0.1, 0.1, 0.1])
gyro_bias = np.array([0.01, 0.01, 0.01])
acc_noise = 0.02
gyro_noise = 0.001

imu_preint = IMUPreintegration(acc_bias, gyro_bias, acc_noise, gyro_noise)

dt = 0.01
acc = np.array([0.0, 0.0, 9.81])
gyro = np.array([0.0, 0.0, 0.5])

for _ in range(100):
    imu_preint.integrate(dt, acc, gyro)

print(f"Delta position: {imu_preint.delta_p}")
print(f"Delta velocity: {imu_preint.delta_v}")
print(f"Delta rotation: \n{imu_preint.delta_R}")

# Update biases and recompute
new_acc_bias = np.array([0.05, 0.05, 0.05])
new_gyro_bias = np.array([0.005, 0.005, 0.005])
imu_preint.update_biases(new_acc_bias, new_gyro_bias)

print(f"Delta position: {imu_preint.delta_p}")
print(f"Delta velocity: {imu_preint.delta_v}")
print(f"Delta rotation: \n{imu_preint.delta_R}")
