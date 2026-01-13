#!/usr/bin/env python3
"""
ROS 2 узел для управления H1 роботом с inference модели Isaac Gym
"""

import rclpy
from rclpy.node import Node
import threading
import time

# ROS 2 сообщения
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, Header

# Импорты из твоего рабочего кода
import math
import numpy as np
import mujoco
import mujoco.viewer
from collections import deque
from scipy.spatial.transform import Rotation as R
import torch
import sys
import os

# Добавляем путь к твоим модулям
sys.path.append(os.path.expanduser('~/RL/Alpha_Human_gym-main'))
from global_config import ROOT_DIR, SPD_X, SPD_Y, SPD_YAW
from configs.h1_constraint_him_trot import H1ConstraintHimRoughCfg

class H1ControllerNode(Node):
    def __init__(self):
        super().__init__('h1_controller')
        
        # Параметры ROS 2
        self.declare_parameters(
            namespace='',
            parameters=[
                ('model_path', 'scripts/model/trot_jitt.pt'),
                ('mujoco_model_path', f'{ROOT_DIR}/resources/h1/xml/world.xml'),
                ('control_frequency', 100.0),
                ('render', True),
                ('spd_x', 0.5),
                ('spd_y', 0.0),
                ('spd_yaw', 0.0),
            ]
        )
        
        # Загружаем модель
        model_path = self.get_parameter('model_path').value
        self.policy = torch.jit.load(model_path, map_location='cpu')
        self.policy.eval()
        
        # Инициализация MuJoCo
        mujoco_model_path = self.get_parameter('mujoco_model_path').value
        self.model = mujoco.MjModel.from_xml_path(mujoco_model_path)
        self.model.opt.timestep = 0.001
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)
        
        # ROS 2 паблишеры
        self.joint_state_pub = self.create_publisher(JointState, '/h1/joint_states', 10)
        self.imu_pub = self.create_publisher(Imu, '/h1/imu', 10)
        self.odom_pub = self.create_publisher(Odometry, '/h1/odom', 10)
        
        # ROS 2 сабскрайберы
        self.cmd_sub = self.create_subscription(Twist, '/h1/cmd_vel', self.cmd_callback, 10)
        
        # Параметры управления
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_dyaw = 0.0
        
        # Инициализация симуляции
        self.init_simulation()
        
        # Запуск потоков
        self.is_running = True
        self.sim_thread = threading.Thread(target=self.simulation_loop)
        self.sim_thread.start()
        
        # Таймер для публикации ROS 2 сообщений
        self.timer = self.create_timer(0.01, self.publish_state)  # 100 Hz
        
        self.get_logger().info("✅ H1 Controller запущен")
    
    def init_simulation(self):
        """Инициализация симуляции (адаптировано из твоего кода)"""
        self.default_dof_pos = np.array([0.0, 0.08, 0.56, -1.12, -0.57, 
                                          0.0, -0.08, -0.56, 1.12, 0.57], dtype=np.float32)
        
        # История наблюдений
        self.hist_obs = deque()
        for _ in range(10):  # history_len = 10
            self.hist_obs.append(np.zeros([1, 39], dtype=np.float16))
        
        # Параметры управления
        self.kp = np.array([13, 15, 15, 15, 13, 13, 15, 15, 15, 13], dtype=np.float32)
        self.kd = np.array([0.3, 0.65, 0.65, 0.65, 0.3, 0.3, 0.65, 0.65, 0.65, 0.3], dtype=np.float32)
        self.tau_limit = 20.0
        
        # Состояние управления
        self.last_actions = np.zeros(10, dtype=np.float32)
        self.count_lowlevel = 0
    
    def simulation_loop(self):
        """Основной цикл симуляции"""
        render = self.get_parameter('render').value
        
        if render:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                while self.is_running and viewer.is_running():
                    self.step_simulation()
                    viewer.sync()
                    time.sleep(0.001)  # 1 kHz
        else:
            while self.is_running:
                self.step_simulation()
                time.sleep(0.001)
    
    def step_simulation(self):
        """Один шаг симуляции (адаптировано из твоего кода)"""
        # Получаем наблюдения из MuJoCo
        q, dq, quat, v, omega, gvec = self.get_obs(self.data)
        q = q[-10:]  # 10 суставов
        dq = dq[-10:]
        
        # Каждые 20 шагов (100 Hz) делаем inference
        if self.count_lowlevel % 20 == 0:
            # Подготавливаем наблюдения
            obs = self.prepare_observation(q, dq, quat, omega)
            
            # Обновляем историю
            self.hist_obs.append(obs)
            self.hist_obs.popleft()
            
            # Inference модели
            action = self.run_inference(obs)
            
            # PD контроль
            target_q = action * 0.25 + self.default_dof_pos
            target_dq = np.zeros(10, dtype=np.float32)
            
            tau = (target_q - q) * self.kp + (target_dq - dq) * self.kd
            tau = np.clip(tau, -self.tau_limit, self.tau_limit)
            
            # Применяем управление
            self.data.ctrl = tau
        
        # Шаг симуляции
        mujoco.mj_step(self.model, self.data)
        self.count_lowlevel += 1
    
    def prepare_observation(self, q, dq, quat, omega):
        """Подготовка наблюдений для модели"""
        obs = np.zeros([1, 39], dtype=np.float16)
        
        # Углы Эйлера
        eu_ang = self.quaternion_to_euler_array(quat)
        eu_ang[eu_ang > math.pi] -= 2 * math.pi
        
        # Заполняем наблюдения
        obs[0, 0] = omega[0] * 0.25  # obs_scales.ang_vel
        obs[0, 1] = omega[1] * 0.25
        obs[0, 2] = omega[2] * 0.25
        obs[0, 3] = eu_ang[0] * 1.0  # obs_scales.quat
        obs[0, 4] = eu_ang[1] * 1.0
        obs[0, 5] = eu_ang[2] * 1.0
        obs[0, 6] = self.cmd_vx * 2.0  # obs_scales.lin_vel
        obs[0, 7] = self.cmd_vy * 2.0
        obs[0, 8] = self.cmd_dyaw * 0.25
        obs[0, 9:19] = (q - self.default_dof_pos) * 1.0  # obs_scales.dof_pos
        obs[0, 19:29] = dq * 0.05  # obs_scales.dof_vel
        obs[0, 29:39] = self.last_actions
        
        return obs
    
    def run_inference(self, obs):
        """Выполнение inference модели"""
        # Текущие наблюдения
        obs_tensor = torch.tensor(obs).half()
        
        # История наблюдений
        hist_obs_3d = np.zeros([1, 10, 39], dtype=np.float16)
        for i in range(10):
            hist_obs_3d[0, i, :] = self.hist_obs[i][0, :]
        
        obs_hist_tensor = torch.tensor(hist_obs_3d).half()
        
        # Inference
        with torch.no_grad():
            action_tensor = self.policy(obs_tensor, obs_hist_tensor)
            
            if isinstance(action_tensor, tuple):
                action_tensor = action_tensor[0]
            
            action = action_tensor[0].float().detach().numpy()
            self.last_actions = action
        
        return action
    
    def quaternion_to_euler_array(self, quat):
        """Конвертация кватерниона в углы Эйлера"""
        x, y, z, w = quat
        
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)
        
        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch_y = np.arcsin(t2)
        
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)
        
        return np.array([roll_x, pitch_y, yaw_z])
    
    def get_obs(self, data):
        """Получение наблюдений из MuJoCo"""
        q = data.qpos.astype(np.float32)
        dq = data.qvel.astype(np.float32)
        quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.float32)
        r = R.from_quat(quat)
        v = r.apply(data.qvel[:3], inverse=True).astype(np.float32)
        omega = data.sensor('angular-velocity').data.astype(np.float32)
        gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.float32)
        return (q, dq, quat, v, omega, gvec)
    
    def cmd_callback(self, msg):
        """Обработка команд от ROS 2"""
        self.cmd_vx = msg.linear.x
        self.cmd_vy = msg.linear.y
        self.cmd_dyaw = msg.angular.z
        
        self.get_logger().info(f"📥 Команда: vx={self.cmd_vx:.2f}, vy={self.cmd_vy:.2f}, ω={self.cmd_dyaw:.2f}")
    
    def publish_state(self):
        """Публикация состояния робота в ROS 2"""
        try:
            # Joint States
            joint_msg = JointState()
            joint_msg.header = Header()
            joint_msg.header.stamp = self.get_clock().now().to_msg()
            joint_msg.header.frame_id = "base_link"
            
            joint_names = ['J_L0', 'J_L1', 'J_L2', 'J_L3', 'J_L4_ankle',
                          'J_R0', 'J_R1', 'J_R2', 'J_R3', 'J_R4_ankle']
            joint_msg.name = joint_names
            joint_msg.position = self.data.qpos[-10:].tolist()
            joint_msg.velocity = self.data.qvel[-10:].tolist()
            
            self.joint_state_pub.publish(joint_msg)
            
            # IMU Data
            imu_msg = Imu()
            imu_msg.header = joint_msg.header
            
            quat = self.data.sensor('orientation').data[[1, 2, 3, 0]]
            omega = self.data.sensor('angular-velocity').data
            accel = self.data.sensor('linear-acceleration').data
            
            imu_msg.orientation.x = quat[0]
            imu_msg.orientation.y = quat[1]
            imu_msg.orientation.z = quat[2]
            imu_msg.orientation.w = quat[3]
            
            imu_msg.angular_velocity.x = omega[0]
            imu_msg.angular_velocity.y = omega[1]
            imu_msg.angular_velocity.z = omega[2]
            
            imu_msg.linear_acceleration.x = accel[0]
            imu_msg.linear_acceleration.y = accel[1]
            imu_msg.linear_acceleration.z = accel[2]
            
            self.imu_pub.publish(imu_msg)
            
        except Exception as e:
            self.get_logger().error(f"❌ Ошибка публикации: {e}")
    
    def destroy_node(self):
        """Корректное завершение"""
        self.get_logger().info("🛑 Завершение работы H1 Controller")
        self.is_running = False
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = H1ControllerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()