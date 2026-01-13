#!/usr/bin/env python3
"""
РАБОЧАЯ версия sim2sim.py
"""

import math
import numpy as np
import mujoco
import mujoco.viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
import torch
import sys
import os

# ========== ПРАВИЛЬНЫЕ ИМПОРТЫ ==========
# Добавляем корень проекта в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from global_config import ROOT_DIR, SPD_X, SPD_Y, SPD_YAW, PLAY_DIR
    from configs.h1_constraint_him_trot import H1ConstraintHimRoughCfg
    print(f"✅ Импорты успешны")
    print(f"   ROOT_DIR: {ROOT_DIR}")
    print(f"   PLAY_DIR: {PLAY_DIR}")
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    sys.exit(1)
# ========================================

default_dof_pos = [0.0, 0.08, 0.56, -1.12, -0.57, 0.0, -0.08, -0.56, 1.12, 0.57]

class cmd:
    vx = 0.0
    vy = 0.0
    dyaw = 0.0

def quaternion_to_euler_array(quat):
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

def get_obs(data):
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

def _low_pass_action_filter(actions, last_actions):
    flt = 0.1
    return last_actions * flt + actions * (1 - flt)
    
def run_mujoco(policy, cfg):
    global default_dof_pos
    
    print(f"🔄 Загрузка модели MuJoCo: {cfg.sim_config.mujoco_model_path}")
    
    if not os.path.exists(cfg.sim_config.mujoco_model_path):
        print(f"❌ Файл не найден: {cfg.sim_config.mujoco_model_path}")
        return
    
    try:
        model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
        model.opt.timestep = cfg.sim_config.dt
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        print(f"✅ MuJoCo модель загружена")
    except Exception as e:
        print(f"❌ Ошибка MuJoCo: {e}")
        return
    
    viewer = mujoco.viewer.MujocoViewer(model, data)

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)
    action_flt = np.zeros((cfg.env.num_actions), dtype=np.double)
    last_actions = np.zeros((cfg.env.num_actions), dtype=np.double)
    hist_obs = deque()
    for _ in range(cfg.env.history_len):
        hist_obs.append(np.zeros([1, cfg.env.n_proprio], dtype=np.double))

    count_lowlevel = 0

    print("🚀 Запуск симуляции...")
    print("Нажмите ESC в окне для выхода")
    
    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating"):

        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.env.num_actions:]
        dq = dq[-cfg.env.num_actions:]
        
        if 1:
            if count_lowlevel % cfg.sim_config.decimation == 0:
                obs = np.zeros([1, cfg.env.n_proprio], dtype=np.float32)

                eu_ang = quaternion_to_euler_array(quat)
                eu_ang[eu_ang > math.pi] -= 2 * math.pi

                cmd.vx = SPD_X
                cmd.vy = SPD_Y
                cmd.dyaw = SPD_YAW

                obs[0, 0] = omega[0] * cfg.normalization.obs_scales.ang_vel
                obs[0, 1] = omega[1] * cfg.normalization.obs_scales.ang_vel
                obs[0, 2] = omega[2] * cfg.normalization.obs_scales.ang_vel
                obs[0, 3] = eu_ang[0] * cfg.normalization.obs_scales.quat
                obs[0, 4] = eu_ang[1] * cfg.normalization.obs_scales.quat
                obs[0, 5] = eu_ang[2] * cfg.normalization.obs_scales.quat
                obs[0, 6] = cmd.vx * cfg.normalization.obs_scales.lin_vel
                obs[0, 7] = cmd.vy * cfg.normalization.obs_scales.lin_vel
                obs[0, 8] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
                obs[0, 9:19] = (q - default_dof_pos) * cfg.normalization.obs_scales.dof_pos
                obs[0, 19:29] = dq * cfg.normalization.obs_scales.dof_vel
                obs[0, 29:39] = last_actions
                obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

                hist_obs.append(obs)
                hist_obs.popleft()

                n_proprio = cfg.env.n_proprio
                n_priv_latent = cfg.env.n_priv_latent
                n_scan = cfg.env.n_scan
                history_len = cfg.env.history_len
                num_observations = cfg.env.num_observations

                policy_input = np.zeros([1, num_observations], dtype=np.float16)
                hist_obs_input = np.zeros([1, history_len * n_proprio], dtype=np.float16)
                
                policy_input[0, 0:n_proprio] = obs
                for i in range(n_priv_latent + n_scan):
                    policy_input[0, n_proprio + i] = 0
                for i in range(history_len):
                    policy_input[0, n_proprio + n_priv_latent + n_scan + i * n_proprio : 
                                 n_proprio + n_priv_latent + n_scan + (i + 1) * n_proprio] = hist_obs[i][0, :]
                
                for i in range(history_len):
                    hist_obs_input[0, i * n_proprio : (i + 1) * n_proprio] = hist_obs[i][0, :]
               
                policy = policy.to('cpu')
                
                # Пробуем разные методы inference
                try:
                    if hasattr(policy, 'act_teacher'):
                        action[:] = policy.act_teacher(torch.tensor(policy_input).half())[0].detach().numpy()
                    elif hasattr(policy, '__call__'):
                        action[:] = policy(torch.tensor(policy_input).half())[0].detach().numpy()
                    else:
                        print("⚠️  Использую нулевое действие")
                        action[:] = np.zeros(10)
                except Exception as e:
                    print(f"⚠️  Ошибка inference: {e}")
                    action[:] = np.zeros(10)
  
                action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
                action_flt = _low_pass_action_filter(action, last_actions)
                last_actions = action

                target_q = action_flt * 0.25 + default_dof_pos

            target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
            tau = pd_control(target_q, q, cfg.robot_config.kps,
                             target_dq, dq, cfg.robot_config.kds)
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
            data.ctrl = tau

        mujoco.mj_step(model, data)
        viewer.render()
        
        # Проверяем, не закрыто ли окно
        if not viewer.is_running():
            break
            
        count_lowlevel += 1

    viewer.close()
    print("✅ Симуляция завершена")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, default=PLAY_DIR,
                        help=f'Path to model file (default: {PLAY_DIR})')
    args = parser.parse_args()

    class Sim2simCfg(H1ConstraintHimRoughCfg):
        class sim_config:
            mujoco_model_path = os.path.join(ROOT_DIR, 'resources/h1/xml/world.xml')
            sim_duration = 30.0  # 30 секунд
            dt = 0.001
            decimation = 20

        class robot_config:
            kps = np.array([13, 15, 15, 15, 13, 13, 15, 15, 15, 13], dtype=np.double)
            kds = np.array([0.3, 0.65, 0.65, 0.65, 0.3, 0.3, 0.65, 0.65, 0.65, 0.3], dtype=np.double)
            tau_limit = 20. * np.ones(10, dtype=np.double)

    # Загружаем модель
    print(f"📦 Загрузка модели: {args.load_model}")
    
    if not os.path.exists(args.load_model):
        print(f"❌ Файл не найден: {args.load_model}")
        print(f"Проверь путь в global_config.py: PLAY_DIR = '{PLAY_DIR}'")
        exit(1)
    
    try:
        policy = torch.load(args.load_model, map_location='cpu')
        print(f"✅ Модель загружена")
        print(f"   Тип: {type(policy)}")
        
        # Переводим в режим inference
        if hasattr(policy, 'eval'):
            policy.eval()
            print(f"   Переведена в режим eval")
            
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        exit(1)
    
    # Запускаем симуляцию
    run_mujoco(policy, Sim2simCfg())
