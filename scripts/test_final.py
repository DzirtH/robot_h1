#!/usr/bin/env python3
import torch
import numpy as np
import os
import sys

sys.path.append('..')
from global_config import ROOT_DIR
from configs.h1_constraint_him_trot import H1ConstraintHimRoughCfg

# Загружаем конфиг
cfg = H1ConstraintHimRoughCfg()

print("Параметры конфига:")
print(f"n_proprio: {cfg.env.n_proprio}")  # Должно быть 39
print(f"num_observations: {cfg.env.num_observations}")  # Должно быть 660

# Пробуем загрузить модель
model_path = os.path.join(ROOT_DIR, 'scripts', 'model', 'trot.pt')
print(f"\nЗагружаем: {model_path}")

if os.path.exists(model_path):
    # Пробуем как готовую модель
    try:
        model = torch.load(model_path, map_location='cpu')
        print(f"✅ Загружено")
        print(f"Тип: {type(model)}")
        
        if isinstance(model, dict):
            print("Это словарь:")
            for key in model.keys():
                print(f"  - {key}: {type(model[key])}")
                
            # Проверяем model_state_dict
            if 'model_state_dict' in model:
                print(f"\nmodel_state_dict первые 5 ключей:")
                for key in list(model['model_state_dict'].keys())[:5]:
                    print(f"  - {key}")
                    
        else:
            print(f"Это объект класса: {model.__class__.__name__}")
            
            # Пробуем inference с разными размерами
            test_sizes = [cfg.env.n_proprio, cfg.env.num_observations]
            
            for size in test_sizes:
                print(f"\n🧪 Тест с размером входа: {size}")
                test_input = torch.randn(1, size)
                
                try:
                    # Пробуем разные методы
                    if hasattr(model, 'act_inference'):
                        output = model.act_inference(test_input)
                        print(f"  ✅ act_inference: {output.shape}")
                    elif hasattr(model, 'act_teacher'):
                        output = model.act_teacher(test_input)
                        print(f"  ✅ act_teacher: {output.shape}")
                    elif hasattr(model, 'forward'):
                        output = model(test_input)
                        print(f"  ✅ forward: {output.shape}")
                    else:
                        print("  ❌ Нет методов inference")
                except Exception as e:
                    print(f"  ❌ Ошибка: {e}")
                    
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
else:
    print("❌ Файл не найден")