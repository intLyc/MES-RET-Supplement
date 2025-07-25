import json
import os
import numpy as np
import scipy.io as sio

def convert_json_to_mat(json_dir='ppo_logs', output_file='ppo_results.mat'):
    """
    将PPO训练结果JSON文件转换为MATLAB可读取的.mat文件
    
    参数:
        json_dir: 存储JSON文件的目录
        output_file: 输出的.mat文件名
    """
    
    generation = 500
    ind_runs = 10
    
    # 获取所有JSON文件
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    # 按任务ID排序
    json_files.sort(key=lambda x: int(x.split('_')[0][1:]))
    
    # 初始化结果字典和列表
    results_dict = {}
    all_tasks_list = []
    
    for json_file in json_files:
        # 解析任务信息
        task_id = int(json_file.split('_')[0][1:])
        env_name = json_file.split('_')[1]
        print(f"Processing task {task_id}, environment {env_name}")
        
        with open(os.path.join(json_dir, json_file), 'r') as f:
            data = json.load(f)
        
        # 检查是否有足够的运行数据
        if len(data['all_rewards']) < ind_runs:
            print(f"Warning: {env_name} has only {len(data['all_rewards'])} runs, skipping")
            continue
        
        # 提取所有运行的数据
        runs_data = []
        for run in data['all_rewards'][:ind_runs]:  # 只取前10次运行
            # 将null转换为NaN
            processed_run = [np.nan if x is None else x for x in run]
            
            if len(processed_run) < generation:
                # 如果数据不足generation个episode，用最后一个值填充
                padded_run = processed_run + [processed_run[-1]] * (generation - len(processed_run))
                runs_data.append(padded_run[:generation])
            else:
                runs_data.append(processed_run[:generation])
        
        # 转换为numpy数组 (10x2generation)
        env_matrix = np.array(runs_data)
        
        # 添加到结果字典
        key = f'task_{task_id}_{env_name}'
        results_dict[key] = env_matrix
        # 直接添加到列表，保持原始顺序
        all_tasks_list.append(env_matrix)
    
    # 将所有任务的数据合并为一个大的3D数组 (num_tasks x 10 x 2generation)
    if all_tasks_list:  # 确保列表不为空
        all_tasks_matrix = np.stack(all_tasks_list)
    else:
        all_tasks_matrix = np.array([])
    
    # 保存为.mat文件
    sio.savemat(output_file, {
        'all_tasks': all_tasks_matrix,
        **results_dict  # 同时保存各个任务的单独矩阵
    })
    
    print(f"Successfully saved results to {output_file}")
    print(f"Total tasks processed: {len(json_files)}")
    if all_tasks_list:
        print(f"Matrix dimensions: {all_tasks_matrix.shape}")
    else:
        print("No valid data to save")

if __name__ == "__main__":
    convert_json_to_mat(json_dir='ppo_logs', output_file='ppo_results.mat')
    convert_json_to_mat(json_dir='a2c_logs', output_file='a2c_results.mat')