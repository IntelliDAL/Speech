import numpy as np
import torch

# ckpt_path = '/home/idal-01/code/TD-learnableFilters/self-supervised/pre-trained_model/EncoderI.pt'
ckpt_path = '/home/idal-01/code/TD-learnableFilters/self-supervised/pre-trained_model/0.ckpt'
state_dict = torch.load(ckpt_path)

for param_name, param_value in state_dict.items():
    # if param_name=='AudioSpectral.FD_Learned_Filters.w':
        print('Parameter Name: ', param_name)
        print('Parameter Value:', param_value)
import os

file_path = './filter.txt'

# 检查文件是否存在
if not os.path.isfile(file_path):
    # 如果文件不存在，创建文件所在的目录
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

# 打开文件以写入数据
with open(file_path, 'w') as file:
    # 获取参数值
    param_value = state_dict['AudioSpectral.FD_Learned_Filters.w'].cpu().numpy()
    # 将数组保存到文件
    np.savetxt(file_path, param_value)
