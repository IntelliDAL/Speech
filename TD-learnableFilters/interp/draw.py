import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.interpolate import interp1d

# 读取数据文件
with open('/home/idal-01/code/TD-learnableFilters/interp/filter.txt', 'r') as file:
    lines = file.readlines()

cn = []
sigma = []

# 解析数据文件
for line in lines:
    line = line.replace('[', '').replace(']', '')  # 去除方括号
    numbers = line.split()
    # 将列表转换为数组
    array_data = np.array(numbers, dtype=float)
    cn.append(float(array_data[0]))
    sigma.append(float(array_data[1]))
sigma = np.array(sigma, dtype=np.float64)
# 计算高斯滤波器权重
nfilt = 64
sample_rate = 16000  # 采样频率
filter_size = 513  # 滤波器大小
NFFT = 1024
weights = np.ones(filter_size)
# weights = np.zeros(filter_size)  # 权重数组
half_filterLen = (((NFFT // 2) + 1) // nfilt) // 2
for i in range(len(cn)):
    center_Location = cn[i] * filter_size  # 中心频率位置
    center_Location = np.float64(center_Location)  # 将类型转换为 float64
    sigma_value = sigma[i]
    band_low = int(center_Location) - half_filterLen if int(center_Location) - half_filterLen >= 0 else 0
    band_high = (int(center_Location) + half_filterLen + 1) if int(
        center_Location) + half_filterLen < filter_size else filter_size - 1
    t = np.arange(band_low, band_high, dtype=sigma.dtype) / filter_size
    # t = np.arange(filter_size) / filter_size
    numerator = t - cn[i]
    temp = np.exp(-0.5 * (numerator / sigma[i]) ** 2)
    weights[band_low:band_high] *= temp

frequency_resolution = sample_rate / NFFT  # 计算频率分辨率

# 将滤波器位置转换为频率
frequencies = np.arange(filter_size) * frequency_resolution

# 找到权重的最大值和最小值
max_weight = np.max(weights)
min_weight = np.min(weights)

# 归一化权重
normalized_weights = (weights - min_weight) / (max_weight - min_weight)
normalized_weights = weights / max_weight
# normalized_weights = weights
# 绘制高斯滤波器权重图像
plt.figure(figsize=(8, 3), dpi=600)
plt.fill_between(frequencies, normalized_weights, color='skyblue')
plt.plot(frequencies, normalized_weights, color='black')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Normalized Weight')
plt.title('Normalized Gaussian Filter Weights')
plt.show()

# 准备滤波器编号和中心频率数据
filter_numbers = np.arange(len(cn))
center_frequencies = np.array(cn) * sample_rate / 2  # 转换为对应的频率
# 将滤波器的中心频率和滤波器数量按照中心频率进行排序
sorted_data = sorted(center_frequencies)

# # 创建图形并绘制曲线
# plt.plot(sorted_data, filter_numbers)
# plt.ylabel('Filter Number')
# plt.xlabel('Center Frequency (Hz)')
# plt.title('Filter Number vs. Center Frequency')
# plt.show()

# 初始化频率组和滤波器数量列表
freq_groups1 = np.arange(1, 9, 1)
freq_groups2 = ['{}-{} k'.format(start_freq, start_freq + 1) for start_freq in range(0, 8, 1)]
filter_counts = [0] * (len(freq_groups1))

# 划分滤波器到对应的频率组
for center_freq in center_frequencies:
    if center_freq >= 8000:
        center_freq = 7999
    group_index = int(center_freq / 1000)
    if group_index < len(filter_counts):
        filter_counts[group_index] += 1

plt.figure(figsize=(9, 4), dpi=600)
# 创建图形并绘制柱状图
plt.bar(freq_groups2, filter_counts, width=0.5, alpha=0.7, label='Filter Count', color='g')

interpolator = interp1d(range(len(freq_groups2)), filter_counts, kind='cubic')
x = np.linspace(0, len(freq_groups2) - 1, 1000)
smooth_filter_counts = interpolator(x)

plt.plot(x, smooth_filter_counts, marker='o', color='red', linestyle='-', linewidth=0.001, label='Line Plot')

plt.xlabel('Frequency Group')
plt.ylabel('Filter Count')
plt.title('Filter Count in Frequency Groups')
plt.xticks(rotation=45, fontsize=8)
plt.legend()
plt.show()

# cd D:\ProgramData\Anaconda3\Scripts
# .\activate myPytorch
# cd D:\northeastern university\EATD-learnableFilters
# python drawSpectrum.py
