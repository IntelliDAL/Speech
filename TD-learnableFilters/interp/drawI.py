import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.interpolate import interp1d

# 读取数据文件
IorT = 'EI'
filepath = '/home/idal-01/code/TD-learnableFilters/interp/filter.txt'
with open(filepath, 'r') as file:
    lines = file.readlines()


cn = []
sigma = []
"""获取sigma中最小的元素，这些为一个竖线的形状基本就是噪音。为了图像更加直观将其去掉"""


# 解析数据文件
for line in lines:
    line = line.replace('[', '').replace(']', '')  # 去除方括号
    numbers = line.split()
    # 将列表转换为数组
    array_data = np.array(numbers, dtype=float)
    cn.append(float(array_data[0]))
    sigma.append(float(array_data[1]))
sigma = np.array(sigma, dtype=np.float64)

nfilt = 64
sample_rate = 16000 #采样频率
filter_size = 513  # 滤波器大小
NFFT = 1024
weights = np.ones(filter_size)
weights_index = []

half_filterLen = (((NFFT // 2) + 1) // nfilt) // 2

for i in range(len(cn)):
    center_Location = cn[i] * filter_size  # 中心频率位置
    center_Location = np.float64(center_Location)  # 将类型转换为 float64
    sigma_value = sigma[i]
    band_low = int(center_Location) - half_filterLen if int(center_Location) - half_filterLen >= 0 else 0
    band_high = (int(center_Location) + half_filterLen + 1) if int(center_Location) + half_filterLen < filter_size else filter_size - 1
    print(band_high-band_low)
    t = np.arange(band_low, band_high, dtype=sigma.dtype) / filter_size
    weights_index.extend(np.arange(band_low, band_high))
    numerator = t - cn[i]
    temp = np.exp(-0.5 * (numerator / sigma[i]) ** 2)
    # 相乘等于衰减
    weights[band_low:band_high] *= temp

# 滤波器不关注的区域不应该为1，为0
weights_index = np.unique(np.array(weights_index))
"""如果对为的频谱范围没有滤波器关注，则应该为0"""
for index, i in enumerate(weights):
    if index not in weights_index:
        weights[index] = 0.0

frequency_resolution = sample_rate / NFFT  # 计算频率分辨率

# 将滤波器位置转换为频率
frequencies = np.arange(filter_size) * frequency_resolution

# 找到权重的最大值和最小值
max_weight = np.max(weights)
min_weight = np.min(weights)

# 归一化权重
normalized_weights = (weights - min_weight) / (max_weight - min_weight)
# normalized_weights = weights / max_weight
# normalized_weights = weights
# 绘制高斯滤波器权重图像
plt.figure(figsize=(10, 4), dpi=600)
plt.fill_between(frequencies, normalized_weights, color='skyblue')
plt.plot(frequencies, normalized_weights, color='black')
num_ticks = 17  # 设置刻度点数量
x_ticks = np.linspace(frequencies.min(), frequencies.max(), num_ticks)
plt.xticks(x_ticks)  # 设置x轴刻度点
plt.subplots_adjust(bottom=0.15)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Normalized Weight')
plt.title('Normalized Gaussian Filter Weights EncoderI')
path = './figure_I'
plt.savefig(path)
plt.show()
plt.close()

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
freq_groups1 = np.arange(1000, 9000, 1000)
freq_groups2 = ['{}-{}k'.format(start_freq, start_freq + 1) for start_freq in range(0, 8, 1)]
filter_counts = [0] * (len(freq_groups1))



# 划分滤波器到对应的频率组
for center_freq in center_frequencies:
    if center_freq>=8000:
        center_freq=7999
    group_index = int(center_freq / 1000)
    if group_index < len(filter_counts):
        filter_counts[group_index] += 1

plt.figure(figsize=(10, 5), dpi=600)
# 创建图形并绘制柱状图
plt.bar(freq_groups2, filter_counts, alpha=0.7, width=0.3, label='Filter Count')

interpolator = interp1d(range(len(freq_groups2)), filter_counts, kind='cubic')
x = np.linspace(0, len(freq_groups2) - 1, 1000)
smooth_filter_counts = interpolator(x)

plt.plot(x, smooth_filter_counts, marker='o', markersize=2, color='red', linestyle='-', linewidth=0.01, label='Line Plot')

plt.xlabel('Frequency Group(Hz)')
plt.ylabel('Filter Count')
plt.title('Filter Count in Frequency Groups EncoderI')
plt.subplots_adjust(bottom=0.25)
plt.legend()
path = './figure'
plt.savefig(path)
#plt.show()
plt.close()

