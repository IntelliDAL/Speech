import glob
import os
import pickle
import random
import wave
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torchaudio
from matplotlib import pyplot as plt
from scipy import signal
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import compute_class_weight
from torch import nn
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler
import csv
import librosa.feature
import librosa.display
import h5py


def find_duplicates(arr):
    unique_elements, counts = np.unique(arr, return_counts=True)
    duplicates = unique_elements[counts > 1]
    return list(duplicates)


def dataprogress(path, excelfile):
    Total_Data, Total_label = [], []
    Total_file, Audio_files = [], []

    # 读取Excel文件
    ExcelData = pd.read_excel(excelfile, sheet_name='mild_moderate')
    # 取出"健康组"和"抑郁组"两列的数据
    data = ExcelData['cust_id'].tolist()
    Group = ExcelData['Group'].tolist()
    #
    age = ExcelData['年龄'].tolist()
    gender = ExcelData['性别'].tolist()
    # depression_level = ExcelData['抑郁等级'].tolist()
    # depression_scores = ExcelData['抑郁分数'].tolist()
    #
    # # 取出"label1"和"label2"列的数据作为标签
    # health_label = ExcelData['健康组抑郁等级（0=无）'].tolist()
    # depression_label = ExcelData['抑郁等级（1=轻度；2=中度；3=重度；4=非常重度）'].tolist()
    #
    # health_score = ExcelData['健康组抑郁分数'].tolist()[0:200]
    # depression_score = ExcelData['抑郁组分数'].tolist()

    # 使用NumPy的where函数获取元素为零的下标
    # zero_indices = np.where(np.array(health_score) == 0)[0]
    # health_data = np.array(health_data)[zero_indices][0:200]
    # health_label = np.array(health_label)[zero_indices][0:200]

    dirList = os.listdir(path)

    # pathList = []cd co
    #
    # for age in ageList:
    #     path_per = os.path.join(path, age)
    #     dirList = os.listdir(path_per)
    #     for file in dirList:
    #         pathList.append(os.path.join(path_per, file))

    # 将数据和标签放入一个列表
    file_list = list(zip(data, Group))
    new_sr = 16000
    Scaler = StandardScaler()
    for idx, item in enumerate(file_list):
        file_name = item[0]
        print(file_name)
        label = item[1]
        for audiofile in dirList:
            if len(audiofile.split('_')) == 8:
                if extract_id_from_filename(audiofile) == str(file_name) and audiofile.split('_')[3] not in Total_file:
                    wavefile = wave.open(os.path.join(path, audiofile))
                    nframes = wavefile.getnframes()
                    sr = wavefile.getframerate()
                    each_data = np.frombuffer(wavefile.readframes(nframes), dtype=np.short)
                    resampled_audio_data = signal.resample(each_data, int(len(each_data) * new_sr / sr))
                    length = len(resampled_audio_data)
                    resampled_audio_data = Scaler.fit_transform(resampled_audio_data.reshape(length, 1))
                    Total_file.append(str(file_name))
                    Total_Data.append(resampled_audio_data.squeeze())
                    if label == 0:  # group = 0 健康人
                        Total_label.append(0)
                    else:  # group = 1 病人
                        Total_label.append(1)
    return Total_Data, Total_label


import re
import os


def extract_id_from_filename(filename):
    # 定义匹配规则的正则表达式
    pattern = re.compile(r'^read.*?_(\d+).*?_(\d+)_.*\.wav$')
    # 使用正则表达式匹配文件名
    match = pattern.match(filename)

    # 如果匹配成功，提取ID并返回；否则返回None
    if match:
        return match.group(2)
    else:
        return None


def estimate_snr(y, sr):
    # 计算能量
    signal_energy = sum(y ** 2)

    # 利用 librosa 进行预加重，以提高信噪比的估计
    y_preemph = librosa.effects.preemphasis(y)

    # 计算噪声的能量
    noise_energy = sum((y - y_preemph) ** 2)

    # 计算信噪比（以分贝为单位）
    snr = 10 * np.log10(signal_energy / noise_energy)

    return snr


def get_ids_from_files(directory):
    # 获取目录下所有文件
    files = os.listdir(directory)

    # 仅选择以 "read" 开头的文件
    read_files = [filename for filename in files if filename.startswith('read')]

    # 用于存储提取的ID
    ids = []

    # 遍历匹配文件名并提取ID
    for filename in read_files:
        if len(filename.split('_')) == 8:
            extracted_id = extract_id_from_filename(filename)
            if extracted_id:
                ids.append(int(extracted_id))

    return set(ids)


if __name__ == '__main__':
    # audiowavfile = "/home/idal-01/data/Adolescent_voice_all/data/中小学转WAV/第二次筛查/射阳"
    # audiowavfile = "/home/idal-01/data/Adolescent_voice_all/data/中小学转WAV/"
    #
    # # sites = ['射阳', '泰州', '宜兴']
    # # sites = ['1Allsites_reading']
    # sites = ['2Allsites_reading']
    # filesList = []
    # for site in sites:
    #     # 示例使用
    #     result_ids = get_ids_from_files(os.path.join(audiowavfile, site))
    #     filesList.extend(result_ids)
    # df = pd.DataFrame(filesList)
    # df.to_excel('adolescent_audio_2.xlsx', index=False)
    dirs = ['1Allsites_reading', '30<length<40/1']
    # dirs = ['1Allsites_reading']
    # exceltable = "/home/idal-01/data/Adolescent_voice_all/label/40-60/label/第一批所有人.xlsx"
    exceltable = "/home/idal-01/data/Adolescent_voice_all/label/30-60/label/重合/两次共有人群中评分不变-classification.xlsx"
    DATA, LABEL = [], []
    for path in dirs:
        audiowavfile = "/home/idal-01/data/Adolescent_voice_all/data/中小学转WAV"
        rootpath = os.path.join(audiowavfile, path)
        Total_Data, Total_label = dataprogress(rootpath, exceltable)
        DATA.extend(Total_Data)
        LABEL.extend(Total_label)
    # 写入pkl文件中
    print("开始写入文件中")
    f = open('./1adolescent_audio_all_123', 'wb')
    # 将数据 放入pickle中保存
    pickle.dump((DATA, LABEL), f)
    f.close()
    print("写入完毕")