import os
import argparse

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import glob
import wave
import numpy as np
import pickle


def data_segmentwd(raw_data, window_size, strides_size):  # 数据分布
    windowList = []
    start = 0
    end = window_size
    while True:
        x = raw_data[start:end]
        if len(x) < window_size:
            break
        windowList.append(x)
        if end >= raw_data.shape[0]:
            break
        start += strides_size
        end += strides_size
    window = np.array(windowList)
    return np.array(window)


def read_wavefile(filename):
    # 开始读取wav文件
    file = wave.open(filename, 'r')
    params = file.getparams()  # 获取得到的所有参数
    n_channels, samp_with, fram_rate, wav_length = params[:4]
    str_data = file.readframes(wav_length)
    wave_data = np.frombuffer(str_data, dtype=np.short)
    time = np.arange(0, wav_length) * (1.0 / fram_rate)
    file.close()
    return wave_data, time, fram_rate


def read_IEMocap(args):
    prefix = '/home/idal-01/data/'
    prfix_labels = 'results.xlsx'
    # 设置数据库的路径
    Scaler = StandardScaler()
    stride = 1 * args.audio_length
    Total_Data = []
    for idx in range(5):
        file_path = 'ward_njnk/P-' + str(idx + 1) + '_resample'
        DepressionAudio = os.path.join(prefix, file_path)
        ExcelData = pd.ExcelFile(os.path.join(prefix, prfix_labels))
        sheets = ExcelData.sheet_names
        for sheet in sheets:
            severe = pd.read_excel(ExcelData, sheet_name=sheet)
            Labels = severe[['diagnosis']]['diagnosis'].tolist()
            Files_name = severe[['standard_id']]['standard_id'].tolist()
            # PHQ_scores = severe[['HAMD17_total_score']]['HAMD17_total_score'].tolist()
            PHQ_scores = severe[['PHQ_total_score']]['PHQ_total_score'].tolist()
            T_age = severe[['age']]['age'].tolist()
            T_sex = severe[['sex']]['sex'].tolist()
            for ii, label in enumerate(Labels):
                if label in [1, 2]:
                    file_name = Files_name[ii]
                    if file_name.split('_')[2] in ['S001']:
                        for sub_file in os.listdir(DepressionAudio):
                            if sub_file.find(str(file_name)) != -1:
                                if PHQ_scores[ii] <= 4 and (label == 1) and 13 <= T_age[ii] < 25:
                                    print(T_sex[ii])
                                    wavefile = wave.open(os.path.join(DepressionAudio, sub_file))
                                    nframes = wavefile.getnframes()
                                    each_data = np.frombuffer(wavefile.readframes(nframes), dtype=np.short)
                                    length = len(each_data)
                                    each_data = Scaler.fit_transform(each_data.reshape(length, 1))
                                    Total_Data.append(each_data.squeeze())
                                    print(file_name)
                                elif 4 < PHQ_scores[ii] and label == 2 and 13 <= T_age[ii] < 25:
                                    wavefile = wave.open(os.path.join(DepressionAudio, sub_file))
                                    nframes = wavefile.getnframes()
                                    each_data = np.frombuffer(wavefile.readframes(nframes), dtype=np.short)
                                    length = len(each_data)
                                    each_data = Scaler.fit_transform(each_data.reshape(length, 1))
                                    Total_Data.append(each_data.squeeze())

    # 写入pkl文件中
    train_dataset = []
    for idx, data in enumerate(Total_Data):
        data_temp = data_segmentwd(data, int(16000 * args.audio_length), int(16000 * stride))
        train_dataset.extend(data_temp)
    print("开始写入文件中")
    print(len(train_dataset))
    f = open('./NRAC_segment.pkl', 'wb')
    # 将数据 放入pickle中保存
    pickle.dump(np.array(train_dataset), f)
    # pickle.dump(Total_Data, f)
    f.close()
    print("写入完毕")
    pass


def read_picklefile():
    f = open('./IEMOCAP.pkl', 'rb')
    train_data, train_lable, times, speech_file = pickle.load(f)
    print(len(train_lable))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fs', type=int, default=16000)
    parser.add_argument('--audio_length', type=int, default=6.46)
    args = parser.parse_args()
    read_IEMocap(args)
    # read_picklefile()
    print("6-kinds")
