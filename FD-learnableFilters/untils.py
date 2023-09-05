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
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import compute_class_weight
from torch import nn
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler
import csv
import librosa.feature
import librosa.display
import json


def dataprogress(prefix, prfix_labels):
    Scaler = StandardScaler()
    Total_Data, Total_label, Total_age = [], [], []
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
                                    Total_label.append(0)
                                    Total_age.append(T_age[ii])
                                    print(file_name)
                                elif 4 < PHQ_scores[ii] and label == 2 and 13 <= T_age[ii] < 25:
                                    wavefile = wave.open(os.path.join(DepressionAudio, sub_file))
                                    nframes = wavefile.getnframes()
                                    each_data = np.frombuffer(wavefile.readframes(nframes), dtype=np.short)
                                    length = len(each_data)
                                    each_data = Scaler.fit_transform(each_data.reshape(length, 1))
                                    Total_Data.append(each_data.squeeze())
                                    Total_label.append(1)
                                    Total_age.append(T_age[ii])
    # 增加后的新样本
    path = '/home/idal-01/data/ward_njnk/normal_control_resample'
    SexPath_1 = '/home/idal-01/data/NC_1.xlsx'
    SexPath_2 = '/home/idal-01/data/NC_2.xlsx'
    ExcelData_1 = pd.ExcelFile(SexPath_1)
    ExcelData_2 = pd.ExcelFile(SexPath_2)
    sex_nc_1 = pd.read_excel(ExcelData_1, sheet_name='无')
    sex_nc_2 = pd.read_excel(ExcelData_2, sheet_name='无抑郁')
    numb_1 = sex_nc_1[['用户编号']]['用户编号'].tolist()
    numb_2 = sex_nc_2[['用户编号']]['用户编号'].tolist()
    XM_1 = sex_nc_1[['录音编号']]['录音编号'].tolist()
    XM_2 = sex_nc_2[['姓名']]['姓名'].tolist()
    age_1 = sex_nc_1[['年龄']]['年龄'].tolist()
    age_2 = sex_nc_2[['年龄']]['年龄'].tolist()
    sex_1 = sex_nc_1[['性别']]['性别'].tolist()
    sex_2 = sex_nc_2[['性别']]['性别'].tolist()
    for idx, number in enumerate(XM_1):
        if age_1[idx] < 25:
            print(sex_1[idx])
            wavefile = wave.open(os.path.join(path, str(number) + '.wav'))
            nframes = wavefile.getnframes()
            sr = wavefile.getframerate()
            each_data = np.frombuffer(wavefile.readframes(nframes), dtype=np.short)
            length = len(each_data)
            each_data = Scaler.fit_transform(each_data.reshape(length, 1))
            Total_Data.append(each_data.squeeze())
            Total_label.append(0)
    for idx, BH in enumerate(XM_2):
        if age_2[idx] < 25:
            print(sex_2[idx])
            wavefile = wave.open(os.path.join(path, str(BH) + '.wav'))
            nframes = wavefile.getnframes()
            each_data = np.frombuffer(wavefile.readframes(nframes), dtype=np.short)
            length = len(each_data)
            each_data = Scaler.fit_transform(each_data.reshape(length, 1))
            Total_Data.append(each_data.squeeze())
            Total_label.append(0)
    # 写入pkl文件中
    print("开始写入文件中")
    f = open('./NRAC_L.pkl', 'wb')
    # 将数据 放入pickle中保存
    pickle.dump((Total_Data, Total_label), f)
    f.close()
    print("写入完毕")
    # return Total_Data, Total_label, Total_age


def dataprogress_type(prefix, prfix_labels):
    Total_Data, Total_label, Total_age = [], [], []
    for idx in range(5):
        file_path = 'P-' + str(idx + 1) + '_resample'
        DepressionAudio = os.path.join(prefix, file_path)
        ExcelData = pd.ExcelFile(os.path.join(prefix, prfix_labels))
        sheets = ExcelData.sheet_names
        for sheet in sheets:
            severe = pd.read_excel(ExcelData, sheet_name=sheet)
            Files_name = severe[['standard_id']]['standard_id'].tolist()
            PHQ_type = severe[['3种病数据超参搜索ALFF模型k=2_combat']]['3种病数据超参搜索ALFF模型k=2_combat'].tolist()
            T_age = severe[['age']]['age'].tolist()

            for ii, label in enumerate(PHQ_type):
                if label in ['A', 'C']:
                    file_name = Files_name[ii]
                    if file_name.split('_')[2] in ['S001']:
                        for sub_file in os.listdir(DepressionAudio):
                            if sub_file.find(str(file_name)) != -1:
                                if PHQ_type[ii] == 'C':
                                    wavefile = wave.open(os.path.join(DepressionAudio, sub_file))
                                    nframes = wavefile.getnframes()
                                    each_data = np.frombuffer(wavefile.readframes(nframes), dtype=np.short)
                                    Total_Data.append(each_data.squeeze())
                                    Total_label.append(1)
                                    Total_age.append(T_age[ii])
    path = '/home/ubnn/hardDisk/ward/normal_control_resample'
    SexPath_1 = '/home/ubnn/hardDisk/ward/NC_1.xlsx'
    SexPath_2 = '/home/ubnn/hardDisk/ward/NC_2.xlsx'
    ExcelData_1 = pd.ExcelFile(SexPath_1)
    ExcelData_2 = pd.ExcelFile(SexPath_2)
    sex_nc_1 = pd.read_excel(ExcelData_1, sheet_name='无')
    sex_nc_2 = pd.read_excel(ExcelData_2, sheet_name='无抑郁')
    XM_1 = sex_nc_1[['录音编号']]['录音编号'].tolist()
    XM_2 = sex_nc_2[['姓名']]['姓名'].tolist()
    for idx, number in enumerate(XM_1):
        print(number)
        wavefile = wave.open(os.path.join(path, str(number) + '.wav'))
        nframes = wavefile.getnframes()
        sr = wavefile.getframerate()
        print(sr)
        each_data = np.frombuffer(wavefile.readframes(nframes), dtype=np.short)
        Total_Data.append(each_data.squeeze())
        Total_label.append(0)
    for idx, BH in enumerate(XM_2):
        print(BH)
        wavefile = wave.open(os.path.join(path, BH + '.wav'))
        nframes = wavefile.getnframes()
        each_data = np.frombuffer(wavefile.readframes(nframes), dtype=np.short)
        Total_Data.append(each_data.squeeze())
        Total_label.append(0)

    return Total_Data, Total_label, Total_age


def dataprogress_test():
    Scaler = StandardScaler()
    Total_Data, Total_label, Total_age = [], [], []
    path = '/home/eric/mechanicalDisk/ward/NC_test/date_resample'
    Files = os.listdir(path)
    for idx, number in enumerate(Files):
        wavefile = wave.open(os.path.join(path, str(number)))
        nframes = wavefile.getnframes()
        each_data = np.frombuffer(wavefile.readframes(nframes), dtype=np.short)
        length = len(each_data)
        each_data = Scaler.fit_transform(each_data.reshape(length, 1))
        Total_Data.append(each_data.squeeze())
        Total_label.append(0)

    return Total_Data, Total_label, Total_age


def total_dataloader(args, Total_Data, Total_label):
    zip_list = list(zip(Total_Data, Total_label))
    random.Random(2022).shuffle(zip_list)
    non_severe_depressionList, labelList = zip(*zip_list)
    seg_level = args.audio_length
    stride = int(seg_level * 1)

    trainList_dataloader, testList_dataloader, valList_dataloader = [], [], []
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    non_severe_depressionList = np.array(non_severe_depressionList, dtype=object)
    labelList = np.array(labelList)
    toPersonList, toSex, CE_weights = [], [], []
    for train_index, test_index in kf.split(non_severe_depressionList, labelList):
        X_train, X_test = non_severe_depressionList[train_index], non_severe_depressionList[test_index]
        y_train, y_test = labelList[train_index], labelList[test_index]
        tv_folder = StratifiedKFold(n_splits=10, random_state=1, shuffle=True).split(X_train, y_train)
        # 划分验证集
        for t_idx, v_idx in tv_folder:
            train_split, train_labels = X_train[t_idx], y_train[t_idx]
            val_split, val_labels = X_train[v_idx], y_train[v_idx]
        # 构造训练集
        train_dataset = []
        train_labels_dataset = []
        for idx, data in enumerate(train_split):
            if len(data) / 16000 > seg_level:
                data_temp = data_segmentwd(data, int(16000 * seg_level), 16000 * stride, idx, 'train', args)

                m = data_temp.shape[0]
                label = train_labels[idx]
                train_labels_dataset.extend(label.repeat(m))
                train_dataset.extend(data_temp)
        """CE_Weight"""
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels_dataset),
                                             y=train_labels_dataset)
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        CE_weights.append(class_weights)

        # 构造验证集
        val_dataset = []
        val_labels_dataset = []
        for idx, data in enumerate(val_split):
            if len(data) / 16000 > seg_level:
                data_temp = data_segmentwd(data, int(16000 * seg_level), 16000 * stride, idx, 'val', args)

                m = data_temp.shape[0]
                label = val_labels[idx]
                val_labels_dataset.extend(label.repeat(m))
                val_dataset.extend(data_temp)
        # 构造测试集
        test_dataset = []
        test_labels_dataset = []
        test_sex_dataset = []
        demo_per = []
        for idx, data in enumerate(X_test):
            if len(data) / 16000 > seg_level:
                data_temp = data_segmentwd(data, int(16000 * seg_level), 16000 * stride, idx, 'test', args)
                m = data_temp.shape[0]
                label = y_test[idx]
                test_labels_dataset.extend(label.repeat(m))
                test_dataset.extend(data_temp)
                demo_per.append(data_temp.shape[0])
        toPersonList.append(demo_per)
        toSex.append(test_sex_dataset)

        # Train
        train_dataset = datafft(np.array(train_dataset), args)
        train_labels_dataset = torch.tensor(train_labels_dataset)
        # Test
        test_dataset = datafft(np.array(test_dataset), args)
        test_labels_dataset = torch.tensor(test_labels_dataset)
        # Val
        val_dataset = datafft(np.array(val_dataset), args)
        val_labels_dataset = torch.tensor(val_labels_dataset)

        train_dataset = datasets(train_dataset, train_labels_dataset)
        test_dataset = datasets(test_dataset, test_labels_dataset)
        val_dataset = datasets(val_dataset, val_labels_dataset)

        train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                           drop_last=True)
        val_dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        trainList_dataloader.append(train_dataset_loader)
        testList_dataloader.append(test_dataset_loader)
        valList_dataloader.append(val_dataset_loader)

    return trainList_dataloader, testList_dataloader, valList_dataloader, toPersonList, CE_weights


class datasets(Dataset):
    def __init__(self, x, label):
        self.labels = label
        self.x = x

    def __getitem__(self, idx):
        return_dic = {'x': self.x[idx],
                      'label': self.labels[idx]}

        return return_dic

    def __len__(self):
        return len(self.labels)


def data_segmentwd(raw_data, window_size, strides_size, idx, flag, args):
    windowList = []
    start = 0
    end = window_size
    while True:
        x = raw_data[start:end]
        if len(x) < window_size:
            break
        # power_spec = datafft(x, args)
        # mel_filter_banks = create_spectrogram_ori(x)
        windowList.append(x)
        if end >= raw_data.shape[0]:
            break
        start += strides_size
        end += strides_size
    window = np.array(windowList)
    return np.array(window)


def saveList(paraList, path):
    output = open(path, 'wb')
    pickle.dump(paraList, output)
    output.close()


def test_LabelVote(list_):
    label_cag = set(list_)
    lab, numbers = [], []
    for item in label_cag:
        numbers.append(list_.count(item))
        lab.append(item)
    demo = 0.0
    if len(numbers) == 1:
        return lab[0], demo
    else:
        numZero, numOne = numbers[0], numbers[1]

        print('The number of ' + str(lab[0]) + ' is : ', numZero)
        print('The number of ' + str(lab[1]) + ' is : ', numOne)

        if numZero > numOne:
            demo = numOne / (numZero + numOne)
            return lab[0], demo
        elif numZero == numOne:
            demo = numZero / (numZero + numOne)
            return 1, demo
        else:
            demo = numZero / (numZero + numOne)
            return lab[1], demo


def LabelVote(list_):
    counter = Counter(list_)
    majority = counter.most_common(1)[0][0]
    return majority


def SaveResult(data):
    f = open('results.csv', 'w')
    with f:
        w = csv.writer(f)
        for row in data:
            w.writerow(row)


def test_dataprogress(prefix, prfix_labels):
    Total_Data, Total_label = [], []
    file_path = 'test'
    DepressionAudio = os.path.join(prefix, file_path)
    ExcelData = pd.ExcelFile(os.path.join(prefix, prfix_labels))
    severe = pd.read_excel(ExcelData, sheet_name='Sheet1')
    Labels = severe[['diagnosis']]['diagnosis'].tolist()
    Files_name = severe[['standard_id']]['standard_id'].tolist()
    HAMD_scores = severe[['HAMD17_total_score']]['HAMD17_total_score'].tolist()
    for ii, label in enumerate(Labels):
        if label in [1, 2]:
            file_name = Files_name[ii]
            if file_name.split('_')[2] in ['S001']:
                for sub_file in os.listdir(DepressionAudio):
                    if sub_file.find(str(file_name)) != -1:
                        if HAMD_scores[ii] != '--' and HAMD_scores[ii] <= 7:
                            print(file_name)
                            wavefile = wave.open(os.path.join(DepressionAudio, sub_file))
                            nframes = wavefile.getnframes()
                            each_data = np.frombuffer(wavefile.readframes(nframes), dtype=np.short)
                            Total_Data.append(each_data.squeeze())
                            Total_label.append(0)
                            create_spectrogram_ori(each_data, int(HAMD_scores[ii]), 'Test组/' + str(ii))
                        elif HAMD_scores[ii] != '--' and 24 <= HAMD_scores[ii]:
                            print(file_name)
                            wavefile = wave.open(os.path.join(DepressionAudio, sub_file))
                            nframes = wavefile.getnframes()
                            each_data = np.frombuffer(wavefile.readframes(nframes), dtype=np.short)
                            Total_Data.append(each_data.squeeze())
                            Total_label.append(1)
                            create_spectrogram_ori(each_data, int(HAMD_scores[ii]), 'Test组/' + str(ii))
    return Total_Data, Total_label


def test_dataloader(args, Total_Data, Total_label):
    testList_dataloader = []
    X_test = np.array(Total_Data, dtype=object)
    y_test = np.array(Total_label)
    toPersonList, toSex = [], []

    seg_level = args.audio_length
    stride = int(0.5 * seg_level)
    test_dataset = []
    test_labels_dataset = []
    test_sex_dataset = []
    demo_per = []
    for idx, data in enumerate(X_test):
        if len(data) / 16000 > seg_level:
            data_temp = data_segmentwd(data, 16000 * seg_level, 16000 * stride, idx, 'test', args)

            m = data_temp.shape[0]
            label = y_test[idx]
            test_labels_dataset.extend(label.repeat(m))
            test_dataset.extend(data_temp)
            demo_per.append(data_temp.shape[0])

    toPersonList.append(demo_per)
    toSex.append(test_sex_dataset)

    test_dataset = np.array(test_dataset)
    test_labels_dataset = np.array(test_labels_dataset)

    test_sampler = datasets(test_dataset, test_labels_dataset)
    test_dataset_loader = torch.utils.data.DataLoader(test_sampler, batch_size=args.batch_size, shuffle=False,
                                                      num_workers=0)
    testList_dataloader.append(test_dataset_loader)

    return testList_dataloader, toPersonList


def create_spectrogram_ori(signal):
    NFFT = 512
    sample_rate = 16000
    pre_emphasis = 0.97
    # signal = signal[0: int(8 * sample_rate)]
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    frame_size, frame_stride = 0.025, 0.01
    frame_length, frame_step = int(round(frame_size * sample_rate)), int(round(frame_stride * sample_rate))
    signal_length = len(emphasized_signal)
    num_frames = int(np.ceil(np.abs(signal_length - frame_length) / frame_step)) + 1

    pad_signal_length = (num_frames - 1) * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)
    indices = np.arange(0, frame_length).reshape(1, -1) + np.arange(0, num_frames * frame_step, frame_step).reshape(-1,
                                                                                                                    1)
    frames = pad_signal[indices]
    hamming = np.hamming(frame_length)
    frames *= hamming
    mag_frames = np.absolute(np.fft.fft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * np.square(mag_frames))
    """从这下面都是mel滤波器"""
    # low_freq_mel = 0
    # high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    # nfilt = 64
    # mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    # hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    # fbank = np.zeros((nfilt, int(NFFT / 2 + 1)))
    # bin = (hz_points / (sample_rate / 2)) * (NFFT / 2)
    # for i in range(1, nfilt + 1):
    #     left = int(bin[i - 1])
    #     center = int(bin[i])
    #     right = int(bin[i + 1])
    #     for j in range(left, center):
    #         fbank[i - 1, j + 1] = (j + 1 - bin[i - 1]) / (bin[i] - bin[i - 1])
    #     for j in range(center, right):
    #         fbank[i - 1, j + 1] = (bin[i + 1] - (j + 1)) / (bin[i + 1] - bin[i])
    # filter_banks = np.dot(pow_frames, fbank.T)
    # filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    # filter_banks = 20 * np.log10(filter_banks)
    return pow_frames


def datafft(signal, args):
    # path = '/home/ubnn/PycharmProjects/LearnableFilter_Voice/drawing/Interpretable_FIg/'
    sample_rate = 16000
    NFFT = args.NFFT
    win_length, hop_length = int(0.025 * sample_rate), int(0.01 * sample_rate)
    # 计算短时傅里叶变换(STFT)
    window = torch.hann_window(win_length)
    stft = torch.stft(input=torch.from_numpy(signal), n_fft=NFFT, window=window, hop_length=hop_length,
                      win_length=win_length, normalized=True, center=False, return_complex=True)

    # 计算功率谱
    # mag_frames = np.abs(stft)
    power_spec = torch.square(torch.abs(stft))

    return power_spec


def dataMelspectrum(signal, args):
    # path = '/home/ubnn/PycharmProjects/LearnableFilter_Voice/drawing/Interpretable_FIg/'
    sample_rate = 16000
    NFFT = args.NFFT
    win_length, hop_length = int(0.025 * sample_rate), int(0.01 * sample_rate)
    # 计算短时傅里叶变换(STFT)
    stft = librosa.stft(signal, n_fft=NFFT, hop_length=hop_length, win_length=win_length)
    power_spec = np.square(np.abs(stft))
    melspectrogram = librosa.feature.melspectrogram(S=power_spec, n_mels=64, sr=sample_rate)

    # melspectrogramDemo = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_fft=NFFT, win_length=win_length, hop_length=hop_length, n_mels=64)
    return melspectrogram


def plot_spectrogram_spec(spec, ylabel, index):
    path = '/home/ubnn/PycharmProjects/LearnableFilter_Voice/drawing/Interpretable_FIg/'
    fig = plt.figure(figsize=(25, 8))
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.xlabel('Time(s)')
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path + str(index + '_' + ylabel))
    plt.close(fig)


import numpy as np


def mean_sd(name, results, args, Info_model):
    path = '/home/idal-01/code/TD-learnableFilters/New_results/'
    prec, recall, acc, f1, auc = [], [], [], [], []

    for i in results:
        prec.append(i['prec'])
        recall.append(i['recall'])
        f1.append(i['F1_w'])
        acc.append(i['acc'])
        auc.append(i['auc'])

    prec_mean, prec_sd = np.mean(prec), np.std(prec)
    recall_mean, recall_sd = np.mean(recall), np.std(recall)
    acc_mean, acc_sd = np.mean(acc), np.std(acc)
    f1_mean, f1_sd = np.mean(f1), np.std(f1)
    auc_mean, auc_sd = np.mean(auc), np.std(auc)

    with open(path + str(name) + '.txt', 'w') as file:
        file.write('Args:\n')
        for arg_name, arg_value in args.__dict__.items():
            file.write(f"{arg_name}: {arg_value}\n")  # 将每个参数名和值写入文件
        file.write('\n' + '=' * 100 + '\n')  # 添加间隔符号
        file.write('\nResults:\n')
        result_str = ""
        for i, result in enumerate(results):
            result_str += f"Result {i + 1}: "
            result_str += f"Precision: {result['prec']:.4f} | "
            result_str += f"Recall: {result['recall']:.4f} | "
            result_str += f"F1: {result['F1_w']:.4f} | "
            result_str += f"Accuracy: {result['acc']:.4f} | "
            result_str += f"AUC: {result['auc']:.4f}\n"

        file.write(result_str)
        file.write('\n' + '=' * 100 + '\n')  # 添加间隔符号
        file.write('\nMean+-SD:\n')
        file.write('Prec_mean + SD == %.3f +- %.3f\n' % (prec_mean, prec_sd))
        file.write('Recall_mean + SD == %.3f +- %.3f\n' % (recall_mean, recall_sd))
        file.write('ACC_mean + SD == %.3f +- %.3f\n' % (acc_mean, acc_sd))
        file.write('F1_mean + SD == %.3f +- %.3f\n' % (f1_mean, f1_sd))
        file.write('AUC_mean + SD == %.3f +- %.3f\n' % (auc_mean, auc_sd))

        file.write('\n' + '=' * 100 + '\n')  # 添加间隔符号
        file.write('\nOur Model:\n')
        for child in Info_model.children():
            if isinstance(child, nn.Module):
                child = str(child)
                file.write('\n' + '=' * 100 + '\n')  # 添加间隔符号
                file.write(child + '\n')


def dataprogress2(prefix):
    all_items = os.listdir(prefix)
    Scaler = StandardScaler()
    Total_Data, Total_label = [], []
    for file_path in all_items:
        DepressionAudio = os.path.join(prefix, file_path)
        waveList = ['positive_out.wav', 'neutral_out.wav', 'negative_out.wav']
        data = np.array([], dtype=np.int16)
        # 循环读取 WAV 文件，并将数据拼接到数组中
        for wav_file in waveList:
            with wave.open(os.path.join(DepressionAudio, wav_file), 'rb') as wf:
                # 读取音频数据并拼接到数组中
                frames = wf.readframes(wf.getnframes())
                arr = np.frombuffer(frames, dtype=np.int16)
                data = np.concatenate((data, arr))
        length = len(data)
        data = Scaler.fit_transform(data.reshape(length, 1))
        Total_Data.append(data.squeeze())

        with open(os.path.join(DepressionAudio, 'new_label.txt'), 'r') as file:
            content = float(file.read())
        if content < 53:
            Total_label.append(0)
        else:
            Total_label.append(1)
            waveList2 = ['negative_out.wav', 'neutral_out.wav', 'positive_out.wav']
            data2 = np.array([], dtype=np.int16)
            # 循环读取 WAV 文件，并将数据拼接到数组中
            for wav_file in waveList2:
                with wave.open(os.path.join(DepressionAudio, wav_file), 'rb') as wf:
                    # 读取音频数据并拼接到数组中
                    frames = wf.readframes(wf.getnframes())
                    sr = wf.getframerate()
                    arr = np.frombuffer(frames, dtype=np.int16)
                    data2 = np.concatenate((data2, arr))
            length = len(data2)
            data2 = Scaler.fit_transform(data2.reshape(length, 1))
            Total_Data.append(data2.squeeze())
            Total_label.append(1)
            waveList3 = ['negative_out.wav', 'positive_out.wav', 'neutral_out.wav']
            data2 = np.array([], dtype=np.int16)
            # 循环读取 WAV 文件，并将数据拼接到数组中
            for wav_file in waveList3:
                with wave.open(os.path.join(DepressionAudio, wav_file), 'rb') as wf:
                    # 读取音频数据并拼接到数组中
                    frames = wf.readframes(wf.getnframes())
                    arr = np.frombuffer(frames, dtype=np.int16)
                    data2 = np.concatenate((data2, arr))
            length = len(data2)
            data2 = Scaler.fit_transform(data2.reshape(length, 1))
            Total_Data.append(data2.squeeze())
            Total_label.append(1)
            waveList4 = ['neutral_out.wav', 'negative_out.wav', 'positive_out.wav']
            data2 = np.array([], dtype=np.int16)
            # 循环读取 WAV 文件，并将数据拼接到数组中
            for wav_file in waveList3:
                with wave.open(os.path.join(DepressionAudio, wav_file), 'rb') as wf:
                    # 读取音频数据并拼接到数组中
                    frames = wf.readframes(wf.getnframes())
                    arr = np.frombuffer(frames, dtype=np.int16)
                    data2 = np.concatenate((data2, arr))
            length = len(data2)
            data2 = Scaler.fit_transform(data2.reshape(length, 1))
            Total_Data.append(data2.squeeze())
            Total_label.append(1)
    return Total_Data, Total_label


def dataprogress_dataset3800(prefix):
    Scaler = StandardScaler()
    Total_Data, Total_label, Total_age = [], [], []
    for idx in range(8):
        file_path = 'audio/age1' + str(idx)
        DepressionAudio = os.path.join(prefix, file_path)
        i = 0
        for sub_file in os.listdir(DepressionAudio):
            i += 1
            if i > 10:
                break
            wavefile = wave.open(os.path.join(DepressionAudio, sub_file))
            nframes = wavefile.getnframes()
            each_data = np.frombuffer(wavefile.readframes(nframes), dtype=np.short)
            length = len(each_data)
            each_data = Scaler.fit_transform(each_data.reshape(length, 1))
            Total_Data.append(each_data.squeeze())
            Total_label.append(0)
            Total_age.append('1' + str(idx))
    for idx in range(8):
        file_path = 'abnormal_audio/age1' + str(idx)
        DepressionAudio = os.path.join(prefix, file_path)
        i = 0
        for sub_file in os.listdir(DepressionAudio):
            i += 1
            if i > 10:
                break
            wavefile = wave.open(os.path.join(DepressionAudio, sub_file))
            nframes = wavefile.getnframes()
            each_data = np.frombuffer(wavefile.readframes(nframes), dtype=np.short)
            length = len(each_data)
            each_data = Scaler.fit_transform(each_data.reshape(length, 1))
            Total_Data.append(each_data.squeeze())
            Total_label.append(1)
            Total_age.append('1' + str(idx))
    return Total_Data, Total_label, Total_age


def dataprogress3(prefix):
    # prefix = '/home/idal-01/data/CMDC/tmp/'
    all_parts = os.listdir(prefix)
    Scaler = StandardScaler()
    Total_Data, Total_label = [], []
    for parts in all_parts:
        file_list = os.path.join(prefix, parts)  # '/home/idal-01/data/CMDC/tmp/part#'
        part_person = os.listdir(file_list)
        for person in part_person:
            file_path = os.path.join(file_list, person)  # '/home/idal-01/data/CMDC/tmp/part#/HC#'
            waveList = glob.glob(os.path.join(file_path, '*.wav'))
            data = np.array([], dtype=np.int16)
            for wav_file in waveList:
                with wave.open(wav_file, 'rb') as wf:
                    # 读取音频数据并拼接到数组中
                    frames = wf.readframes(wf.getnframes())
                    sr = wf.getframerate()
                    arr = np.frombuffer(frames, dtype=np.int16)
                    seconds = len(arr) / sr
                    data = np.concatenate((data, arr))
            length = len(data)
            if length != 0:
                data = Scaler.fit_transform(data.reshape(length, 1))
                Total_Data.append(data.squeeze())
                if person[:2] == 'HC':
                    Total_label.append(0)
                else:
                    Total_label.append(1)

    return Total_Data, Total_label
