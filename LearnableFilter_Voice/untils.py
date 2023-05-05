import os
import pickle
import random
import wave
from collections import Counter
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import compute_class_weight
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler
import csv


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
            PHQ_scores = severe[['HAMD17_total_score']]['HAMD17_total_score'].tolist()
            T_age = severe[['age']]['age'].tolist()
            T_sex = severe[['sex']]['sex'].tolist()
            for ii, label in enumerate(Labels):
                if label in [1, 2]:
                    file_name = Files_name[ii]
                    if file_name.split('_')[2] in ['S001']:
                        for sub_file in os.listdir(DepressionAudio):
                            if sub_file.find(str(file_name)) != -1:
                                if PHQ_scores[ii] <= 7 and (label == 1) and 13 <= T_age[ii] < 25:
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
                                elif 7 < PHQ_scores[ii]<17 and label == 2 and 13 <= T_age[ii] < 25:
                                    wavefile = wave.open(os.path.join(DepressionAudio, sub_file))
                                    nframes = wavefile.getnframes()
                                    each_data = np.frombuffer(wavefile.readframes(nframes), dtype=np.short)
                                    length = len(each_data)
                                    each_data = Scaler.fit_transform(each_data.reshape(length, 1))
                                    Total_Data.append(each_data.squeeze())
                                    Total_label.append(1)
                                    Total_age.append(T_age[ii])
    return Total_Data, Total_label, Total_age


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
                data_temp = data_segmentwd(data, 16000 * seg_level, 16000 * stride, idx, 'train')
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
                data_temp = data_segmentwd(data, 16000 * seg_level, 16000 * stride, idx, 'val')
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
                data_temp = data_segmentwd(data, 16000 * seg_level, 16000 * stride, idx, 'test')
                m = data_temp.shape[0]
                label = y_test[idx]
                test_labels_dataset.extend(label.repeat(m))
                test_dataset.extend(data_temp)
                demo_per.append(data_temp.shape[0])
        toPersonList.append(demo_per)
        toSex.append(test_sex_dataset)

        # Train
        train_dataset = np.array(train_dataset)
        train_labels_dataset = np.array(train_labels_dataset)
        # Test
        test_dataset = np.array(test_dataset)
        test_labels_dataset = np.array(test_labels_dataset)
        # Val
        val_dataset = np.array(val_dataset)
        val_labels_dataset = np.array(val_labels_dataset)

        train_dataset = datasets(train_dataset, train_labels_dataset)
        test_dataset = datasets(test_dataset, test_labels_dataset)
        val_dataset = datasets(val_dataset, val_labels_dataset)

        train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                           pin_memory=True, drop_last=True)
        val_dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                                         pin_memory=True)
        test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                          pin_memory=True)

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


def data_segmentwd(raw_data, window_size, strides_size, idx, flag):
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
            data_temp = data_segmentwd(data, 16000 * seg_level, 16000 * stride, idx, 'test')
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
    path = '/home/ubnn/PycharmProjects/LearnableFilter_Voice/drawing/Interpretable_FIg/'
    sample_rate = 16000

    pre_emphasis = 0.97
    signal = signal[0: int(8 * sample_rate)]
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
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))

    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    nfilt = 64
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    fbank = np.zeros((nfilt, int(NFFT / 2 + 1)))
    bin = (hz_points / (sample_rate / 2)) * (NFFT / 2)
    for i in range(1, nfilt + 1):
        left = int(bin[i - 1])
        center = int(bin[i])
        right = int(bin[i + 1])
        for j in range(left, center):
            fbank[i - 1, j + 1] = (j + 1 - bin[i - 1]) / (bin[i] - bin[i - 1])
        for j in range(center, right):
            fbank[i - 1, j + 1] = (bin[i + 1] - (j + 1)) / (bin[i + 1] - bin[i])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    return filter_banks


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


def Mean_SD(name, results):
    path = '/home/idal-01/code/Local/LearnableFilter_Voice/Results_all/'
    Prec, Recall, Acc, F1, Auc = [], [], [], [], []
    for i in results:
        Prec.append(i['prec'])
        Recall.append(i['recall'])
        F1.append(i['F1_w'])
        Acc.append(i['acc'])
        Auc.append(i['auc'])

    Prec_mean, Prec_SD = np.array(Prec).mean(), np.array(Prec).std()
    Recall_mean, Recall_SD = np.array(Recall).mean(), np.array(Recall).std()
    ACC_mean, ACC_SD = np.array(Acc).mean(), np.array(Acc).std()
    F1_mean, F1_SD = np.array(F1).mean(), np.array(F1).std()
    Auc_mean, Auc_SD = np.array(Auc).mean(), np.array(Auc).std()
    file = open(path + str(name) + '.txt', 'w')
    file.write('Prec_mearn + SD == %.3f  +- ' % Prec_mean), file.write('%.3f' % Prec_SD + '\n' + '\n')
    file.write('Recall_mearn + SD == %.3f  +- ' % Recall_mean), file.write('%.3f' % Recall_SD + '\n' + '\n')
    file.write('ACC_mearn + SD == %.3f  +- ' % ACC_mean), file.write('%.3f' % ACC_SD + '\n' + '\n')
    file.write('F1_mearn + SD == %.3f  +- ' % F1_mean), file.write('%.3f' % F1_SD + '\n' + '\n')
    file.write('AUC_mearn + SD == %.3f  +- ' % Auc_mean), file.write('%.3f' % Auc_SD + '\n' + '\n')
    file.close()
