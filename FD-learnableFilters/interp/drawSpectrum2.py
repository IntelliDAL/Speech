# - * - coding: utf-8 - * -
import argparse
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.autograd import Variable
from torch import nn
from model.Filter_files.FD_filters import GaussianShape
from model.Filter_files import postprocessing
from model.Filter_files.LearnableFilters import LearnableFilters
from model.model_main import InfoAttentionClassifier
from torch.utils.data.dataset import Dataset

def displayWaveform():
    x, sr = librosa.load(r'D:\P-1_resample\NN_00000004_S002-P.wav', sr=16000)
    # 提取前6秒的语音信号
    duration = 6  # 时间段长度（秒）
    y = x[:int(sr * duration)]
    time = np.arange(0, len(y)) * (1.0 / sr)
    # time = librosa.times_like(y, sr=sr)
    # librosa.display.waveplot(samples, sr)
    plt.figure(figsize=(8, 3), dpi=600)
    plt.plot(time, y)
    plt.title("Waveform")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.savefig("D:\\northeastern university\\EATD-learnableFilters\\drawing\\Spectrum\\波形图.png")
    plt.show()
    plt.close()


def displaySpectrum():
    x, sr = librosa.load(r'D:\P-1_resample\NN_00000004_S002-P.wav', sr=16000)
    # 提取前6秒的语音信号
    duration = 6  # 时间段长度（秒）
    y = x[:int(sr * duration)]
    # 计算FFT
    fft_result = np.fft.fft(y)

    # 提取频率轴
    freq_axis = np.fft.fftfreq(len(fft_result), d=1 / sr)
    # 保留非负频率部分
    positive_freq_axis = freq_axis[:len(freq_axis) // 2]
    positive_fft_result = fft_result[:len(fft_result) // 2]
    # 绘制频率谱线
    plt.figure(figsize=(8, 3), dpi=600)
    plt.plot(positive_freq_axis, np.abs(positive_fft_result))
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.savefig("D:\\northeastern university\\EATD-learnableFilters\\drawing\\Spectrum\\幅度谱.png")
    plt.show()
    plt.close()


def displaySpectrogram():
    x, sr = librosa.load(r'D:\P-1_resample\NN_00000004_S002-P.wav', sr=16000)
    # 提取前6秒的语音信号
    duration = 6  # 时间段长度（秒）
    y = x[:int(sr * duration)]

    sample_rate = 16000
    NFFT = 512
    win_length, hop_length = int(0.025 * sample_rate), int(0.01 * sample_rate)
    # 计算语谱图
    spectrogram = librosa.stft(y, n_fft=NFFT, hop_length=hop_length, win_length=win_length) ** 2
    spectrogram_db = librosa.power_to_db(np.abs(spectrogram), ref=np.max)
    # spectrogram = librosa.stft(y, n_fft=NFFT, hop_length=hop_length, win_length=win_length)
    # spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)
    # 绘制语谱图
    plt.figure(figsize=(8, 3), dpi=600)
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis='s', y_axis='linear', vmin=-80, vmax=0)
    # plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.savefig("D:\\northeastern university\\EATD-learnableFilters\\drawing\\Spectrum\\originalresult.png")
    # plt.show()
    plt.close()


def displayMelspectrogram():
    x, sr = librosa.load(r'D:\P-1_resample\NN_00000004_S002-P.wav', sr=16000)
    # 提取前6秒的语音信号
    duration = 6  # 时间段长度（秒）
    y = x[:int(sr * duration)]

    hop_length = int(0.01 * sr)
    win_length = int(0.025 * sr)
    # 计算梅尔频谱
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, win_length=win_length, n_mels=64, hop_length=hop_length)
    melspectrogram_db = librosa.power_to_db(melspectrogram, ref=np.max)

    # 绘制梅尔频谱图
    plt.figure(figsize=(8, 3), dpi=600)
    librosa.display.specshow(melspectrogram_db, sr=sr, x_axis='s', y_axis='linear', vmin=-80, vmax=0)
    # plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.savefig("D:\\northeastern university\\EATD-learnableFilters\\drawing\\Spectrum\\Melresult.png")
    # plt.show()
    plt.close()

def displayDiff():
    x, sr = librosa.load(r'D:\P-1_resample\NN_00000004_S002-P.wav', sr=16000)
    # 提取前6秒的语音信号
    duration = 6  # 时间段长度（秒）
    y = x[:int(sr * duration)]
    # 计算语谱图
    spectrogram = np.abs(librosa.stft(y)) ** 2
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    # 计算梅尔频谱图
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    melspectrogram_db = librosa.power_to_db(melspectrogram, ref=np.max)

    # # 调整语谱图的形状以匹配梅尔频谱图
    spectrogram1 = np.resize(spectrogram_db, (128, 626))
    # 绘制差距频谱图
    diff_spectrogram = melspectrogram - spectrogram1
    diff_spectrogram_db = librosa.power_to_db(diff_spectrogram, ref=np.max)

    # 绘制语谱图
    plt.subplot(3, 1, 1)
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')

    # 绘制梅尔频谱图
    plt.subplot(3, 1, 2)
    librosa.display.specshow(melspectrogram_db, sr=sr, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')

    # 绘制差距频谱图
    plt.subplot(3, 1, 3)
    librosa.display.specshow(diff_spectrogram_db, sr=sr, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Difference Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()
    plt.close()

class datasets(Dataset):
    def __init__(self, x):
        self.x = x

    def __getitem__(self, idx):
        return_dic = {'x': self.x[idx]}

        return return_dic

    def __len__(self):
        return len(self.x)

def total_dataloader(Total_Data):
    seg_level = 6
    stride = 6

    # 构造测试集
    test_dataset = []
    data_temp = data_segmentwd(Total_Data, 16000 * seg_level, 16000 * stride)
    test_dataset.extend(data_temp)

    # Test
    test_dataset = np.array(test_dataset)

    test_dataset.reshape(-1,257,601)

    return test_dataset

def data_segmentwd(raw_data, window_size, strides_size):
    windowList = []
    start = 0
    end = window_size
    while True:
        x = raw_data[start:end]
        if len(x) < window_size:
            break
        power_spec = datafft(x)
        windowList.append(power_spec)
        # mel_filter_banks = create_spectrogram_ori(x)
        # windowList.append(mel_filter_banks)
        if end >= raw_data.shape[0]:
            break
        start += strides_size
        end += strides_size
    window = np.array(windowList)
    return np.array(window)

def datafft(signal):
    # path = '/home/ubnn/PycharmProjects/LearnableFilter_Voice/drawing/Interpretable_FIg/'
    sample_rate = 16000
    NFFT = 512
    win_length, hop_length = int(0.025 * sample_rate), int(0.01 * sample_rate)
    # 计算短时傅里叶变换(STFT)
    stft = librosa.stft(signal, n_fft=NFFT, hop_length=hop_length, win_length=win_length)

    # 计算功率谱
    # mag_frames = np.abs(stft)
    power_spec = np.square(np.abs(stft))

    return power_spec

if __name__ == '__main__':
    # displayWaveform()
    # displaySpectrum()
    # displaySpectrogram()
    # displayMelspectrogram()
    # displayDiff()
    filename = 'D:\\P-1_resample\\NN_00000004_S002-P.wav'
    librosa_audio, librosa_sample_rate = librosa.load(filename, sr=16000)
    # 提取前6秒的语音信号
    duration = 6  # 时间段长度（秒）
    librosa_audio = librosa_audio[:int(16000 * duration)]
    checkpoint_path = './modelpacksigma0.005/model_1.ckpt'
    localmodel = torch.load(checkpoint_path)
    model = localmodel.eval()
    ckpt_path_Info = './modelpacksigma0.005/weights_1.ckpt'
    pretrained_dict = torch.load(ckpt_path_Info)  # 更改加载设备
    if ckpt_path_Info is not None:
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    input = np.array(librosa_audio)

    data_test = total_dataloader(input)

    input = torch.Tensor(data_test).cuda()

    output = model.AudioSpectral.FD_Learned_Filters(input)
    output = output.view(64, -1)
    output = output.cpu().detach().numpy()
    Xdb = librosa.power_to_db(output, ref=np.max)
    # 绘制高斯滤波后的语谱图
    plt.figure(figsize=(8, 3), dpi=600)
    librosa.display.specshow(Xdb, sr=16000, x_axis='s', y_axis='linear', vmin=-80, vmax=0)
    # librosa.display.specshow(Xdb, sr=fs, x_axis='s', y_axis='linear')
    # plt.colorbar(format='%+2.0f')
    plt.title('Our Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.savefig("./modelpacksigma0.005/ourresult1.png")
    # plt.show()
    plt.close()


