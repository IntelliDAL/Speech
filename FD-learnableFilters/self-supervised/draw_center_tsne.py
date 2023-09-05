import argparse
import os
import pickle
from time import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from numpy import loadtxt
from scipy.io import loadmat
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
import random
from pathlib import Path
import matplotlib as mpl

fpath = Path(mpl.get_data_path(), "fonts/ttf/cmb10.ttf")
from torch.utils.data import Dataset

from simsiam_CD import SimSiam


def get_data_():
    digits = datasets.load_digits(n_class=6)  # 取前六种数字图片，0-5
    data = digits.data  # data.shape=[1083,64]，表示1084张图片，每个图片8*8但是将图片表示为一个行向量
    label = digits.target  # 表示取出的1083个图片对应的数字
    n_samples, n_features = data.shape  # 图片数1083和每张图片的维度64
    return data, label, n_samples, n_features


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


def plot_embedding(index, resultI, resultT, title):  # 传入1083个2维数据，1083个标签，图表标题
    # x_min, x_max = np.min(resultI, 0), np.max(resultI, 0)  # 分别求出每一列最小值和最大值
    # resultI = (resultI - x_min) / (x_max - x_min)  # 将数据进行正则化，分母为数据的总长度，因此分子一定小于分母，生成的矩阵元素都是0-1区间内的
    # x_min, x_max = np.min(resultT, 0), np.max(resultT, 0)  # 分别求出每一列最小值和最大值
    # resultT = (resultT - x_min) / (x_max - x_min)  # 将数据进行正则化，分母为数据的总长度，因此分子一定小于分母，生成的矩阵元素都是0-1区间内的

    # plt.figure(figsize=(6,5))  # 创建一个画布
    plt.figure()  # 创建一个画布

    colors = ['#c22f2f', '#449945']
    colors_T = ['#c22f2f', 'blue']

    # markers = ['*', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 'P', '*', 'h', '+', '*', 'x']
    markers = ['o', 'x']
    plt.scatter(resultI[:, 0], resultI[:, 1], s=12, marker=markers[1], c=colors[0], label='SRs')
    plt.scatter(resultT[:, 0], resultT[:, 1], s=12, marker=markers[0], c=colors_T[1], label='QRs')
    legend = plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), frameon=False, prop={'size': '12'})
    for text in legend.get_texts():
        text.set_fontsize(30)  # 设置字体大小
        text.set_fontweight('bold')  # 设置字体加粗
        text.set_font(fpath)  # 设置字体类型
        # text.set_color('darkred')  # 设置字体颜色
    plt.xticks((), weight='bold', font=fpath)  # 不显示坐标刻度
    plt.yticks((), weight='bold', font=fpath)
    # plt.title(title)  # 设置标题
    plt.savefig(str(index) + '_' + 't-SNE.pdf')
    plt.show()


"""主函数"""


def total_dataloader(args, Total_Data, Total_label):
    seg_level = args.audio_length
    stride = int(seg_level * 1)
    train_split = np.array(Total_Data, dtype=object)
    Total_label = np.array(Total_label)

    train_dataset = []
    train_labels_dataset = []
    for idx, data in enumerate(train_split):
        if len(data) / 16000 > seg_level:
            data_temp = data_segmentwd(data, int(16000 * seg_level), 16000 * stride, idx, 'train', args)
            m = data_temp.shape[0]
            label = Total_label[idx]
            train_labels_dataset.extend(label.repeat(m))
            train_dataset.extend(data_temp)
        # Train
    train_dataset = datafft(np.array(train_dataset), args)
    train_labels_dataset = torch.tensor(train_labels_dataset)
    train_dataset = datasets(train_dataset, train_labels_dataset)
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                       drop_last=True)

    return train_dataset_loader


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


def set_seed(data):
    torch.manual_seed(data)
    torch.cuda.manual_seed_all(data)
    np.random.seed(data)
    random.seed(data)
    torch.backends.cudnn.deterministic = True


def main():
    set_seed(1)
    fs = 16000
    cw_len = 6.46
    wlen = int(fs * cw_len)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate_1', type=float, default=0.0005)
    parser.add_argument('--input_dim', type=int, default=wlen)
    parser.add_argument('--fs', type=int, default=16000)
    parser.add_argument('--audio_length', type=int, default=cw_len)
    parser.add_argument('--input_channels', type=int, default=1)
    parser.add_argument('--output_channels', type=int, default=64)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--skip_channels', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=5)
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--dilation', type=int, default=2)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--aptim', type=str, default='adam')
    parser.add_argument('--initializer', type=str, default='random')  # mel_scale
    parser.add_argument('--experiment', type=str, default='FD-filterPretrained')

    # TD滤波器
    parser.add_argument('--filter_size', type=int, default=513)
    parser.add_argument('--filter_num', type=int, default=64)
    parser.add_argument('--frame_num', type=int, default=640)
    parser.add_argument('--NFFT', type=int, default=1024)
    parser.add_argument('--sigma_coff', type=float, default=0.0015)
    args = parser.parse_args()
    model = SimSiam(args).cuda()
    print(model)
    """获取测试集数据，分析测试集合中，标签的对应类别，大概意思测试集合有些标签，我们根据这些进行表征提取和显示"""
    # 观察与训练后的结果
    f = open('/home/idal-01/code/TD-learnableFilters/NRAC_L.pkl', 'rb')
    data, Labels = pickle.load(f)
    trainList_dataloader = total_dataloader(args, data, Labels)
    ckpt_path = '/home/idal-01/code/TD-learnableFilters/self-supervised/pre-trained_model/Encoder_50_I.pt'
    ckpt_path_T = '/home/idal-01/code/TD-learnableFilters/self-supervised/pre-trained_model/Encoder_50_T.pt'
    pretrained_dict_ = torch.load(ckpt_path)
    pretrained_dict_T = torch.load(ckpt_path_T)
    model_dict = model.encoder.EncoderI.state_dict()
    model_dict_T = model.encoder.EncoderT.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict_.items() if k in model_dict}
    pretrained_dict_T = {k: v for k, v in pretrained_dict_T.items() if k in model_dict_T}
    model_dict.update(pretrained_dict)
    model_dict_T.update(pretrained_dict_T)
    model.encoder.EncoderT.load_state_dict(model_dict_T, strict=False)
    model.encoder.EncoderI.load_state_dict(model_dict, strict=False)
    print('Load data finished' + '\n')

    print('The number of patient: ', len(np.where(np.array(Labels) == 1)[0]))
    print('The number of NC: ', len(np.where(np.array(Labels) == 0)[0]))
    # data, label, n_samples, n_features = get_data()  # data种保存[1083,64]的向量
    for train_index in range(1):
        QRs, SRs, Label = [], [], []
        for batch_idx, batch_data in enumerate(trainList_dataloader):
            data, label = batch_data['x'], batch_data['label']
            featuresI, featuresT, _ = model.encoder(data.type(torch.FloatTensor).cuda())
            # 不同类别的SR和QR特征
            SR, QR = featuresI.cpu().detach().numpy(), featuresT.cpu().detach().numpy()

            SR = SR.reshape(SR.shape[0], -1)
            QR = QR.reshape(QR.shape[0], -1)
            QRs.extend(QR)
            SRs.extend(SR)
            Label.extend(label.cpu().detach().numpy())
        Length = len(SRs)
        combined_data = np.vstack((np.array(SRs), np.array(QRs)))
        # x_min, x_max = np.min(combined_data, 0), np.max(combined_data, 0)  # 分别求出每一列最小值和最大值
        # combined_data = (combined_data - x_min) / (x_max - x_min)  # 将数据进行正则化，分母为数据的总长度，因此分子一定小于分母，生成的矩阵元素都是0-1区间内的
        tsne = TSNE(n_components=2, init='pca',
                    random_state=0)  # n_components将64维降到该维度，默认2；init设置embedding初始化方式，可选pca和random，pca要稳定些
        results = tsne.fit_transform(combined_data)  # 进行降维，[1083,64]-->[1083,2]
        resultI, resultT = results[0:Length, ], results[Length:2 * Length, ]
        plot_embedding(train_index, resultI, resultT, 't-SNE embedding of speech representation')  # 显示数据


if __name__ == '__main__':
    device_index = 0  # 目标GPU的索引
    torch.cuda.set_device(device_index)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    main()
