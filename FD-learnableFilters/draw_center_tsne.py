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

from untils import total_dataloader

"""获取数据"""
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_data_():
    digits = datasets.load_digits(n_class=6)  # 取前六种数字图片，0-5
    data = digits.data  # data.shape=[1083,64]，表示1084张图片，每个图片8*8但是将图片表示为一个行向量
    label = digits.target  # 表示取出的1083个图片对应的数字
    n_samples, n_features = data.shape  # 图片数1083和每张图片的维度64
    return data, label, n_samples, n_features


def plot_embedding(result, label, title):  # 传入1083个2维数据，1083个标签，图表标题
    # x_min, x_max = np.min(result, 0), np.max(result, 0)  # 分别求出每一列最小值和最大值
    # data = (result - x_min) / (x_max - x_min)  # 将数据进行正则化，分母为数据的总长度，因此分子一定小于分母，生成的矩阵元素都是0-1区间内的
    data = result
    plt.figure(figsize=(10, 8))  # 创建一个画布
    # plt.figure(figsize=(12,8))  # 创建一个画布
    # central_kind = ['Caltech', 'CMU', 'KKI', 'Leuven', 'MaxMun', 'OHSU', 'Olin', 'Pitt', 'SBL', 'SDSU', 'Stanford', 'Trinity', 'UCLA', 'UM', 'USM', 'Yale', 'NYU']
    central_kind = ['Stanford','MaxMun','UM', 'USM','NYU']
    data = {}
    for central_kind_ in range(len(central_kind)):
        data_ = []
        for i in range(label.shape[0]):
            if label[i] == central_kind_:
                if result[i][0] < 70 and -70 <result[i][1] < 60: #original graph
                # if -50 <result[i][0] < 50 and -50 <result[i][1] < 50: #original graph
                    data_.append(result[i])
                # data_.append(result[i])

        data[central_kind_] = data_
    # print('data：',data[0])
        # print('data[center_kind]: ',data,': ',data[0])
    # print('data[center_kind]: ',data[0])
    # colors = ['red','blue','grey','yellow','red','blue','red','blue','red','blue','red','blue','red','blue','red','blue','red']
    # colors = ['c', '#000080', 'g', '#CD853F', '#FA8072', 'm', 'y', 'k', 'red','#008B8B','brown','#7FFF00','#FFD700','#FF00FF','yellow','#696969','blue']
    colors = ['black', 'red', 'green', '#FF00FF', 'blue']
    markers = [',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 'P', '*', 'h', '+', '*', 'x']
    for k, v in data.items():
        if k == 4:
            plt.scatter(np.array(v)[:, 0], np.array(v)[:, 1], s=40, marker='x', c=colors[k], label=central_kind[k])
            plt.legend(loc=2, bbox_to_anchor=(1.0,1.0),prop = {'size':8})
        else:
            plt.scatter(np.array(v)[:, 0], np.array(v)[:, 1], s=20, marker='x', c=colors[k], label=central_kind[k])
            plt.legend(loc=2, bbox_to_anchor=(1.0, 1.0), prop={'size': 8})
    # plt.xticks(())  # 不显示坐标刻度
    # plt.yticks(())
    plt.title(title)  # 设置标题
    # plt.savefig(str(1) + '_' + 't-SNE.png')
    plt.savefig(str(1) + '_' + 'transfer_t-SNE.png')
    plt.show()


"""主函数"""

def cross_val(adjs, labels):
    kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    zip_list = list(zip(adjs, labels))
    random.Random(0).shuffle(zip_list)
    adjs, labels = zip(*zip_list)
    adjs = np.array(adjs)
    labels = np.array(labels)
    return adjs, labels, kf.split(adjs, labels)

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
    model = SimSiam_CD(args).cuda()
    print(model)
    """获取测试集数据，分析测试集合中，标签的对应类别，大概意思测试集合有些标签，我们根据这些进行表征提取和显示"""
    # 观察与训练后的结果
    f = open('/home/idal-01/code/TD-learnableFilters/NRAC_L.pkl', 'rb')
    data, Labels = pickle.load(f)

    # labelsPath = 'results.xlsx'
    # prefix = '/home/idal-01/data/'
    # prefix_EATD = '/home/idal-01/data/EATD-Corpus/EATD-Corpus/EATD-Corpus/'
    # prefixCMDC = '/home/idal-01/data/CMDC/tmp/'

    # data, labels, sex = dataprogress(prefix, labelsPath)
    # data, labels = dataprogress2(prefix_EATD)
    # data, labels = dataprogress3(prefixCMDC)
    print('Load data finished' + '\n')
    print('The number of patient: ', len(np.where(np.array(Labels) == 1)[0]))
    print('The number of NC: ', len(np.where(np.array(Labels) == 0)[0]))
    trainList_dataloader, testList_dataloader, valList_dataloader, toPersonList, CE_weights = total_dataloader(args,
                                                                                                               data,
                                                                                                               Labels)
    # data, label, n_samples, n_features = get_data()  # data种保存[1083,64]的向量
    data, label = get_adj_label_data()  # data种保存[1083,64]的向量


    # data_1 = torch.transpose(data, 1, 0)
    # data = data_1[0]
    print('data.shape: ',data.shape,' type(label): ',type(label))
    print('label: ',label)
    data = data.reshape(data.shape[0],-1)
    data = data.cpu().detach().numpy()
    # tsne = TSNE(n_components=2, init='pca',
    tsne = TSNE(n_components=2, init='pca', random_state=0)  # n_components将64维降到该维度，默认2；init设置embedding初始化方式，可选pca和random，pca要稳定些
    t0 = time()  # 记录开始时间
    result = tsne.fit_transform(data)  # 进行降维，[1083,64]-->[1083,2]
    # plot_embedding(result, label, 't-SNE embedding of Original Graph (time %.2fs)' % (time() - t0))  # 显示数据
    # plot_embedding(result, label, 't-SNE embedding of Original Graph') #显示数据
    plot_embedding(result, label, 't-SNE embedding of Generated Graph') #显示数据

if __name__ == '__main__':
    main()