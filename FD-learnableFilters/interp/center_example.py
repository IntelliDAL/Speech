import math
import os
import random
import argparse

import librosa
import librosa.display
import numpy as np
import pylab as pl
import torch
import torchaudio
from matplotlib import pyplot as plt, ticker
from matplotlib.pyplot import plot
from numpy import conj
from numpy.fft import fftshift
from scipy.fft import fft
from scipy.stats import norm
import matplotlib as mpl
from pathlib import Path

from model.model_main import InfoAttentionClassifier

fpath = Path(mpl.get_data_path(), "fonts/ttf/cmb10.ttf")
from model.Filter_files.impulse_responses import gabor_filters
from scipy import signal


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def hz_to_erb(frequency_hz):
    return 21.4 * np.log10(4.37 * frequency_hz / 1000 + 1)


def erb_to_hz(erb_value):
    return 1000 * ((10 ** (erb_value / 21.4)) - 1) / 4.37 / 1000


if __name__ == '__main__':
    fs = 16000
    cw_len = 6.46  # s
    wlen = int(fs * cw_len)
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument('--log_step', type=int, default=1, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=10, help='step size for saving trained models')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--lr_D', type=float, default=0.00005)
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
    parser.add_argument('--worker_size', type=int, default=2)
    parser.add_argument('--initializer', type=str, default='random')  # mel_scale
    # TD滤波器
    parser.add_argument('--filter_size', type=int, default=513)
    parser.add_argument('--filter_num', type=int, default=64)
    parser.add_argument('--frame_num', type=int, default=640)
    parser.add_argument('--NFFT', type=int, default=1024)
    parser.add_argument('--sigma_coff', type=float, default=0.0015)
    args = parser.parse_args()
    Batchsize = args.batch_size
    print(args)
    Info_model = InfoAttentionClassifier(args).cuda()
    ckpt_path = '/home/idal-01/code/TD-learnableFilters/self-supervised/pre-trained_model/Encoder_200_I.pt'
    if ckpt_path is not None:
        print('Loading pre-trained model' + '\n')
        pretrained_dict_ = torch.load(ckpt_path)
        # pretrained_dict = OrderedDict()
        # for key in pretrained_dict_:
        #     pretrained_dict[key[14:]] = pretrained_dict_[key]
        # model_dict = Info_model.state_dict()
        model_dict = Info_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict_.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        # Info_model.load_state_dict(model_dict, strict=False)
        Info_model.load_state_dict(model_dict, strict=False)

    centers_learnable = (Info_model.AudioSpectral.FD_Learned_Filters.w[:, 0]).tolist()
    centers_learnable = [x*8 for x in centers_learnable]

    #  获取到kernel参数
    kernel_size = 401
    filtersNum = 64
    """Mel初始化"""
    mel_filters = torchaudio.functional.melscale_fbanks(n_freqs=1024 // 2 + 1, f_min=0, f_max=8000, n_mels=64, sample_rate=16000)
    mel_scale = torchaudio.transforms.MelScale()
    mel_filters = mel_filters.transpose(1, 0)
    sqrt_filters = torch.sqrt(mel_filters)
    center_frequencies = torch.argmax(sqrt_filters, dim=1)
    FIN = np.linspace(0, 8000, 513)
    Mel_center_frequencies = FIN[center_frequencies] / 1000
    "Mel初始化——————3k"
    mel_filters_truncate = torchaudio.functional.melscale_fbanks(n_freqs=1024 // 2 + 1, f_min=0, f_max=3000, n_mels=64,
                                                                 sample_rate=16000)
    mel_scale_truncate = torchaudio.transforms.MelScale()
    mel_filters_truncate = mel_filters_truncate.transpose(1, 0)
    sqrt_filters_truncate = torch.sqrt(mel_filters_truncate)
    center_frequencies_truncate = torch.argmax(sqrt_filters_truncate, dim=1)
    FIN_truncate = np.linspace(0, 8000, 513)
    Mel_center_truncate = FIN_truncate[center_frequencies_truncate] / 1000

    """随机初始化"""
    # ======================
    center_freqz = torch.linspace(0, 8, filtersNum)
    """截断到3k范围以内"""
    center_3k = torch.linspace(0, 3, filtersNum)
    """高斯"""
    # --------------------------------------
    mu = 0
    sigma = 1
    random_value = norm.rvs(loc=mu,scale=sigma,size=128)
    bounded_value = np.clip(random_value, 0, 8)
    # ---------------------------------------------------
    "gammatone"
    num_values = 64
    lowest_freq = 0  # Hz
    highest_freq = 8000  # Hz
    # Convert frequency to ERB units
    lowest_erb = hz_to_erb(lowest_freq)
    highest_erb = hz_to_erb(highest_freq)
    # Generate uniform values in ERB units
    erb_values = np.linspace(lowest_erb, highest_erb, num_values)
    # Convert ERB units back to frequency
    frequency_values = erb_to_hz(erb_values)
    # --------------------------------------------------
    XX = np.arange(filtersNum)
    plt.close()
    fig = plt.figure()

    plt.scatter(XX, Mel_center_frequencies, s=15, c='#26a0da', label='MEL F1=0.832')
    plt.scatter(XX, frequency_values, s=15, c='#ff9900', label='ERB F1=0.835')
    plt.scatter(XX, Mel_center_truncate, s=15, c='c', label='truncated MEL F1=0.781', linestyle='--')
    # plt.scatter(XX, center_freqz.numpy(), s=15, c='#cc0000', label='Uniform F1=0.792')
    plt.scatter(XX, center_3k.numpy(), s=15, c='#33cc33', label='truncated uniform F1=0.849')
    plt.scatter(XX, np.array(centers_learnable), s=15, c='blue', label='learnable F1=0.887')
    plt.xlabel('Filter index', weight='bold', color='k', font=fpath)
    plt.ylabel('Center frequency (KHz)', weight='bold', color='k', font=fpath)
    plt.grid(True, ls='--')
    plt.ylim([-0.01, 8.1])
    plt.xlim([-0.01, 64.1])
    plt.legend(loc="upper left", markerscale=1, fontsize=8, frameon=False)
    plt.savefig('diff_center.pdf')
    plt.show()
