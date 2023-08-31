import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from augmentation import augmentation
import torch.distributed as dist

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, args, base_transform):
        self.base_transform = base_transform
        # self.base_transform.shuffle = True
        self.sample_rate = args.fs
        self.audio_length = args.audio_length

    def __call__(self, x):
        # only one of transformation applied
        m, n = x.shape
        augmentationType_up = []
        augmentationType_down = []
        augmentedDate = []
        # 确定transformation Index
        for item in x:
            # First augmentation
            up = self.base_transform.transform_index
            q = self.base_transform(item, sample_rate=self.sample_rate).reshape(1, n)

            # Second augmentation
            k = self.base_transform(item, sample_rate=self.sample_rate).reshape(1, n)
            down = self.base_transform.transform_index

            # 保证两次数据增强的方法不同
            while up == down:
                k = self.base_transform(item, sample_rate=self.sample_rate).reshape(1, n)
                down = self.base_transform.transform_index
                if up != down:
                    break
            data = np.concatenate([q, k], axis=0)

            augmentedDate.append(data)
            augmentationType_down.append(down)
            augmentationType_up.append(up)
        return_data = np.array(augmentedDate)
        print(augmentationType_up)
        print(augmentationType_down)
        return return_data


class datasets(Dataset):
    def __init__(self, x, args):
        self.windows_Length = args.audio_length
        self.x = x
        self.args = args

    def __getitem__(self, idx):
        return_x = create_batches_rnd(self.x[idx], self.windows_Length, self.args)
        return return_x

    def __len__(self):
        return len(self.x)


def create_batches_rnd(signal, wlen, args):
    """
    random
    每次选取若干个样本，
    每个样本随机采样两次，两个片段。
    """
    number = 2
    segmentLength = int(args.fs * wlen)
    sig_batch = np.zeros([number, segmentLength])
    for i in range(number):
        snt_len = signal.shape[0]
        snt_beg = np.random.randint(snt_len - segmentLength - 1)
        snt_end = snt_beg + segmentLength
        sig_batch[i, :] = signal[snt_beg:snt_end]

    sig_batch = torch.from_numpy(sig_batch)
    sig_batch = datafft(sig_batch, args)
    return sig_batch


def Dataloader(args, Total_Data):
    # transformed_audio = TwoCropsTransform(args, augmentation)(Total_Data)
    # transformed_audio = torch.from_numpy(Total_Data).cuda()
    train_dataset = datasets(Total_Data, args)
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    return train_dataset_loader


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


def datafft(signal, args):
    # path = '/home/ubnn/PycharmProjects/LearnableFilter_Voice/drawing/Interpretable_FIg/'
    sample_rate = 16000
    NFFT = args.NFFT
    win_length, hop_length = int(0.025 * sample_rate), int(0.01 * sample_rate)
    # 计算短时傅里叶变换(STFT)
    window = torch.hann_window(win_length)
    stft = torch.stft(input=signal, n_fft=NFFT, window=window, hop_length=hop_length, win_length=win_length, normalized=True, center=False, return_complex=True)

    # 计算功率谱
    # mag_frames = np.abs(stft)
    power_spec = torch.square(torch.abs(stft))

    return power_spec