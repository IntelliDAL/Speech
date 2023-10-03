import argparse
import os
import pickle
import random
import numpy as np
from torch.backends import cudnn
import torch.distributed as dist
from SSL_CD import SimSiam, codeDiscriminator
import torch
import torch.multiprocessing as mp
from methods_CD import train


def init_seeds(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def Run(args, data):
    for train_idx in range(1):
        print("build the models ...")
        model = SimSiam(args).cuda()
        Discriminator = codeDiscriminator(args).cuda()
        print(model)
        trained_model = train(args, data, model, Discriminator)
        # 保存EncoderI and EncoderT
        torch.save(trained_model.encoder.EncoderI.state_dict(),'/home/idal-01/code/TD-learnableFilters/self-supervised/pre-trained_model/Encoder_'+str(args.num_epochs)+'_I.pt')
        torch.save(trained_model.encoder.EncoderT.state_dict(),'/home/idal-01/code/TD-learnableFilters/self-supervised/pre-trained_model/Encoder_'+str(args.num_epochs)+'_T.pt')
        print('Finishing..........')


# ____________________________________________________________________________________________________________________
def main():
    init_seeds(1)
    fs = 16000
    cw_len = 6.46  # s
    wlen = int(fs * cw_len)
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument('--log_step', type=int, default=1, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=10, help='step size for saving trained models')
    parser.add_argument('--num_epochs', type=int, default=100)
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
    # f = open('/home/idal-01/code/TD-learnableFilters/self-supervised/NRAC_segment.pkl', 'rb')
    f = open('/home/idal-01/code/TD-learnableFilters/self-supervised/NRAC.pkl', 'rb')
    data = pickle.load(f)
    print("Load data finished" + '\n')
    init_seeds(1)
    device_index = 1  # 目标GPU的索引
    torch.cuda.set_device(device_index)
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    # 创建进程组
    Run(args, data)


if __name__ == '__main__':
    main()
