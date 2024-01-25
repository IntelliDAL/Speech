import argparse
import os
import pickle
import random
import logging
from logging import handlers
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from torch.backends import cudnn
from drawing.drawing import plot_confusion_matrix
from method import train, evaluate, evaluate_val
from model.model_transformer import InfoAttentionClassifier
from qwer.crateModel import CRATE_base_demo, CRATE_base, CRATE_base_21k
from untils import dataprogress, total_dataloader, saveList, mean_sd, dataprogress2, dataprogress3
import datetime
from pathlib import Path
import matplotlib as mpl

fpath = Path(mpl.get_data_path(), "fonts/ttf/cmb10.ttf")


def init_seeds(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def run(args):
    init_seeds(1)
    print(args)
    # labelsPath = 'results.xlsx'
    #
    # prefix = '/home/idal-01/data/'
    # prefix_EATD = '/home/idal-01/data/EATD-Corpus/EATD-Corpus/EATD-Corpus/'
    # prefixCMDC = '/home/idal-01/data/CMDC/tmp/'

    f = open('/home/idal-01/code/TD-learnableFilters/NRAC_L.pkl', 'rb')
    # f = open("/home/idal-01/haoyong/FD-learnableFilters/Ori_NRAC_L.pkl", 'rb')
    # f = open('/home/eric/code_for_ywj/FD-learnableFilters/FD_learnableFilters/NRAC_L.pkl', 'rb')
    # f = open('/home/idal-01/code/TD-learnableFilters/CMDC.pkl', 'rb')
    # f = open('/home/eric/code_for_ywj/FD-learnableFilters/FD_learnableFilters/SDCL/NRAC.pkl', 'rb')
    data, Labels = pickle.load(f)

    # data, labels, sex = dataprogress(prefix, labelsPath)
    # data, Labels = dataprogress2(prefix_EATD)
    # data, Labels = dataprogress3(prefixCMDC)
    print('Load data finished' + '\n')
    print('The number of patient: ', len(np.where(np.array(Labels) == 1)[0]))
    print('The number of NC: ', len(np.where(np.array(Labels) == 0)[0]))
    trainList_dataloader, testList_dataloader, valList_dataloader, toPersonList, CE_weights = total_dataloader(args, data, Labels)

    results, results_segments, aucs, tprs = [], [], [], []
    base_fpr = np.linspace(0, 1, 100)

    for train_idx in range(5):
        print("build the models ...")
        # Info_model = InfoAttentionClassifier(args).cuda()
        # Info_model = CRATE_base(args).cuda()
        # Info_model = CRATE_base_21k(args).cuda()
        Info_model = CRATE_base_demo(args).cuda()
        total_info = sum([param.nelement() for param in Info_model.parameters()])
        print('Number of parameter: %.6f' % total_info)
        RETURNED_MODEL, LastEpModel = train(args, trainList_dataloader[train_idx], Info_model, testList_dataloader[train_idx], CE_weights[train_idx], train_idx)
        print('Testing..........' + '\n')
        result_R = evaluate_val(testList_dataloader[train_idx], RETURNED_MODEL)
        print(result_R)

        results_segments.append(result_R)
        result_L = evaluate_val(testList_dataloader[train_idx], LastEpModel)
        torch.save(LastEpModel.state_dict(), "/home/idal-01/code/TD-learnableFilters/trained_model/ward/%d_Lastmodel.ckpt" % train_idx)
        print(result_L)

        result, fpr, tpr, = evaluate(testList_dataloader[train_idx], RETURNED_MODEL, toPersonList[train_idx])
        print(result)
        torch.save(RETURNED_MODEL.state_dict(), "/home/idal-01/code/TD-learnableFilters/trained_model/ward/%d.ckpt" % train_idx)
        results.append(result)
        aucs.append(result['auc'])

        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)
        del Info_model

    current_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    "模型结果存储"
    print(results)
    print(results_segments)

    # ------------------------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------------------------
    # saveList(results, 'result_voice_' + str(current_date) + '.pickle')
    # Info_model = InfoAttentionClassifier(args).cuda()
    Info_model = CRATE_base_demo(args).cuda()
    name = str(args.experiment) + '_' + str(current_date)
    mean_sd(name, results, args, Info_model)

    matrix_ori = np.zeros((2, 2)).astype(int)
    for itm in results:
        matrix_ori = np.array(itm['matrix']) + matrix_ori
    print(matrix_ori)
    plot_confusion_matrix(cm=matrix_ori, normalize=False, target_names=['NC', 'Patient'], title="Confusion Matrix")
    plt.savefig('./drawing/confusionMatrix' + str(args.experiment) + '.pdf')

    plt.figure(figsize=(5, 5))
    plt.axes().set_aspect('equal', 'datalim')
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)
    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std
    plt.plot(base_fpr, mean_tprs, 'b')
    print(metrics.auc(base_fpr, mean_tprs))
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xticks(size=12, weight='bold', font=fpath)
    plt.yticks(size=12, weight='bold', font=fpath)
    plt.ylabel('True Positive Rate', weight='bold', font=fpath)
    plt.xlabel('False Positive Rate', weight='bold', font=fpath)
    plt.savefig('./drawing/ROC_ward.pdf')
    plt.show()
    plt.close()


def main():
    init_seeds(3407)
    fs = 16000
    # cw_len = 10.3
    cw_len = 6.46
    wlen = int(fs * cw_len)
    filter_num = 128
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate_1', type=float, default=0.0005)
    parser.add_argument('--input_dim', type=int, default=wlen)
    parser.add_argument('--fs', type=int, default=16000)
    parser.add_argument('--audio_length', type=int, default=cw_len)
    parser.add_argument('--input_channels', type=int, default=1)
    parser.add_argument('--output_channels', type=int, default=filter_num)
    parser.add_argument('--hidden_channels', type=int, default=filter_num)
    parser.add_argument('--skip_channels', type=int, default=filter_num)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--dilation', type=int, default=2)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--aptim', type=str, default='adam')
    # random, mel_scale, ERB, Bark, Truncated Mel, Truncated U-Hz
    parser.add_argument('--initializer', type=str, default='random')
    parser.add_argument('--experiment', type=str, default='FD-supervised_Learning')
    parser.add_argument('--label_smooth', default=0.0, type=float, metavar='L',  help='label smoothing coef')
    # FD滤波器
    parser.add_argument('--filter_size', type=int, default=513)
    parser.add_argument('--filter_num', type=int, default=filter_num)
    parser.add_argument('--frame_num', type=int, default=640)
    parser.add_argument('--NFFT', type=int, default=1024)
    parser.add_argument('--sigma_coff', type=float, default=0.0015)
    "trani or qwer"
    parser.add_argument('--flag', type=bool, default=False)
    args = parser.parse_args()
    Info_model = InfoAttentionClassifier(args).cuda()
    # Info_model = CRATE_base(args).cuda()
    # Info_model = CRATE_base_demo(args).cuda()
    print(Info_model)
    total_info = sum([param.nelement() for param in Info_model.parameters()])
    print('Number of parameter: %.6f' % total_info)
    run(args)


if __name__ == '__main__':
    init_seeds(1)
    device_index = 0  # 目标GPU的索引
    torch.cuda.set_device(device_index)
    torch.set_default_dtype(torch.float32)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    main()