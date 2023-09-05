import argparse
import os
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
from model.model_main import InfoAttentionClassifier
from untils import dataprogress, total_dataloader, saveList, mean_sd, dataprogress_dataset3800
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

    # prefix = '/home/idal-01/data/'
    prefix = '/home/idal-01/data/dataset3800/'
    labelsPath = 'results.xlsx'
    # data, labels, sex = dataprogress(prefix, labelsPath)
    data, labels, age = dataprogress_dataset3800(prefix)
    print('Load data finished' + '\n')
    print('The number of patient: ', len(np.where(np.array(labels) == 1)[0]))
    print('The number of NC: ', len(np.where(np.array(labels) == 0)[0]))
    trainList_dataloader, testList_dataloader, valList_dataloader, toPersonList, CE_weights = total_dataloader(args, data, labels)

    results, results_segments, aucs, tprs = [], [], [], []
    base_fpr = np.linspace(0, 1, 100)

    for train_idx in range(5):
        print("build the models ...")
        Info_model = InfoAttentionClassifier(args).cuda()
        total_info = sum([param.nelement() for param in Info_model.parameters()])
        print('Number of parameter: %.6f' % total_info)
        ckpt_path = None
        # ckpt_path = '/home/idal-01/code/TD-learnableFilters/'+str(train_idx)+'.ckpt'
        if ckpt_path is not None:
            print('Loading pre-trained model' + '\n')
            pretrained_dict = torch.load(ckpt_path)
            model_dict = Info_model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            Info_model.load_state_dict(model_dict)

        RETURNED_MODEL, LastEpModel = train(args, trainList_dataloader[train_idx], Info_model, testList_dataloader[train_idx], CE_weights[train_idx], train_idx)
        print('Testing..........' + '\n')
        result_R = evaluate_val(testList_dataloader[train_idx], RETURNED_MODEL)
        print(result_R)

        results_segments.append(result_R)
        result_L = evaluate_val(testList_dataloader[train_idx], LastEpModel)
        torch.save(RETURNED_MODEL.state_dict(), "%d_Info_model.ckpt" % train_idx)
        print(result_L)

        result, fpr, tpr, = evaluate(testList_dataloader[train_idx], RETURNED_MODEL, toPersonList[train_idx])
        print(result)
        torch.save(RETURNED_MODEL.state_dict(), "%d.ckpt" % train_idx)
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
    Info_model = InfoAttentionClassifier(args).cuda()
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
    init_seeds(1)
    fs = 16000
    cw_len = 10.3
    wlen = int(fs * cw_len)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=200)
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
    parser.add_argument('--n_blocks', type=int, default=4)
    parser.add_argument('--dilation', type=int, default=2)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--aptim', type=str, default='adam')
    parser.add_argument('--initializer', type=str, default='LowAndHighFreq')
    parser.add_argument('--experiment', type=str, default='FD-filters-attempt')

    # TD滤波器
    parser.add_argument('--filter_size', type=int, default=513)
    parser.add_argument('--filter_num', type=int, default=64)
    parser.add_argument('--frame_num', type=int, default=1024)
    parser.add_argument('--NFFT', type=int, default=1024)
    parser.add_argument('--sigma_coff', type=float, default=0.0015)
    args = parser.parse_args()
    Info_model = InfoAttentionClassifier(args).cuda()
    print(Info_model)
    run(args)


if __name__ == '__main__':
    init_seeds(1)
    device_index = 1  # 目标GPU的索引
    torch.cuda.set_device(device_index)
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    main()
