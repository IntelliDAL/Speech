import matplotlib.pyplot as plt
import numpy as np
import itertools
import matplotlib as mpl
from pathlib import Path

fpath = Path(mpl.get_data_path(), "fonts/ttf/cmb10.ttf")


def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    if cmap is None:
        cmap = plt.get_cmap('Greens')
    # plt.figure(figsize=(6,5))
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.colorbar()
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        # plt.xticks(tick_marks, target_names, rotation=45)
        plt.xticks(tick_marks, target_names, size=12, weight='bold', font=fpath)
        plt.yticks(tick_marks, target_names, size=12, weight='bold', font=fpath)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=12, fontweight='bold', font=fpath)
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=12, fontweight='bold', font=fpath)
    # plt.tight_layout()
    plt.ylabel('True label', size=12, weight='bold', color='k', font=fpath)
    plt.xlabel('Predicted label', size=12, weight='bold', color='k', font=fpath)
    # plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))


def plot_loss_curve(y_train_loss, train_idx):
    x_train_loss = range(len(y_train_loss))  # loss的数量，即x轴
    plt.figure()
    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('iters')  # x轴标签
    plt.ylabel('loss')  # y轴标签
    # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
    # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
    plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
    plt.legend()
    plt.title('Loss curve')
    # plt.savefig('./drawing/Loss_curve' + str(train_idx)+'.png')
