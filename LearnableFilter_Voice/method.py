import time
from collections import OrderedDict
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm

from drawing.drawing import plot_loss_curve
from untils import LabelVote


def train(args, train_dataset, AudioSpecFeat, validate_dataset, CE_weights, train_idx):
    criterion = nn.CrossEntropyLoss(weight=CE_weights, reduction='sum').cuda()
    optim = args.aptim
    Lr_1, Lr_2 = args.learning_rate_1, args.learning_rate_2
    if optim == 'sgd':
        optimizer1 = torch.optim.SGD((param for param in AudioSpecFeat.parameters() if param.requires_grad), lr=Lr_1, weight_decay=0.001)
    elif optim == 'adam':
        optimizer1 = torch.optim.Adam((param for param in AudioSpecFeat.parameters() if param.requires_grad), lr=Lr_1, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.001)
    elif optim == 'adamW':
        optimizer1 = torch.optim.AdamW((param for param in AudioSpecFeat.parameters() if param.requires_grad), lr=Lr_1, weight_decay=0.001)
    elif optim == 'RMSprop':
        optimizer1 = torch.optim.RMSprop((param for param in AudioSpecFeat.parameters() if param.requires_grad), lr=Lr_1, alpha=0.95, eps=1e-8, weight_decay=0.001)

    optim_scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer=optimizer1, step_size=10, gamma=0.9)
    RETURNED_AudioSpecFeat = AudioSpecFeat
    patience, Length, Plot_loss, best_val_acc = 10, len(train_dataset), [], 0
    start_time = time.time()
    for epoch in range(args.num_epochs):
        Loop = tqdm(enumerate(train_dataset), total=Length, colour='blue', leave=True)
        Loop.set_description(f'Epoch [{epoch}/{args.num_epochs}]')
        AudioSpecFeat.train()
        LossList, running_loss = [], 0.0
        for batch_idx, batch_data in Loop:
            train_batchSamples, train_batchLabels = batch_data['x'], batch_data['label']
            outputs = AudioSpecFeat(train_batchSamples.type(torch.FloatTensor).cuda())
            one_hot_labels = Variable(torch.zeros(args.batch_size, 2).scatter_(1, train_batchLabels.view(-1, 1), 1).cuda())
            Loss_classification = criterion(outputs, one_hot_labels)

            AudioSpecFeat.zero_grad()

            Loss_classification.backward()

            optimizer1.step()

            optim_scheduler1.step()

            running_loss += Loss_classification.item()
            Loop.set_postfix(Loss_classification=Loss_classification.item())
            LossList.append(Loss_classification.item())
        print(running_loss / Length)
        if validate_dataset is not None:
            val_result = evaluate_val(validate_dataset, AudioSpecFeat)
            print(val_result)
            if val_result['acc'] >= best_val_acc:
                RETURNED_AudioSpecFeat = AudioSpecFeat
                best_val_acc = val_result['acc']

    plot_loss_curve(Plot_loss, train_idx)
    plt.savefig('./drawing/Loss_curve' + str(train_idx)+'.png')
    RETURN_MODEL = RETURNED_AudioSpecFeat
    MODEL = AudioSpecFeat
    print('Running time is :', time.time()-start_time)
    return RETURN_MODEL, MODEL

def evaluate(dataset, MODEL, toPersonList):
    AudioSpecFeat = MODEL.eval()
    with torch.no_grad():
        target, Labels_, probs = [], [], []
        total_rewards = 0
        total_labels = 0
        for m, data_test in enumerate(dataset):
            V_data = Variable(data_test['x'].to(torch.float32), requires_grad=False).cuda()
            true_labels = data_test['label'].long().numpy()

            ypred = AudioSpecFeat(V_data.type(torch.FloatTensor).cuda())

            _, predict_labels = torch.max(ypred, dim=1)
            preds = predict_labels.cpu().data.numpy()
            prob = ypred[:, 1].cpu().data.numpy()

            Length = len(true_labels)
            rewards = [1 if predict_labels[j] == true_labels[j] else 0 for j in range(Length)]
            total_rewards += np.sum(rewards)
            total_labels += Length

            Labels_.extend(true_labels)
            target.extend(preds)
            probs.extend(prob)
        accuracy = total_rewards / total_labels
        auc = metrics.roc_auc_score(Labels_, probs)
        print('accuracy', accuracy)
        print('auc', auc)
        true_labels, true_targets, true_probs, true_sex = [], [], [], []
        for idx, per in enumerate(toPersonList):
            if idx == 0:
                start = 0
            end = start + per
            target_split = target[start:end]
            true_target = LabelVote(target_split)
            true_targets.append(true_target)
            labels_split = Labels_[start:end]
            true_label = LabelVote(labels_split)
            true_labels.append(true_label)
            probs_split = probs[start:end]
            true_prob = np.mean(probs_split)
            true_probs.append(true_prob)
            start = end

        fpr, tpr, threshold = metrics.roc_curve(true_labels, true_probs)
        roc_auc = metrics.auc(fpr, tpr)
        print('Target', target)
        print('True..', Labels_)
        print('-----' * 5)
        print('True_labels.', true_labels)
        print('True_targets', true_targets)

        Result = {'prec': metrics.precision_score(true_labels, true_targets),
                  'recall': metrics.recall_score(true_labels, true_targets),
                  'acc': metrics.accuracy_score(true_labels, true_targets),
                  'F1_w': metrics.f1_score(true_labels, true_targets, average='weighted'),
                  'F1_micro': metrics.f1_score(true_labels, true_targets, average='micro'),
                  'F1_macro': metrics.f1_score(true_labels, true_targets, average='macro'),
                  'auc': roc_auc,
                  'matrix': confusion_matrix(true_labels, true_targets)}
    return Result, fpr, tpr


def evaluate_val(dataset, MODEL):
    AudioSpecFeat = MODEL.eval()
    with torch.no_grad():
        target, Labels_ = [], []
        for m, data_test in enumerate(dataset):
            V_data = Variable(data_test['x'].to(torch.float32), requires_grad=False).cuda()
            true_labels = data_test['label'].long().numpy()

            ypred = AudioSpecFeat(V_data.type(torch.FloatTensor).cuda())

            _, predict_labels = torch.max(ypred, dim=1)
            preds = predict_labels.cpu().data.numpy()
            Labels_.extend(true_labels)
            target.extend(preds)
        print('True_labels.', Labels_)
        print('True_targets', target)
        Result = \
            {
                'prec': metrics.precision_score(Labels_, target),
                'recall': metrics.recall_score(Labels_, target),
                'acc': metrics.accuracy_score(Labels_, target),
                'F1_w': metrics.f1_score(Labels_, target, average='weighted'),
                'F1_micro': metrics.f1_score(Labels_, target, average='micro'),
                'F1_macro': metrics.f1_score(Labels_, target, average='macro'),
                'matrix': confusion_matrix(Labels_, target)
            }
    return Result


def load_parallal_model(model, pretrain_dir):
    state_dict_ = torch.load(pretrain_dir, map_location='cuda:0')
    print('loaded pretrained weights form %s !' % pretrain_dir)
    state_dict = OrderedDict()

    for key in state_dict_:
        if key.startswith('module') and not key.startswith('module_list'):
            state_dict[key[7:]] = state_dict_[key]
        else:
            state_dict[key] = state_dict_[key]


    model_state_dict = model.state_dict()
    for key in state_dict:
        if key in model_state_dict:
            if state_dict[key].shape != model_state_dict[key].shape:
                print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
                    key, model_state_dict[key].shape, state_dict[key].shape))
                state_dict[key] = model_state_dict[key]
        else:
            print('Drop parameter {}.'.format(key))
    for key in model_state_dict:
        if key not in state_dict:
            print('No param {}.'.format(key))
            state_dict[key] = model_state_dict[key]
    model.load_state_dict(state_dict, strict=False)

    return model
