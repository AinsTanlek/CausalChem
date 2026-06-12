import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.autograd import Variable

import os
import random
import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from config.args import parser
import loss_updating as loss_expect
import matplotlib.pyplot as plt
import warnings


best_acc1 = 0
args = parser.parse_args()


class Net(nn.Module):
    def __init__(self, n_feature=9, n_hidden=17, n_output=1, w=3):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)
        nn.init.kaiming_normal_(self.hidden1.weight)
        self.hidden2 = torch.nn.Linear(17, 17)
        nn.init.kaiming_normal_(self.hidden2.weight)
        self.hidden3 = torch.nn.Linear(17, 1)
        nn.init.kaiming_normal_(self.hidden3.weight)
        # self.hidden4 = torch.nn.Linear(10, 1)
        # nn.init.kaiming_normal_(self.hidden4.weight)
        # self.hiddens = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for i in range(w)])
        # for m in self.hiddens:
        #     nn.init.kaiming_normal_(m.weight)

        # self.predict = torch.nn.Linear(n_hidden, n_output)
        # nn.init.kaiming_normal_(self.predict.weight)
        self.register_buffer('pre_features', torch.zeros(args.n_feature, args.feature_dim))
        self.register_buffer('pre_weight', torch.ones(args.n_feature, 1))

    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.hidden3(x)
        x = F.relu(x)
        # x = self.hidden4(x)
        # x = F.relu(x)
        return x


def lr_setter(optimizer, epoch, args, bl=False):

    lr = args.lr
    if bl:
        lr = args.lrbl * (0.1 ** (epoch // (args.epochb * 0.5)))
    else:
        if args.cos:
            lr *= ((0.01 + math.cos(0.5 * (math.pi * epoch / args.epochs))) / 1.01)
        else:
            if epoch >= args.epochs_decay[0]:
                lr *= 0.1
            if epoch >= args.epochs_decay[1]:
                lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def weight_learner(cfeatures, pre_features, pre_weight1, args, global_epoch=0, iter=0):

    softmax = nn.Softmax(0)
    weight = Variable(torch.ones(cfeatures.size()[0], 1))
    weight.requires_grad = True
    cfeaturec = Variable(torch.FloatTensor(cfeatures.size()))
    cfeaturec.data.copy_(cfeatures.data)
    all_feature = torch.cat([cfeaturec, pre_features.detach()], dim=0)
    optimizerbl = torch.optim.SGD([weight], lr=args.lrbl, momentum=0.9)

    for epoch in range(args.epochb):
        lr_setter(optimizerbl, epoch, args, bl=True)
        all_weight = torch.cat((weight, pre_weight1.detach()), dim=0)
        optimizerbl.zero_grad()

        COV, lossb = loss_expect.lossb_expect(all_feature, softmax(all_weight), args.num_f, args.sum)
        lossp = softmax(weight).pow(args.decay_pow).sum()
        lambdap = args.lambdap * max((args.lambda_decay_rate ** (global_epoch // args.lambda_decay_epoch)),
                                     args.min_lambda_times)
        lossg = lossb / lambdap + lossp
        if global_epoch == 0:
            lossg = lossg * args.first_step_cons

        lossg.backward(retain_graph=True)
        optimizerbl.step()

    # if global_epoch == 0 and iter < 10:
    #     pre_features = (pre_features * iter + cfeatures) / (iter + 1)
    #     pre_weight1 = (pre_weight1 * iter + weight) / (iter + 1)

    if cfeatures.size()[0] < pre_features.size()[0]:
        pre_features[:cfeatures.size()[0]] = pre_features[:cfeatures.size()[0]] * args.presave_ratio + cfeatures * (
                    1 - args.presave_ratio)
        pre_weight1[:cfeatures.size()[0]] = pre_weight1[:cfeatures.size()[0]] * args.presave_ratio + weight * (
                    1 - args.presave_ratio)

    else:
        pre_features = pre_features * args.presave_ratio + cfeatures * (1 - args.presave_ratio)
        pre_weight1 = pre_weight1 * args.presave_ratio + weight * (1 - args.presave_ratio)

    softmax_weight = softmax(weight)

    return COV, softmax_weight, pre_features, pre_weight1


def train(train_dataset, model, criterion, optimizer, epoch, epoch_loss, args):

    model.train()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True)

    for i, (features, target) in enumerate(train_loader):
        output = model(features)
        cfeatures = features

        if i <= (len(train_dataset) // args.batch_size) - 1:
            args.n_feature = len(train_dataset) - args.batch_size
        else:
            args.n_feature = i * args.batch_size

        pre_features = model.pre_features
        pre_weight = model.pre_weight

        if epoch >= args.epochp:
            COV, weight1, pre_features, pre_weight = weight_learner(cfeatures, pre_features,
                                                                    pre_weight, args, epoch, i)

        else:
            weight1 = Variable(torch.ones(cfeatures.size()[0], 1))

        model.pre_features.data.copy_(pre_features)
        model.pre_weight.data.copy_(pre_weight)

        loss = criterion(output, target.view(-1,1)).view(1, -1).mm(weight1).view(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * features.size(0)

    return COV, epoch_loss


def main():
    if args.concat:
        args.sum = False
    if args.dataset == "POEs discovery":
        args.feature_dim = 9
    elif args.dataset == "MOFs screening":
        args.feature_dim = 9
    else:
        args.feature_dim = 7

    """
    # modeling of 32 selectable features for Ligands optimization tasks
    if args.dataset == "Ligands optimization":
        args.feature_dim = 32  
    """

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    main_worker(args)


def main_worker(args):
    global best_acc1

    model = Net(n_feature=9, n_hidden=17, n_output=1, w=3)

    criterion = nn.MSELoss()
    criterion_train = nn.MSELoss(reduction='none')

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    traindir = os.path.join(args.data, 'train.xlsx')
    testdir = os.path.join(args.data, 'test.xlsx')

    scaler = StandardScaler()

    data_train = pd.read_excel(traindir)
    data_train_X = torch.tensor(scaler.fit_transform(data_train.iloc[:, 1:]))
    data_train_Y = torch.tensor(data_train.iloc[:, 0].to_numpy())
    train_dataset = torch.utils.data.TensorDataset(data_train_X, data_train_Y)

    data_test = pd.read_excel(testdir)
    data_test_X = torch.tensor(scaler.transform(data_test.iloc[:, 1:]))
    data_test_Y = torch.tensor(data_test.iloc[:, 0].to_numpy())
    test_dataset = torch.utils.data.TensorDataset(data_test_X,data_test_Y)

    losses = []
    for epoch in range(args.start_epoch, args.epochs):
        epoch_loss = 0
        lr_setter(optimizer, epoch, args)
        COV, epoch_loss = train(train_dataset, model, criterion_train, optimizer,
                                epoch, epoch_loss, args)

        epoch_loss /= len(train_dataset)
        losses.append(epoch_loss)

        if epoch == args.epochs - 1:
            covariance = COV
            print('DWR covariance:', covariance)

    # torch.save(model, 'model_saved/model.pth')

    y_pred = model(data_train_X).detach().numpy()
    y_true = data_train_Y.detach().numpy()

    r2_train = r2_score(y_true, y_pred).round(5)
    MSE_train = mean_squared_error(y_true, y_pred).round(5)

    y_pred_test = model(data_test_X).detach().numpy()
    y_true_test = data_test_Y.detach().numpy()

    r2_test = r2_score(y_true_test, y_pred_test).round(5)
    MSE_test = mean_squared_error(y_true_test, y_pred_test).round(5)

    print("MSE_train:", MSE_train)
    print("MSE_test:", MSE_test)


if __name__ == '__main__':
    main()
