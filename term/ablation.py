import os
import math
import numpy as np
import pandas as pd
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from config.args import parser

args = parser.parse_args()
random.seed(args.seed)
torch.manual_seed(args.seed)


class Net(nn.Module):
    def __init__(self, n_feature=9, n_hidden=17, n_output=1, w=3):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)
        nn.init.kaiming_normal_(self.hidden1.weight)
        self.hidden2 = torch.nn.Linear(17, 17)
        nn.init.kaiming_normal_(self.hidden2.weight)
        self.hidden3 = torch.nn.Linear(17, 1)
        nn.init.kaiming_normal_(self.hidden3.weight)

    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.hidden3(x)
        x = F.relu(x)
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


def train(train_dataset, model, criterion, optimizer, epoch, epoch_loss, args):

    model.train()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True)
    for i, (features, target) in enumerate(train_loader):
        output = model(features)

        if i <= (len(train_dataset) // args.batch_size) - 1:
            args.n_feature = len(train_dataset) - args.batch_size
        else:
            args.n_feature = i * args.batch_size

        loss = criterion(output, target.view(-1,1)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * features.size(0)
    return epoch_loss


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
data_train_X = torch.tensor(scaler.fit_transform(data_train.iloc[:, 1:]), dtype=torch.float32)
data_train_Y = torch.tensor(data_train.iloc[:, 0].to_numpy(), dtype=torch.float32)
train_dataset = torch.utils.data.TensorDataset(data_train_X, data_train_Y)

data_test = pd.read_excel(testdir)
data_test_X = torch.tensor(scaler.transform(data_test.iloc[:, 1:]), dtype=torch.float32)
data_test_Y = torch.tensor(data_test.iloc[:, 0].to_numpy(), dtype=torch.float32)
test_dataset = torch.utils.data.TensorDataset(data_test_X,data_test_Y)

losses = []
for epoch in range(args.start_epoch, args.epochs):
    epoch_loss = 0
    lr_setter(optimizer, epoch, args)
    epoch_loss = train(train_dataset, model, criterion_train, optimizer, epoch, epoch_loss, args)
    epoch_loss /= len(train_dataset)
    losses.append(epoch_loss)

# torch.save(model, 'model_saved/model_ablation.pth')

y_pred = model(data_train_X).detach().numpy()
y_true = data_train_Y.detach().numpy()
r2_train = r2_score(y_true, y_pred)
MSE_train = mean_squared_error(y_true, y_pred)

y_pred_test = model(data_test_X).detach().numpy()
y_true_test = data_test_Y.detach().numpy()
r2_test = r2_score(y_true_test, y_pred_test)
MSE_test = mean_squared_error(y_true_test, y_pred_test)
