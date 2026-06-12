import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from config.args import parser
import random
import shap
shap.initjs()

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(1)

data_train = pd.read_excel("train.xlsx")
data_test = pd.read_excel("test.xlsx")

scaler = StandardScaler()

x_train_np = scaler.fit_transform(data_train.iloc[:,1:].to_numpy())
y_train_np = data_train.iloc[:,0].to_numpy().reshape(-1, 1)
x_test_np = scaler.transform(data_test.iloc[:,1:].to_numpy())
y_test_np = data_test.iloc[:,0].to_numpy().reshape(-1, 1)
feature_names = data_train.iloc[:, 1:].columns.tolist()

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


model = torch.load('model_saved/model.pth')


def model_predict(x):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32)
        pred_tensor = model(x_tensor)
        pred_np = pred_tensor.numpy()
    return pred_np


background_data = x_train_np[np.random.choice(x_train_np.shape[0], 2451, replace=False)]

explainer = shap.KernelExplainer(
    model=model_predict,
    data=background_data,
    link="identity"
)

shap_values = explainer.shap_values(
    X=x_test_np,
    nsamples=512
)


shap_values_np = np.array(shap_values).squeeze()
mean_abs_shap = np.abs(shap_values_np).mean(axis=0)
print(mean_abs_shap)

shap.summary_plot(
    shap_values_np,
    features=x_test_np,
    feature_names=feature_names,
    plot_type="dot",
    show=False,
    # color_bar=False
    cmap="viridis",
    plot_size=(8, 6)
)
plt.tight_layout()
plt.show()

shap.summary_plot(
    shap_values_np,
    features=x_test_np,
    feature_names=feature_names,
    plot_type="bar",
    show=False,
    plot_size=(8, 6)
)
plt.tight_layout()
plt.show()