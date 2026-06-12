import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from config.args import parser

args = parser.parse_args()

np.random.seed(42)
torch.manual_seed(42)

traindir = os.path.join(args.data, 'train.xlsx')
testdir = os.path.join(args.data, 'test.xlsx')

scaler = StandardScaler()

data_train = pd.read_excel(traindir)
data_test = pd.read_excel(testdir)
data = pd.concat([data_train,data_test],ignore_index=True)
data_scaler = torch.tensor(scaler.fit_transform(data.to_numpy()), dtype=torch.float32)
y_c = data_scaler[:, 0]


class BetaVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, beta=1.0):
        super(BetaVAE, self).__init__()
        self.beta = beta

        self.fc1 = nn.Linear(input_dim, 30)
        self.fc11 = nn.Linear(30,10)
        self.fc21 = nn.Linear(10, latent_dim)
        self.fc22 = nn.Linear(10, latent_dim)

        self.fc3 = nn.Linear(latent_dim, 10)
        self.fc31 = nn.Linear(10,30)
        self.fc4 = nn.Linear(30, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        h1 = torch.relu(self.fc11(h1))
        mu = self.fc21(h1)
        log_var = self.fc22(h1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        h3 = torch.relu(self.fc31(h3))
        return self.fc4(h3)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


def loss_function(recon_x, x, mu, log_var, beta=1.0):

    MSE = nn.MSELoss(reduction='sum')
    BCE = MSE(recon_x, x)
    KL_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return BCE + beta * KL_divergence, BCE, KL_divergence


input_dim = 10
latent_dim = 4
beta = 1
model = BetaVAE(input_dim, latent_dim, beta)
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_loader = DataLoader(TensorDataset(data_scaler), batch_size=32, shuffle=True)
epochs = 3000
total_losses = []
kl_losses = []
bce_losses = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    bce_loss = 0
    kl_loss = 0
    for batch_idx, (data_batch,) in enumerate(train_loader):
        data_batch = data_batch.squeeze(1)
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data_batch)
        loss, bce, kl = loss_function(recon_batch, data_batch, mu, log_var, beta)

        loss.backward()
        total_loss += loss.item()
        optimizer.step()

        bce_loss += bce.item()
        kl_loss += kl.item()

    bce_losses.append(bce_loss / len(train_loader.dataset))
    kl_losses.append(kl_loss / len(train_loader.dataset))
    total_losses.append(total_loss / len(train_loader.dataset))

    # print(f"Epoch {epoch + 1}/{epochs}, Total Loss: {total_loss / len(train_loader.dataset):.4f}, "
    # f"BCE Loss: {bce_loss / len(train_loader.dataset):.4f}, KL Loss: {kl_loss / len(train_loader.dataset):.4f}")

print(kl_losses)

fig, ax = plt.subplots(figsize=(10, 8))
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

plt.plot(range(epochs), kl_losses, label="KL_divergence")
plt.plot(range(epochs), bce_losses, label="Reconstruction_loss")
plt.plot(range(epochs), total_losses, label="Total_loss",)
plt.tight_layout()
plt.show()

_, mu1, logvar1 = model(data_scaler)
latent_space = mu1.detach().cpu().numpy()

model.eval()
with torch.no_grad():
   # torch.manual_seed(30)
   z = torch.randn(8000, latent_dim)
   generated = model.decode(z)
   inverse_data = scaler.inverse_transform(generated)
   df = pd.DataFrame(inverse_data)
   df.columns = pd.DataFrame(data_train).columns
   # df.to_excel("./generated_OOD_data/MOFs screening/generated_data_beta1.xlsx", index=False)
