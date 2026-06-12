import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data_train = pd.read_excel('train.xlsx')
data_train_X = data_train.iloc[:, 1:]

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# extract covariance matrix from training process
cov_matrix = np.array([[ 3.0359e-03,  4.6655e-03,  4.7213e-03,  2.5626e-03,  2.3842e-03,
         -3.6887e-04,  2.0363e-03,  2.6906e-04,  9.7477e-04],
        [ 4.6655e-03,  1.0169e-02,  8.0681e-03,  6.3572e-03,  2.6816e-03,
         -2.3432e-03,  1.7431e-03,  2.3365e-03,  3.0596e-03],
        [ 4.7213e-03,  8.0681e-03,  8.1316e-03,  3.5999e-03,  3.3816e-03,
         -3.0991e-04,  3.3009e-03,  7.6294e-05,  1.3167e-03],
        [ 2.5626e-03,  6.3572e-03,  3.5999e-03,  1.3736e-02,  1.5299e-03,
         -5.7424e-03, -1.5734e-03,  1.0388e-02,  7.6894e-03],
        [ 2.3842e-03,  2.6816e-03,  3.3816e-03,  1.5301e-03,  3.5591e-03,
         -1.0238e-04,  3.6514e-03,  7.4458e-04,  1.1083e-03],
        [-3.6887e-04, -2.3432e-03, -3.0991e-04, -5.7424e-03, -1.0238e-04,
          3.6854e-03,  1.4069e-03, -5.1671e-03, -3.9847e-03],
        [ 2.0363e-03,  1.7431e-03,  3.3008e-03, -1.5734e-03,  3.6514e-03,
          1.4069e-03,  6.1315e-03, -1.0785e-03, -7.3743e-04],
        [ 2.6906e-04,  2.3365e-03,  7.5698e-05,  1.0388e-02,  7.4458e-04,
         -5.1671e-03, -1.0785e-03,  1.2244e-02,  7.8119e-03],
        [ 9.6714e-04,  3.0596e-03,  1.3167e-03,  7.6894e-03,  1.1083e-03,
         -3.9847e-03, -7.3743e-04,  7.8119e-03,  6.4998e-03]])

# cov_matrix = cov * cov
df = pd.DataFrame(cov_matrix)
df.columns = data_train_X.columns
print(df)

# calculate Pearson correlation coefficient matrix from covariance matrix
std_devs = np.sqrt(np.diag(cov_matrix))
print('standard deviation matrix:', pd.DataFrame(std_devs))

correlation_matrix = cov_matrix / np.outer(std_devs, std_devs)
correlation_matrix[np.isnan(correlation_matrix)] = 0

print("outer():", pd.DataFrame(np.outer(std_devs, std_devs)))
print('Person_matrix:', pd.DataFrame(correlation_matrix))

Person_matrix = pd.DataFrame(correlation_matrix)
# Person_matrix.columns = data_train_X.columns
COLUMNS = []
for i in data_train_X.columns:
    x = '(' + i + ')'
    COLUMNS.append(x)
Person_matrix.columns = COLUMNS
Person_matrix.index = COLUMNS
# Person_matrix.to_excel('./Person matrix.xlsx', index=False, header=False)
print('sum:', np.sum(np.abs(Person_matrix.values)))

plt.figure(figsize=(8, 6))
ax = sns.heatmap(Person_matrix, annot=False, fmt='.2f',
                 annot_kws={"size":12},cmap='seismic', center=0, vmin=-1, vmax=1)
cbar = ax.collections[0].colorbar

plt.tight_layout()
plt.show()

