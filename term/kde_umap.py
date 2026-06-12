import numpy as np
import pandas as pd
import umap
import random
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

np.random.seed(42)
random.seed(42)

gen_data_paths = {
    "β = 1": "./generated_OOD_data/MOFs screening/generated_data_beta1.xlsx",
    "β = 1.5": "./generated_OOD_data/MOFs screening/generated_data_beta1-5.xlsx",
    "β = 2": "./generated_OOD_data/MOFs screening/generated_data_beta2.xlsx",
    "β = 2.5": "./generated_OOD_data/MOFs screening/generated_data_beta2-5.xlsx",
    "β = 3": "./generated_OOD_data/MOFs screening/generated_data_beta3.xlsx",
    "β = 3.5": "./generated_OOD_data/MOFs screening/generated_data_beta3-5.xlsx",
    "β = 4": "./generated_OOD_data/MOFs screening/generated_data_beta4.xlsx",
    "β = 4.5": "./generated_OOD_data/MOFs screening/generated_data_beta4-5.xlsx",
    "β = 5": "./generated_OOD_data/MOFs screening/generated_data_beta5.xlsx",
}

raw_df = pd.read_excel("dataset.xlsx")   # raw data
raw_df["source"] = "raw data"

gen_dfs = []
for label, path in gen_data_paths.items():
    df = pd.read_excel(path)
    df["source"] = label
    gen_dfs.append(df)

all_df = pd.concat([raw_df] + gen_dfs, ignore_index=True)

X = all_df.iloc[:,:-1].values
labels = all_df["source"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42,
    init="spectral",
    metric='euclidean',
    angular_rp_forest=False,
    transform_seed=42,
    force_approximation_algorithm=False
)

X_umap = reducer.fit_transform(X_scaled)

fig, ax = plt.subplots()
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

# raw data density
"""
idx_raw = labels == "raw data"
sns.kdeplot(
    x=X_umap[idx_raw, 0],
    y=X_umap[idx_raw, 1],
    levels=10,
    fill=True,
    alpha=1,
    cmap="Oranges",
    linewidths=1.5,
)
"""
# Generated data density
# """
idx_gen = labels == "β = 4.5"
sns.kdeplot(
    x=X_umap[idx_gen, 0],
    y=X_umap[idx_gen, 1],
    levels=10,
    fill=True,
    alpha=0.8,
    cmap="Oranges",
)

raw_patch = mpatches.Rectangle((0, 0), 1, 1, facecolor='#ff7f0e', alpha=0.7, edgecolor='none')
beta_patch = mpatches.Rectangle((0, 0), 1, 1, facecolor='#ff7f0e', alpha=0.8, edgecolor='none')

legend_elements = [
    # (raw_patch, "raw data-MOFs"),
    (beta_patch, "β = 4.5")
]

ax.legend(
    handles=[elem[0] for elem in legend_elements],
    labels=[elem[1] for elem in legend_elements],
    prop={'size':28,'family':'Arial'},
    loc='upper center',
    bbox_to_anchor=(0.5, 1.15),
    frameon=False,
    ncol=2,
    handlelength=1.5,
    handletextpad=0.5,
    columnspacing=2
)

plt.xlabel("UMAP-1", fontsize=28, fontname='Arial')
plt.ylabel("UMAP-2", fontsize=28, fontname='Arial')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
