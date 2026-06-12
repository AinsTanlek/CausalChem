import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel('./generated_OOD_data/MOFs screening/generated_data_beta5.xlsx')
beta = 5.0

colors = [
    'skyblue', 'lightgreen', 'lightcoral', 'gold', 'lightpink',
    'lightblue', 'lavender', 'cyan', 'peachpuff', 'blueviolet'
]

fig, axes = plt.subplots(2, 5, figsize=(10, 3))
data.hist(bins=15, ax=axes)

for i, ax in enumerate(axes.flatten()):
    c = colors[i % len(colors)]
    for patch in ax.patches:
        patch.set_facecolor(c)
    ax.grid(False)

fig.text(0.02, 0.5, f"β = {beta}", va="center", ha="center",
         rotation="vertical", fontsize=16)
plt.tight_layout(rect=[0.04, 0, 1, 1])
plt.show()
