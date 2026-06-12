import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler

raw_data = pd.read_excel('dataset.xlsx')
data_candidate = pd.read_excel("predict.xlsx")
data_generated = pd.read_excel('./generated_OOD_data/MOFs screening/generated_data_beta1.xlsx')


# ===============================
scaler = StandardScaler()
scaler.fit(raw_data.values)

raw_data = scaler.transform(raw_data.values)
candidate_data = scaler.transform(data_candidate.values)
gen_data = scaler.transform(data_generated.values)


def compute_mmd(x: np.ndarray, y: np.ndarray, gamma: float = None) -> float:

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    if gamma is None:
        np.random.seed(42)
        sample_size = min(1000, x.shape[0], y.shape[0])
        idx_x = np.random.choice(x.shape[0], sample_size, replace=False)
        idx_y = np.random.choice(y.shape[0], sample_size, replace=False)
        pairwise_dists = np.linalg.norm(
            x[idx_x, None, :] - y[None, idx_y, :], axis=-1
        )
        median_dist = np.median(pairwise_dists)
        gamma = 1.0 / (2 * (median_dist ** 2 + 1e-8))

    Kxx = rbf_kernel(x, x, gamma=gamma)
    Kyy = rbf_kernel(y, y, gamma=gamma)
    Kxy = rbf_kernel(x, y, gamma=gamma)

    # MMD = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
    mmd = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
    return float(mmd)


mmd_value_candidate = compute_mmd(raw_data, candidate_data)
mmd_value_gen = compute_mmd(raw_data, gen_data)

print(f"MMD between raw and candidate data: {mmd_value_candidate:.6f}")
print(f"MMD between raw and generated data: {mmd_value_gen:.6f}")

