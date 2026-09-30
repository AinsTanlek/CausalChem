# CausalChem
This is the code for the paper 'CausalChem: Causally stable chemical modeling for materials prediction'. Machine learning has drastically shortened the cycle of materials screening and discovery and cut the overall costs incurred throughout the materials development pipeline. Nevertheless, prevailing chemical modeling frameworks to date remain correlation-driven; such paradigms typically suffer from poor interpretability and limited generalization capability under unseen chemical environments.
Herein, we propose a causally stable chemical modeling framework called CausalChem. We benchmark the OOD predictive performance of CausalChem against existing chemical modeling algorithms across three widely investigated chemical tasks: metal-organic frameworks (MOFs) screening for light hydrocarbons separation (MOFs screening), perovskite oxide electrodes discovery for oxygen reduction reaction activity (POEs discovery), and chiral N,O-bidentate salicyloxazoline-type ligands optimization in nickel electrocatalysis for enhanced enantioselectivity (Ligands optimization). By eliminating spurious correlations among chemical descriptors, CausalChem identifies intrinsically stable causal relationships between approximately independent descriptors and target material properties. This approach yields improved predictive accuracy under OOD scenarios while substantially enhancing the intrinsic interpretability of predictive models. 

## Installation
### Requirements
- Python = 3.11
- PyTorch = 2.1.2
- numpy = 1.26.0
- shap = 0.49.1
- umap-learn = 0.5.9
- pandas = 2.1.1

## Quick Start
### Train CausalChem
```bash
python main.py --gpu 0
```
You can also run on the CPU, but it will be a bit slow.
```bash
python main.py --gpu -1
```

## Parameters
CausalChem is tailored for tabular chemical data, the most prevalent format for chemical datasets that conforms to standard chemical data recording conventions. All datasets collected from comparative literatures are stored under the `dataset` folder. To switch between different chemical tasks, only the input architecture of the network needs fine-tuning for the target task. Hyperparameters(such as hidden layers, batch size and learning rate) should be tuned according to the task complexity and sample size of each dataset.

### MOFs screening task
|Component|	Setting/Value	|Notes|
|---|---|---|
|Input dimension|	9	|Number of input features|
|Hidden layers|	2×17 units|	Fully connected layers with ReLU activation|
|Output layer|	1 unit (linear)|	ReLU activation|	
|Optimizer|	SGD	|Momentum = 0.9; L2 weight decay = 1×10⁻⁴
|Initial learning rate|	0.09 |	
|Learning rate schedule|	Cosine decay, or step decay| |	
|Batch size| 64 | |	
|Loss function|	Mean squared error	|reduction='none'|
|random_seed|	3 |	
|Weight initialization|	Kaiming normal (He initialization)|	nn.init.kaiming_normal_, applied to all layers|
|Bias initialization|	Zeros	|Default setting|

### POEs discovery task
|Component|	Setting/Value	|Notes|
|---|---|---|
|Input dimension|	9	|Number of input features|
|Hidden layers|	2×17 units|	Fully connected layers with ReLU activation|
|Output layer|	1 unit (linear)|	No activation function applied|
|Optimizer|	SGD	|Momentum = 0.9; L2 weight decay = 1×10⁻⁴
|Initial learning rate|	0.08 |	
|Learning rate schedule|	Cosine decay, or step decay| |	
|Batch size| 8 | |	
|Loss function|	Mean squared error	|reduction='none'|
|random_seed|	3 |	
|Weight initialization|	Default PyTorch (Kaiming uniform)|	Applied to all layers|
|Bias initialization|	Zeros	|Default setting|

### Ligands optimization task
|Component|	Setting/Value	|Notes|
|---|---|---|
|Input dimension|	7	|Number of input features|
|Hidden layers|	2×9 units|	Fully connected layers with ReLU activation|
|Output layer|	1 unit (linear)|	No activation function applied|	
|Optimizer|	SGD	|Momentum = 0.9; L2 weight decay = 1×10⁻⁴
|Initial learning rate|	0.009 |	
|Learning rate schedule|	Cosine decay, or step decay| |	
|Batch size| 4 | |	
|Loss function|	Mean squared error	|reduction='none'|
|random_seed|	3 |	
|Weight initialization|	Kaiming normal (He initialization)|	nn.init.kaiming_normal_, applied to all layers|
|Bias initialization|	Zeros	|Default setting|

## Generate samples
The file of `beta-VAE.py` is used to generate chemical OOD data, which converts chemical samples into latent representations `z` and outputs similar structures through a decoder.

```bash
z = torch.randn(number_new_samples, latent_dim)
output = model.decode(z)
generated_data = scaler.inverse_transform(output)
```
The generated data is stored in folder `generated_OOD_data`, for example:

- `./MOFs screening/generated_data_beta1.xlsx` — the generated OOD data of β = 1.
- `./MOFs screening/generated_data_beta1-5.xlsx` — the generated OOD data of β = 1.5.

A two-dimensional kernel density estimation (KDE) approach was employed to qualitatively examine the extent to which the generated samples deviated from the original data in chemical space. In addition, the Maximum Mean Discrepancy (MMD) metric was used to quantitatively evaluate the distributional shift of the generated data. These analyses help ensure the reliability of the generated samples for OOD generalization assessment.

```bash
from mmd_distance import compute_mmd
mmd_value = compute_mmd(raw_data, generated_data)
```

## Interpretability
SHAP analysis was conducted to demonstrate the enhanced intrinsic interpretability provided by CausalChem in chemical systems.

```bash
background_data = x_train[np.random.choice(x_train.shape[0], num_train_samples, replace=False)]

explainer = shap.KernelExplainer(
    model=model_predict,
    data=background_data,
    link="identity"
)
```

## Reform
### Chemical adaptation
The examples presented in this study mainly involve learning the regression mapping between `material compositions` and `material properties` using CausalChem. For `classification` tasks in other chemical domains, this framework can be readily adapted by modifying the loss function and output layer. In addition, for `non-tabular` chemical data(such as `spectroscopic data`, `electrical current signals`, and `electron microscopy images`), the backbone network can be customized to perform initial feature extraction and representation learning.

### Cross-Domain Applications
As discussed in our manuscript, the CausalChem framework is built upon the `invariance` of underlying chemical principles. In theory, this framework can be extended to other non-chemical tabular prediction tasks governed by invariant inherent rules. Limited by insufficient domain expertise and available datasets from alternative research fields, we have not conducted extensive explorations.
