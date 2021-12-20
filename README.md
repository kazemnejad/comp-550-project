# Scaling Neural Text Classifiers in Low-Parameter Regime

Instructions to replicate
## Install dependencies
```bash
pip install -r requirements.txt
```
## Download and prepare the data
Follow the instruction in `prepare_data.ipynb`
## Run the experiments
1. CNN
```
python grid_search.py cnn agnews
```
2. LSTM
```
python grid_search.py lstm agnews
```
3. Transformer
```
python grid_search.py transformer agnews
```
You can run another instance of each on other nodes if you have more GPUs available.
## Generate the plots
Follow the instructions in `analyze.ipynb`

## Validation result on different various hyperparameters
1. Transformer

|    |    32 |   160 |   288 |   416 |   544 |   672 |   800 |   928 |
|---:|------:|------:|------:|------:|------:|------:|------:|------:|
|  1 | 90.76 | 90.82 | 91    | 90.67 | 91.12 | 90.92 | 90.67 | 90.83 |
|  2 | 90.57 | 91.09 | 91.13 | 90.25 | 89.7  | 89.4  | 90.82 | 90.19 |
|  3 | 91.04 | 90.23 | 91.01 | 90.92 | 89.8  | 90.1  | 90.08 | 90.5  |
|  4 | 91.11 | 91.05 | 89.93 | 90.26 | 89.38 | 90.22 | 89.35 | 89.7  |
|  5 | 90.78 | 90.63 | 89.72 | 89.99 | 88.86 | 89.67 | 89.58 | 88.56 |
|  6 | 90.39 | 89.8  | 90.58 | 87.88 | 89.26 | 90    | 89.71 | 86.62 |

2. LSTM
|    |    32 |   160 |   288 |   416 |   544 |   672 |   800 |   928 |
|---:|------:|------:|------:|------:|------:|------:|------:|------:|
|  1 | 90.35 | 90.22 | 90.61 | 90.28 | 90.56 | 90.2  | 90.22 | 89.83 |
|  2 | 90.5  | 90.3  | 90.41 | 90.28 | 90.04 | 89.92 | 89.9  | 89.78 |
|  3 | 89.99 | 90.4  | 90.01 | 90.17 | 90.08 | 89.95 | 89.66 | 90.24 |
|  4 | 88.93 | 90.04 | 89.79 | 89.41 | 89.98 | 89.71 | 89.45 | 89.99 |
|  5 | 89.89 | 90.43 | 89.61 | 89.57 | 89.68 | 89.48 | 89.56 | 89.59 |
|  6 | 89.32 | 90.77 | 89.61 | 90.09 | 90.09 | 89.95 | 89.6  | 89.05 |

3. CNN
|    |    32 |   160 |   288 |   416 |   544 |   672 |   800 |   928 |
|---:|------:|------:|------:|------:|------:|------:|------:|------:|
|  1 | 55.18 | 77.19 | 77.83 | 83.05 | 83.17 | 83.7  | 77.77 | 83.09 |
|  2 | 80.26 | 88.26 | 89.2  | 89.03 | 88.98 | 89.49 | 89.02 | 89.59 |
|  3 | 83    | 89.31 | 86.95 | 87.27 | 86.51 | 87.73 | 87.61 | 88.08 |
|  4 | 84.34 | 86.07 | 87.58 | 88.12 | 87.97 | 87.91 | 88.36 | 88.49 |
|  5 | 85.19 | 87.76 | 88.09 | 88.1  | 88.58 | 88.5  | 88.41 | 88.66 |
|  6 | 88.68 | 87.43 | 88.17 | 88.42 | 88.74 | 88.62 | 88.89 | 89.27 |

### Acknowledgement
The template for this repository is taken from AllenNLP.
