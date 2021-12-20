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

### Acknowledgement
The template for this repository is taken from AllenNLP.
