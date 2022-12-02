# [Re] Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks

- Rescience C paper on [overleaf](https://www.overleaf.com/read/yzdtjgjppgkg)

## Install the python environment 

### With conda env

create the conda environment for jodie: 
```bash
$ conda create --name jodie python=3.10
$ conda activate jodie
$ python -m pip install -r requirements.txt  
```

### With virtual env
```bash
$ python -m venv env
$ source env/bin/activate
$ python -m pip install -r requirements.txt  
```

### Run the model and make predictions

```bash
$ python train_evaluate_wikipedia.py
```

## Code

The folder jodie gather the main functions.

## Results

To replicate results from table 2, lauch the following script with the hyperparameters setup (see the inside each script for some template):
```bash
$ python train_evaluate_reddit_state.py
$ python train_evaluate_wikipedia_state.py
$ python train_evaluate_mooc_state.py
```

To replicate results from table 3:
- train_evaluate_reddit.py
- train_evaluate_wikipedia.py
- train_evaluate_lastfm.py
