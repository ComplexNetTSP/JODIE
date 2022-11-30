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
$ sbatch slurm_wikipedia.sh
```

## Code

The folder jodie gather the main functions.

## Result

To replicate results from table 2:
- train_evaluate_reddit_state.py
- train_evaluate_wikipedia_state.py
- train_evaluate_mooc.py

To replicate results from table 3:
- train_evaluate_reddit.py
- train_evaluate_wikipedia.py
- train_evaluate_lastfm.py
