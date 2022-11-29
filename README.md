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

# Code

The folder jodie gather the main functions.
