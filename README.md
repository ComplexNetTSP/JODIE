# JODIE

- Rescience C paper on [overleaf](https://www.overleaf.com/read/yzdtjgjppgkg)

## Install the python environment 

## lab IA 

* [LabIA documentation](https://doc.lab-ia.fr/getting-started/)

### With conda 

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

```bash
python -m pip install --proxy=http://webproxy.lab-ia.fr:8080 -r requirements.txt
```

### Run the model and make predictions

```bash
$ sbatch slurm_wikipedia.sh
```

### Change hyperparameters
In file train_evaluate_wikipedia.py, change the configuration config_wiki
