# JODIE

- Rescience C paper on [overleaf](https://www.overleaf.com/read/yzdtjgjppgkg)

## Install the python environement 

### With conda 

create the conda environement for jodie: 
```bash
$ conda create --name jodie python=3.10
$ conda activate jodie
$ python -m pip install -r requirements.txt  
```

### With virtual env
```bash
$ python -m venv jodie-env
$ source jodie-env/bin/activate
$ python -m pip install -r requirements.txt  
```
### Run the model 

```bash
$ python training_example.py
```

### Make predictions 

```bash
$ python evaluate_example.py
```
### Run the model and make predictions
```bash
$ python train_evaluate.py
```
