### Code repository of the paper entitled "[[Re] Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks](https://github.com/ComplexNetTSP/JODIE-RESCIENCE/blob/master/article.pdf)"

#### [R.Haton, R. Ait Ali Yahia, V. Gauthier and A. Bouzeghoub, "[Re] Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks", ReScience C 2022](https://github.com/ComplexNetTSP/JODIE-RESCIENCE/blob/master/article.pdf)

#### This Repository containing the code and article for the replication of the article 'A circuit model of auditory cortex.' PLoS Computational Biology, 16(7), e1008016 by Youngmin Park and Maria N. Geffen.

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

The folder **jodie** gathers the main functions, namely the JODIE model in model.py, the preprocess of data in preprocessing.py, the training loop in train.py and the file evaluate.py to evaluate the model.

## Results

To replicate results from table 2, lauch the following script with the hyperparameters setup (see each script for some template):
```bash
$ python train_evaluate_reddit_state.py
$ python train_evaluate_wikipedia_state.py
$ python train_evaluate_mooc_state.py
```

To replicate results from table 3, lauch the following script wit the hyperparameters setup (see each script for some template):
```bash
$ python train_evaluate_reddit.py
$ python train_evaluate_wikipedia.py
$ python train_evaluate_lastfm.py
```
