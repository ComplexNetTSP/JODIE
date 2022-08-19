from ray import tune
from jodie.preprocessing import *
from jodie.model import *
from jodie.train import *
from ray.tune.schedulers import ASHAScheduler
import os

# Simple config
config_fast = {
    "embedding_dim": 8,
    "learning_rate": 1e-3,
    "split": 500,
    "lambda_u": 1,
    "lambda_i": 1,
    "dataset": "mooc",
    "n_epoch": 5,
    "prop_train": 0.06,
    "state" : True
}

config_long = {
    "embedding_dim": tune.grid_search([8, 16, 32, 64, 128]),
    "learning_rate": 1e-3,
    "split": tune.grid_search([5, 500, 50000]),
    "lambda_u": tune.grid_search([0.1, 1, 10]),
    "lambda_i": tune.grid_search([0.1, 1, 10]),
    "dataset": "mooc",
    "n_epoch": 5,
    "prop_train": 0.6,
    "state" : True
}


if __name__ == '__main__':
    print("Start the trainning")
    analysis = tune.run(train_ray,
                        num_samples=1,
                        config=config_fast,
                        resources_per_trial={"cpu": 20},
                        local_dir="./result",
                        verbose=0)
