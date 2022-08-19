from ray import tune
from jodie.preprocessing import *
from jodie.model import *
from jodie.train import *
from ray.tune.schedulers import ASHAScheduler
import os

# Romain
config = {
    "embedding_dim": tune.grid_search([8, 16, 32, 64, 128]),
    "learning_rate": 1e-3,
    "split": tune.grid_search([5, 500, 50000]),
    "lambda_u": tune.grid_search([0.1, 1, 10]),
    "lambda_i": tune.grid_search([0.1, 1, 10]),
    "dataset": "mooc"
}

if __name__ == '__main__':
    analysis = tune.run(train_ray,
                        num_samples=1,
                        config=config,
                        resources_per_trial={"cpu": 2},
                        local_dir="./jodie/result",
                        verbose=0)
