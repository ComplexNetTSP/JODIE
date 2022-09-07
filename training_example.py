from ray import tune
from jodie.preprocessing import *
from jodie.model import *
from jodie.train import *
from ray.tune.schedulers import ASHAScheduler
import os
import logging

logging.disable(logging.CRITICAL)
ray.init(logging_level=logging.FATAL)
logging.basicConfig(level=logging.CRITICAL)

for logger_name in logging.root.manager.loggerDict:
    logger = logging.getLogger(logger_name)
    logger.propagate = False
    logger.disabled = True
"""
config_format = {
    "embedding_dim" : 8, 16, 32, 64, 128 or tune.grid_search([8, 16, 32, 64, 128]),
    "learning_rate" : float (i.e.: 1e-5)
    "split" : integer between 1 and dataset number of observation,
    "lambda_u" : integer between 0 and infinity,
    "lambda_i" : integer between 0 and infinity,
    "dataset" : "mooc", "wikipedia", "lastfm" or "reddit",
    "n_epoch" : integer 50,
    "prop_train" : between 0 and 1,
    "state" : True or False,
    "device" : "cpu" or "gpu",
    "directory" : "/path/reporitory"
}
"""
# Simple config
config_fast_mooc = {
    "embedding_dim": tune.grid_search([8, 9]),
    "learning_rate": 1e-3,
    "split": 500,
    "lambda_u": 1,
    "lambda_i": 1,
    "dataset": "mooc",
    "n_epoch": 1,
    "prop_train": 0.6,
    "state" : True,
    "device": "cpu",
    "directory" : "/home/gauthierv/jodie/test"
}

config_long = {
    "embedding_dim": tune.grid_search([8, 16, 32, 64, 128]),
    "learning_rate": 1e-3,
    "split": tune.grid_search([5, 500, 50000]),
    "lambda_u": tune.grid_search([0.1, 1, 10]),
    "lambda_i": tune.grid_search([0.1, 1, 10]),
    "dataset": "mooc",
    "n_epoch": 50,
    "prop_train": 0.6,
    "state" : True,
    "device": "cpu",
    "directory" : "/home/gauthierv/jodie"
}


if __name__ == '__main__':
    print("Start the training")
    analysis = tune.run(train_ray,
                        num_samples=1,
                        config=config_fast_mooc,
                        resources_per_trial={"cpu": 4},
                        local_dir="./result",
                        verbose=0)
