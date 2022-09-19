from ray import tune
import ray
from jodie.preprocessing import *
from jodie.model import *
from jodie.train import *
from jodie.evaluate import *
from ray.tune.schedulers import ASHAScheduler
import os
import csv
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
    "embedding_dim": 8,#tune.grid_search([8,16,32]),
    "learning_rate": 1e-3,
    "split": 500,#tune.grid_search([5,500,50000]),
    "lambda_u": 1,#tune.grid_search([0.1,1,10]),
    "lambda_i": 1,#tune.grid_search([0.1,1,10]),
    "dataset": "mooc",
    "n_epoch": 1,
    "prop_train": 0.6,
    "state" : True,
    "device": "cpu",
    "directory" : "/home/gauthierv/jodie"
}

if __name__ == '__main__':
    print("*************************** Start the training for ",end='')
    print("state change prediction" if config_fast_mooc["state"] else "future interaction prediction ",end='')
    print("***************************")
    analysis = tune.run(train_ray,
                        num_samples=1,
                        config=config_fast_mooc,
                        resources_per_trial={"cpu": 10},
                        local_dir="./result",
                        verbose=0)
    
    print("*************************** Start the evaluation process ***************************")
    filename = config_fast_mooc["directory"]+"/"+ config_fast_mooc["dataset"]+"_hyper-parameter.txt"
    with open(filename, 'r') as hyperparameters_file:
        reader = csv.reader(hyperparameters_file, delimiter=',')
        for hyperparameters in reader:
            print("embedding_dim:", hyperparameters[0], 
                  ", learning_rate:", hyperparameters[1],
                  ", split:", hyperparameters[2],
                  ", lambda_u:",hyperparameters[3],
                  ", lambda_i:",hyperparameters[4],
                  )
            perf_val, perf_test = evaluate(','.join(hyperparameters), 
                                           config_fast_mooc["dataset"], 
                                           config_fast_mooc["n_epoch"], 
                                           config_fast_mooc["device"], 
                                           config_fast_mooc["prop_train"], 
                                           config_fast_mooc["state"],
                                           config_fast_mooc["directory"])
            print("validation:", perf_val["val"], ", test:", perf_test["test"])
