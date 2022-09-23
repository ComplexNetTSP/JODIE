from jodie.evaluate import *

data = "reddit"   # 4 options : "mooc", "wikipedia", "lastfm" and "reddit"
evaluate_epoch = 50  # depend on training epochs
device = "cuda"  # 2 options : "cpu" and "gpu"
proportion_train = 0.8  # between 0 and 1
state = False    # 2 options : True or False
directory = "/mnt/beegfs/home/gauthier/JODIE/"    # format must be : /home/name/reporitory

fichier = open(directory+"/"+data+"_hyper-parameter.txt", "r")
lines = fichier.readlines()

for line in lines:
    perf_val, perf_test = evaluate(line.strip("\n"), data, evaluate_epoch, device, proportion_train, state, directory)
    print(perf_val, perf_test)

fichier.close()