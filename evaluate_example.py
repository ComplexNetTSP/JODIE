from jodie.evaluate import *

data = "mooc"   # 4 options : "mooc", "wikipedia", "lastfm" and "reddit"
evaluate_epoch = 1  # depend on training epochs
device = "cpu"  # 2 options : "cpu" and "gpu"
proportion_train = 0.6  # between 0 and 1
state = True    # 2 options : True or False
directory = "/Users/vgauthier/Documents/TelecomSudParis/TravauxRecherche/Python/JODIE"    # format must be : /home/name/reporitory

fichier = open(directory+"/"+data+"_hyper-parameter.txt", "r")
lines = fichier.readlines()

for line in lines:
    perf_val, perf_test = evaluate(line.strip("\n"), data, evaluate_epoch, device, proportion_train, state, directory)
    print(perf_val, perf_test)

fichier.close()