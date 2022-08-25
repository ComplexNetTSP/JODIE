from jodie.evaluate import *

# hyperparameter format : embedding_size, learning_rate, split, lambda_u and lambda_i
hyperparameter = [8, 1e-3, 500, 1, 1]
data = "mooc"
epoch = 50
device = "cpu"
proportion_train = 0.6
state = True

perf_val, perf_test = evaluate(hyperparameter, data, epoch, device, proportion_train, state)
print(perf_val, perf_test)