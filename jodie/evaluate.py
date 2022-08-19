# packages
import argparse
from preprocessing import *
from model import *
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm

import train_ray as t

parser = argparse.ArgumentParser()
parser.add_argument('--split_select')
args = parser.parse_args()

data = pd.read_csv("/home/gauthierv/jodie/ray_tune2/mooc.csv")

# small dataset for testing
#data = data[:len(data)*2//5]

df = data.to_numpy()
proportion_train = 0.6
device = "cpu"
epoch = 50

# preprocessing
user_id, id_user_sequence, delta_u, previous_item_sequence, item_id, id_item_sequence, delta_i, timestamp_sequence, feature_sequence, true_labels = preprocess(df)
num_interaction = len(id_user_sequence)
num_user = len(user_id)
num_item = len(item_id) + 1
num_feature = len(feature_sequence[0])
ratio_label = len(true_labels) / (1 + sum(true_labels))
print(len(true_labels))

embedding_dim, learning_rate, split, lambda_u, lambda_i = load_param(args.split_select)
print(args.split_select)
print(embedding_dim, learning_rate, split, lambda_u, lambda_i)

idx_train = int(num_interaction * proportion_train)
idx_val = int(num_interaction * (proportion_train + ((1 - proportion_train) / 2)))
idx_test = int(num_interaction)

# clue for testing
#idx_val = int(num_interaction * (proportion_train + 0.01))
#idx_test = int(num_interaction * (proportion_train + 0.02))

tbatch_timespan = (timestamp_sequence[-1] - timestamp_sequence[0]) / split
id_time_sequence = []

# initialize model and parameters
model = RODIE(embedding_dim, num_user, num_item, num_feature, activation_rnn = "tanh", MLP_hidden_layer_dim = 50).to(device)
weight = torch.Tensor([1, ratio_label]).to(device)
CE_loss = nn.CrossEntropyLoss(weight = weight)
MSE = nn.MSELoss()

# initialize model
lr = learning_rate
optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 1e-5)

# load model and parameters
model, optimizer, embedding_dynamic_static_user, embedding_dynamic_static_item, embedding_user_timeserie, embedding_item_timeserie, idx_training = load_model(model, optimizer, epoch, device, embedding_dim, learning_rate, split, lambda_u, lambda_i)

if idx_train != idx_training:
  sys.exit("Training proportion different")

set_embedding(id_user_sequence, id_item_sequence, proportion_train, embedding_user_timeserie, embedding_item_timeserie, embedding_dynamic_static_user, embedding_dynamic_static_item)

embedding_item = embedding_dynamic_static_item[:, : embedding_dim]
embedding_item = embedding_item.clone()
embedding_item_static = embedding_dynamic_static_item[:, embedding_dim :]
embedding_item_static = embedding_item_static.clone()

embedding_user = embedding_dynamic_static_user[:, : embedding_dim]
embedding_user = embedding_user.clone()
embedding_user_static = embedding_dynamic_static_user[:, embedding_dim :]
embedding_user_static = embedding_user_static.clone()

val_pred = []
val_true = []

test_pred = []
test_true = []

tbatch_time = None
loss = 0

# no t-batch, treat each interaction
for i in tqdm(range(idx_train, idx_test), desc = 'Progress bar'):
  id_user = id_user_sequence[i]
  id_item = id_item_sequence[i]
  id_time = timestamp_sequence[i]
  id_feature = feature_sequence[i]
  id_delta_u = delta_u[i]
  id_delta_i = delta_i[i]
  id_previous = previous_item_sequence[i]

  if tbatch_time is None:
    tbatch_time = id_time

  embedding_user_input = embedding_user[torch.LongTensor([id_user])]
  embedding_user_static_input = embedding_user_static[torch.LongTensor([id_user])]
  embedding_item_input = embedding_item[torch.LongTensor([id_item])]
  embedding_item_static_input = embedding_item_static[torch.LongTensor([id_item])]

  feature_tensor = Variable(torch.Tensor(id_feature).to(device)).unsqueeze(0)
  delta_u_tensor = Variable(torch.Tensor([id_delta_u]).to(device)).unsqueeze(0)
  delta_i_tensor = Variable(torch.Tensor([id_delta_i]).to(device)).unsqueeze(0)

  item_embedding_previous = embedding_item[torch.LongTensor([id_previous])]

  projected_embedding_user = model.projection(embedding_user_input, delta_u_tensor)
  embedding_user_item = torch.cat([projected_embedding_user, item_embedding_previous, embedding_item_static[torch.LongTensor([id_previous])], embedding_user_static_input], dim = 1)

  predict_embedding_item = model.predict_embedding_item(embedding_user_item)

  loss += MSE(predict_embedding_item, torch.cat([embedding_item_input, embedding_item_static_input], dim = 1).detach())

  update_embedding_user = model.update_rnn_user(embedding_user_input, embedding_item_input, feature_tensor, delta_u_tensor)
  update_embedding_item = model.update_rnn_item(embedding_item_input, embedding_user_input, feature_tensor, delta_i_tensor)

  embedding_item[id_item, :] = update_embedding_item.squeeze(0)
  embedding_user[id_user, :] = update_embedding_user.squeeze(0)
  embedding_user_timeserie[i, :] = update_embedding_user.squeeze(0)
  embedding_item_timeserie[i, :] = update_embedding_item.squeeze(0)

  loss += regularizer(update_embedding_user, embedding_user_input.detach(), lambda_u)
  loss += regularizer(update_embedding_item, embedding_item_input.detach(), lambda_i)
  loss += model.loss_predict_state(model, device, [i], embedding_user_timeserie, true_labels, CE_loss)

  if id_time - tbatch_time > tbatch_timespan:
    tbatch_time = id_time
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    loss = 0
    embedding_item.detach_()
    embedding_user.detach_()
    embedding_user_timeserie.detach_()
    embedding_item_timeserie.detach_()

  proba = model.predict_state(update_embedding_user)

  if i < idx_val:
    val_pred.extend(proba.data.cpu().numpy())
    val_true.extend([true_labels[i]])
  else:
    test_pred.extend(proba.data.cpu().numpy())
    test_true.extend([true_labels[i]])

val_pred = np.array(val_pred)
test_pred = np.array(test_pred)

perf_val = {}
perf_test = {}

auc_val = roc_auc_score(val_true, val_pred[:, 1])
perf_val["val"] = [auc_val]
auc_test = roc_auc_score(test_true, test_pred[:, 1])
perf_test["test"] = [auc_test]
print(perf_val, perf_test)

file = open("/home/gauthierv/jodie/resultats/{}_{}_{}_{}_{}.txt".format(embedding_dim, learning_rate, split, lambda_u, lambda_i), "a")
metrics = ["AUC"]
for i in range(len(metrics)):
  file.write("Validation : " + metrics[i] + " : " + str(perf_val["val"][i]) + "\n")
  file.write("Test : " + metrics[i] + " : " + str(perf_test["test"][i]) + "\n")

file.flush()
file.close()