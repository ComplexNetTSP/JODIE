from jodie.model import *
from jodie.preprocessing import *

# packages
import argparse

from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm
import jodie.train as t

def evaluate(hyperparameter, data, epoch=50, device="cpu", proportion_train=0.6, state=True, directory="./"):
    
    dataset = fetch_datasets(data)
    
    df = dataset.to_numpy()

    # preprocessing
    user_id, id_user_sequence, delta_u, previous_item_sequence, item_id, id_item_sequence, delta_i, timestamp_sequence, feature_sequence, true_labels = preprocess(df)
    num_interaction = len(id_user_sequence)
    num_user = len(user_id)
    num_item = len(item_id) + 1
    num_feature = len(feature_sequence[0])
    ratio_label = len(true_labels) / (1 + sum(true_labels))
    
    embedding_dim, learning_rate, split, lambda_u, lambda_i = hyperparameter.split(",")
    embedding_dim, learning_rate, split = int(embedding_dim), float(learning_rate), int(split)
    if lambda_u == '0.1':
        lambda_u = float(lambda_u)
    else:
        lambda_u = int(lambda_u)
    
    if lambda_i == '0.1':
        lambda_i = float(lambda_i)
    else:
        lambda_i = int(lambda_i)
        
    print("embedding_size : {}, learning_rate : {}, split : {}, lambda_u : {}, lambda_i : {}".format(embedding_dim, learning_rate, split, lambda_u, lambda_i))

    idx_train = int(num_interaction * proportion_train)
    idx_val = int(num_interaction * (proportion_train + ((1 - proportion_train) / 2)))
    idx_test = int(num_interaction)

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
    model, optimizer, embedding_dynamic_static_user, embedding_dynamic_static_item, embedding_user_timeserie, embedding_item_timeserie, idx_training = load_model(model, optimizer, epoch, device, embedding_dim, learning_rate, split, lambda_u, lambda_i, data, directory)

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

    tbatch_time = None
    loss = 0

    if state:

        val_pred = []
        val_true = []

        test_pred = []
        test_true = []

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

            if device == "cpu":
                embedding_user_input = embedding_user[torch.LongTensor([id_user])]
                embedding_user_static_input = embedding_user_static[torch.LongTensor([id_user])]
                embedding_item_input = embedding_item[torch.LongTensor([id_item])]
                embedding_item_static_input = embedding_item_static[torch.LongTensor([id_item])]
                item_embedding_previous = embedding_item[torch.LongTensor([id_previous])]

            if device == "gpu":
                embedding_user_input = embedding_user[torch.cuda.LongTensor([id_user])]
                embedding_user_static_input = embedding_user_static[torch.cuda.LongTensor([id_user])]
                embedding_item_input = embedding_item[torch.cuda.LongTensor([id_item])]
                embedding_item_static_input = embedding_item_static[torch.cuda.LongTensor([id_item])]
                item_embedding_previous = embedding_item[torch.cuda.LongTensor([id_previous])]

            feature_tensor = Variable(torch.Tensor(id_feature).to(device)).unsqueeze(0)
            delta_u_tensor = Variable(torch.Tensor([id_delta_u]).to(device)).unsqueeze(0)
            delta_i_tensor = Variable(torch.Tensor([id_delta_i]).to(device)).unsqueeze(0)

            projected_embedding_user = model.projection(embedding_user_input, delta_u_tensor)
            if device == "cpu":
                embedding_user_item = torch.cat([projected_embedding_user, item_embedding_previous, embedding_item_static[torch.LongTensor([id_previous])], embedding_user_static_input], dim = 1)

            if device == "gpu":
                embedding_user_item = torch.cat([projected_embedding_user, item_embedding_previous, embedding_item_static[torch.cuda.LongTensor([id_previous])], embedding_user_static_input], dim = 1)

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

        file = open(directory+"/resultats/"+data+"/resultats_"+data+"_{}_{}_{}_{}_{}.txt".format(embedding_dim, learning_rate, split, lambda_u, lambda_i), "a")
        metrics = ["AUC"]
        for i in range(len(metrics)):
            file.write("Validation : " + metrics[i] + " : " + str(perf_val["val"][i]) + "\n")
            file.write("Test : " + metrics[i] + " : " + str(perf_test["test"][i]) + "\n")

        file.flush()
        file.close()

        return perf_val, perf_test

    if not state:
        val_rank = []
        test_rank = []

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
            
            # load embedding user and item
            if device == "cpu":
                embedding_user_input = embedding_user[torch.LongTensor([id_user])]
                embedding_user_static_input = embedding_user_static[torch.LongTensor([id_user])]
                embedding_item_input = embedding_item[torch.LongTensor([id_item])]
                embedding_item_static_input = embedding_item_static[torch.LongTensor([id_item])]
                item_embedding_previous = embedding_item[torch.LongTensor([id_previous])]

            if device == "gpu":
                embedding_user_input = embedding_user[torch.cuda.LongTensor([id_user])]
                embedding_user_static_input = embedding_user_static[torch.cuda.LongTensor([id_user])]
                embedding_item_input = embedding_item[torch.LongTensor([id_item])]
                embedding_item_static_input = embedding_item_static[torch.LongTensor([id_item])]
                item_embedding_previous = embedding_item[torch.LongTensor([id_previous])]

            feature_tensor = Variable(torch.Tensor(id_feature).to(device)).unsqueeze(0)
            delta_u_tensor = Variable(torch.Tensor([id_delta_u]).to(device)).unsqueeze(0)
            delta_i_tensor = Variable(torch.Tensor([id_delta_i]).to(device)).unsqueeze(0)

            projected_embedding_user = model.projection(embedding_user_input, delta_u_tensor)
            
            if device == "cpu":
                embedding_user_item = torch.cat([projected_embedding_user, item_embedding_previous, embedding_item_static[torch.LongTensor([id_previous])], embedding_user_static_input], dim = 1)

            if device == "gpu":
                embedding_user_item = torch.cat([projected_embedding_user, item_embedding_previous, embedding_item_static[torch.cuda.LongTensor([id_previous])], embedding_user_static_input], dim = 1)

            predict_embedding_item = model.predict_embedding_item(embedding_user_item)

            loss += MSE(predict_embedding_item, torch.cat([embedding_item_input, embedding_item_static_input], dim = 1).detach())

            # calculate distance of predicted item embedding to all items embeddings
            euclidean_dist = nn.PairwiseDistance()(predict_embedding_item.repeat(num_item, 1), torch.cat([embedding_item, embedding_item_static], dim = 1)).squeeze(-1)

            # calculate rank of the true item among all items
            true_item_dist = euclidean_dist[id_item]
            euclidean_dist_smaller = (euclidean_dist < true_item_dist).data.cpu().numpy()
            true_item_rank = np.sum(euclidean_dist_smaller) + 1

            if i < idx_val:
                val_rank.append(true_item_rank)
            else:
                test_rank.append(true_item_rank)

            # update embedding_user and item
            update_embedding_user = model.update_rnn_user(embedding_user_input, embedding_item_input, feature_tensor, delta_u_tensor)
            update_embedding_item = model.update_rnn_user(embedding_item_input, embedding_user_input, feature_tensor, delta_i_tensor)

            embedding_item[id_item, :] = update_embedding_item.squeeze(0)
            embedding_user[id_user, :] = update_embedding_user.squeeze(0)
            embedding_user_timeserie[i, :] = update_embedding_user.squeeze(0)
            embedding_item_timeserie[i, :] = update_embedding_item.squeeze(0)

            # calculate loss regularization
            loss += regularizer(update_embedding_user, embedding_user_input.detach(), lambda_u)
            loss += regularizer(update_embedding_item, embedding_item_input.detach(), lambda_i)

            if id_time - tbatch_time > tbatch_timespan:
                tbatch_time = id_time
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # reset loss
                loss = 0
                embedding_item.detach_()
                embedding_user.detach_()
                embedding_user_timeserie.detach_()
                embedding_item_timeserie.detach_()

        mrr_val = np.mean([1.0 / r for r in val_rank])
        recall10_val = sum(np.array(val_rank) <= 10) * 1.0 / len(val_rank)
        
        mrr_test = np.mean([1.0 / r for r in test_rank])
        recall10_test = sum(np.array(test_rank) <= 10) * 1.0 / len(test_rank)

        perf_val = {}
        perf_test = {}

        perf_val["val"] = [mrr_val, recall10_val]
        perf_test["test"] = [mrr_test, recall10_test]

        file = open(directory+"/resultats/"+data+"/resultats_"+data+"_{}_{}_{}_{}_{}.txt".format(embedding_dim, learning_rate, split, lambda_u, lambda_i), "a")
        metrics = ["Mean Reciprocal Rank", "Recall@10"]
        for i in range(len(metrics)):
            file.write("Validation : " + metrics[i] + " : " + str(perf_val["val"][i]) + "\n")
            file.write("Test : " + metrics[i] + " : " + str(perf_test["test"][i]) + "\n")

        file.flush()
        file.close()

        return perf_val, perf_test