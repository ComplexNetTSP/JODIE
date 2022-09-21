# nos packages
from jodie.preprocessing import *
from jodie.model import *

# jodie packages
import time
from tqdm import tqdm

from torch.autograd import Variable
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
#from ray import tune
import sys
import jodie.train as t
from ray.air import session

def train_ray(config):
    
    save_param(config["embedding_dim"], config["learning_rate"],
               config["split"], config["lambda_u"], config["lambda_i"], config["dataset"], config["directory"])

    fichier = open(config["directory"]+"/"+config["dataset"]+"_hyper-parameter.txt", "a")
    fichier.write("{},{},{},{},{}".format(
        config["embedding_dim"], config["learning_rate"], config["split"], config["lambda_u"], config["lambda_i"]))
    fichier.write("\n")
    fichier.close()
    
    data = fetch_datasets(config["dataset"], config["directory"] + "/data/")

    df = data.to_numpy()
    user_id, id_user_sequence, delta_u, previous_item_sequence, item_id, id_item_sequence, delta_i, timestamp_sequence, feature_sequence, true_labels = preprocess(
        df)
    num_interaction = len(id_user_sequence)
    num_users = len(user_id)
    num_items = len(item_id) + 1
    num_features = len(feature_sequence[0])
    ratio_label = len(true_labels) / (1 + sum(true_labels))
    activation_rnn = "tanh"
    MLP_hidden_layer_dim = 50

    device = config["device"]
    nb_epoch = config["n_epoch"]
    state_change = config["state"]
    proportion_train = config["prop_train"]

    idx_train = int(num_interaction * proportion_train)
    idx_val = int(num_interaction *
                  (proportion_train + (1 - proportion_train) / 2))
    idx_test = int(num_interaction)

    loss_train = []

    # initialize model and parameters
    model = RODIE(int(config["embedding_dim"]), num_users, num_items, num_features,
                  activation_rnn="tanh", MLP_hidden_layer_dim=50).to(device)
    weight = torch.Tensor([1, ratio_label]).to(device)
    CE_loss = nn.CrossEntropyLoss(weight=weight)
    MSE = nn.MSELoss()


    # initialize embedding
    init_embedding_user = nn.Parameter(F.normalize(
        torch.rand(config["embedding_dim"]).to(device), dim=0))
    init_embedding_item = nn.Parameter(F.normalize(
        torch.rand(config["embedding_dim"]).to(device), dim=0))
    model.init_embedding_user = init_embedding_user
    model.init_embedding_item = init_embedding_item

    # same initialize
    embedding_user = init_embedding_user.repeat(num_users, 1)
    embedding_item = init_embedding_item.repeat(num_items, 1)
    embedding_user_static = Variable(torch.eye(num_users).to(device))
    embedding_item_static = Variable(torch.eye(num_items).to(device))
    
    # initialize model
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)

    # epoch
    tbatch_empty = True

    for ep in range(nb_epoch):

        print("Epoch : {}, emb_dim : {}, learning rate : {}, split : {}, lambda_u : {}, lambda_i : {}".format(
            ep+1, config["embedding_dim"], config["learning_rate"], config["split"], config["lambda_u"], config["lambda_i"]))

        optimizer.zero_grad()
        init_tbatch()

        total_loss = 0
        loss_tbatch = 0

        embedding_user_timeserie = Variable(torch.Tensor(
            num_interaction, config["embedding_dim"]).to(device))
        embedding_item_timeserie = Variable(torch.Tensor(
            num_interaction, config["embedding_dim"]).to(device))

        if tbatch_empty:
            tbatch_empty = False
            storage_tbatch_user, storage_tbatch_item, storage_tbatch_timestamp, storage_tbatch_feature, storage_tbatch_label, storage_tbatch_user_timediffs, storage_tbatch_item_timediffs, storage_tbatch_previous_item, storage_tbatch_interactionids, id_time_sequence = t_batch(
                id_user_sequence, id_item_sequence, timestamp_sequence, feature_sequence, true_labels, previous_item_sequence, delta_u, delta_i, idx_train, config["split"])
            init_tbatch()

        # , total = len(storage_tbatch_user.items()), desc = 'Progress bar'):
        for stop, time in enumerate(id_time_sequence):
            actual_tbatch_user = storage_tbatch_user[time]
            actual_tbatch_item = storage_tbatch_item[time]
            actual_tbatch_feature = storage_tbatch_feature[time]
            actual_tbatch_user_timediffs = storage_tbatch_user_timediffs[time]
            actual_tbatch_item_timediffs = storage_tbatch_item_timediffs[time]
            actual_tbatch_interactionids = storage_tbatch_interactionids[time]
            actual_tbatch_previous_item = storage_tbatch_previous_item[time]

            for i, _ in enumerate(storage_tbatch_user[time]):
                if ep == 0:
                    actual_tbatch_user[i] = torch.LongTensor(
                        actual_tbatch_user[i]).to(device)
                    actual_tbatch_item[i] = torch.LongTensor(
                        actual_tbatch_item[i]).to(device)
                    actual_tbatch_interactionids[i] = torch.LongTensor(
                        actual_tbatch_interactionids[i]).to(device)
                    actual_tbatch_previous_item[i] = torch.LongTensor(
                        actual_tbatch_previous_item[i]).to(device)

                    actual_tbatch_user_timediffs[i] = torch.Tensor(
                        actual_tbatch_user_timediffs[i]).to(device)
                    actual_tbatch_item_timediffs[i] = torch.Tensor(
                        actual_tbatch_item_timediffs[i]).to(device)

                    actual_tbatch_feature[i] = torch.Tensor(
                        np.array(actual_tbatch_feature[i])).to(device)

                tbatch_user = actual_tbatch_user[i]
                tbatch_item = actual_tbatch_item[i]
                tbatch_interactionids = actual_tbatch_interactionids[i]

                feature_tensor = Variable(actual_tbatch_feature[i])
                delta_u_tensor = Variable(
                    actual_tbatch_user_timediffs[i]).unsqueeze(1)
                delta_i_tensor = Variable(
                    actual_tbatch_item_timediffs[i]).unsqueeze(1)
                tbatch_previous_item = actual_tbatch_previous_item[i]
                item_embedding_previous = embedding_item[tbatch_previous_item, :]

                # project operator
                projected_embedding_user = model.projection(
                    embedding_user[tbatch_user, :], delta_u_tensor)
                embedding_user_item = torch.cat([projected_embedding_user, item_embedding_previous,
                                                embedding_item_static[tbatch_previous_item, :], embedding_user_static[tbatch_user, :]], dim=1)

                # predict next embedding item
                predict_embedding_item = model.predict_embedding_item(
                    embedding_user_item)

                # calculate loss
                loss_tbatch += MSE(predict_embedding_item, torch.cat(
                    [embedding_item[tbatch_item, :], embedding_item_static[tbatch_item, :]], dim=1).detach())

                # update embedding user and item
                update_embedding_user = model.update_rnn_user(
                    embedding_user[tbatch_user, :], embedding_item[tbatch_item, :], feature_tensor, delta_u_tensor)
                update_embedding_item = model.update_rnn_item(
                    embedding_item[tbatch_item, :], embedding_user[tbatch_user, :], feature_tensor, delta_i_tensor)

                embedding_item[tbatch_item, :] = update_embedding_item
                embedding_user[tbatch_user, :] = update_embedding_user

                embedding_user_timeserie[tbatch_interactionids,
                                         :] = update_embedding_user
                embedding_item_timeserie[tbatch_interactionids,
                                         :] = update_embedding_item

                # calculate loss regularization
                loss_tbatch += regularizer(update_embedding_user,
                                           embedding_user[tbatch_user, :].detach(), config["lambda_u"])
                loss_tbatch += regularizer(update_embedding_item,
                                           embedding_item[tbatch_item, :].detach(), config["lambda_i"])

                if state_change:
                    loss_tbatch += model.loss_predict_state(
                        model, device, tbatch_interactionids, embedding_user_timeserie, true_labels, CE_loss)

            # back propagation
            total_loss += loss_tbatch
            loss_tbatch.backward()
            optimizer.step()
            optimizer.zero_grad()

            # reset loss for next t-batch
            loss_tbatch = 0
            embedding_item.detach_()
            embedding_user.detach_()
            embedding_item_timeserie.detach_()
            embedding_user_timeserie.detach_()
        
        if device == "cpu":
            loss_train.append(total_loss.detach().numpy())
        if device == "cuda":
            loss_train.append(total_loss.detach().cpu().numpy())

        embedding_dynamic_static_item = torch.cat(
            [embedding_item, embedding_item_static], dim=1)
        embedding_dynamic_static_user = torch.cat(
            [embedding_user, embedding_user_static], dim=1)

        embedding_user = init_embedding_user.repeat(num_users, 1)
        embedding_item = init_embedding_item.repeat(num_items, 1)

        if ep == nb_epoch - 1:
            save_model(model, optimizer, ep+1, loss_train, embedding_dynamic_static_user, embedding_dynamic_static_item, idx_train, embedding_user_timeserie,
                       embedding_item_timeserie, config["embedding_dim"], config["learning_rate"], config["split"], config["lambda_u"], config["lambda_i"], config["dataset"], config["directory"])
