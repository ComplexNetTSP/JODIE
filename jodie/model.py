# jodie packages
from sklearn.preprocessing import scale

# nos packages
from jodie.preprocessing import *

import torch
from torch import nn
from torch.nn import RNNCell
from torch.nn.functional import one_hot
import math
from torch.nn import functional as F
import pandas as pd

import jodie.train as t

import os

# jodie packages
from torch.autograd import Variable

# This custom class of linear, enables to initialize the weights of the layer to belong to a normal distribution


class NormalLinear(nn.Linear):
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv)

# This function enables to create the dynamic embedding of each node


class RODIE(nn.Module):

    def __init__(self, embedding_dim, num_users, num_items, num_features, activation_rnn="tanh", MLP_hidden_layer_dim=50):

        super(RODIE, self).__init__()
        self.num_features = num_features
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.activation_rnn = activation_rnn
        self.MLP_hidden_layer_dim = MLP_hidden_layer_dim

        # initialize embeddings
        self.init_embedding_user = nn.Parameter(
            torch.Tensor(self.embedding_dim))
        self.init_embedding_item = nn.Parameter(
            torch.Tensor(self.embedding_dim))

        # initialize RNNs
        input_rnn_user_dim = self.embedding_dim + self.num_features + 1
        input_rnn_item_dim = self.embedding_dim + self.num_features + 1

        self.rnn_user = RNNCell(
            input_rnn_user_dim, self.embedding_dim, nonlinearity=self.activation_rnn)
        self.rnn_item = RNNCell(
            input_rnn_item_dim, self.embedding_dim, nonlinearity=self.activation_rnn)

        # differents layers (projection, prediction)
        self.layer_projection = NormalLinear(1, self.embedding_dim)
        self.layer_prediction = nn.Linear(
            self.num_users + self.num_items + self.embedding_dim * 2, self.num_items + self.embedding_dim)
        self.layer1_pred_label = nn.Linear(
            self.embedding_dim, self.MLP_hidden_layer_dim)
        self.layer2_pred_label = nn.Linear(self.MLP_hidden_layer_dim, 2)

    # update embedding
    def update_rnn_item(self, embedding_item, embedding_user, features, delta_i):
        concat_input1 = torch.cat(
            [embedding_user, features, delta_i.reshape(-1, 1)], dim=1)
        embedding_item_output = self.rnn_item(concat_input1, embedding_item)
        return F.normalize(embedding_item_output)

    def update_rnn_user(self, embedding_user, embedding_item, features, delta_u):
        concat_input2 = torch.cat(
            [embedding_item, features, delta_u.reshape(-1, 1)], dim=1)
        embedding_user_output = self.rnn_user(concat_input2, embedding_user)
        return F.normalize(embedding_user_output)

    # projection operation
    def projection(self, embedding_user, delta_u):
        projected_user = embedding_user * (1 + self.layer_projection(delta_u))
        return projected_user

    # prediction user state
    def predict_state(self, embedding_user):
        output = nn.ReLU()(self.layer1_pred_label(embedding_user))
        output = self.layer2_pred_label(output)
        return output

    # prediction embedding item
    def predict_embedding_item(self, embedding_user):
        output = self.layer_prediction(embedding_user)
        return output

    # calculate loss for prediction user state
    def loss_predict_state(self, model, device, interaction_id, all_previous_embeddings_user, true_label, loss_function):       
        proba = model.predict_state(
            all_previous_embeddings_user[interaction_id, :])
        y_true = Variable(torch.LongTensor(
            true_label).to(device)[interaction_id])
        loss = loss_function(proba, y_true)
        return loss

# save models
def save_model(model, optimizer, epoch, loss, user_embeddings, item_embeddings, train_end_idx, user_embeddings_time_series, item_embeddings_time_series, embedding_dim, learning_rate, split, lambda_u, lambda_i, dataset, directory):
  state = {
    "user_embeddings" : user_embeddings.cpu().numpy(),
    "item_embeddings" : item_embeddings.cpu().numpy(),
    "epoch" : epoch,
    "loss" : loss,
    "model" : model.state_dict(),
    "optimizer" : optimizer.state_dict(),
    "train_end_idx" : train_end_idx,
    "user_embeddings_time_series" : user_embeddings_time_series.cpu().numpy(),
    "item_embeddings_time_series" : item_embeddings_time_series.cpu().numpy()
  }
  dir = os.path.join(directory+"/saved_models_"+dataset+"/", "saved_model")
  if not os.path.exists(dir):
    os.makedirs(dir)
  filename = os.path.join(dir, "{}_{}_{}_{}_{}_{}".format(embedding_dim, learning_rate, split, lambda_u, lambda_i, epoch))
  torch.save(state, filename)

# save parameters
def save_param(embedding_dim, learning_rate, split, lambda_u, lambda_i, dataset, directory):
  state = {
    "embedding_dim" : embedding_dim,
    "learning_rate" : learning_rate,
    "split" : split,
    "lambda_u" : lambda_u,
    "lambda_i" : lambda_i
  }
  dir = os.path.join(directory+"/saved_params_"+dataset+"/", "saved_param")
  if not os.path.exists(dir):
    os.makedirs(dir)
  filename = os.path.join(dir, "{}_{}_{}_{}_{}".format(embedding_dim, learning_rate, split, lambda_u, lambda_i))
  torch.save(state, filename)

# load parameters
def load_param(embedding_dim, learning_rate, split, lambda_u, lambda_i, dataset, directory):
  filename = directory+"/saved_params_"+dataset+"/saved_param/{}_{}_{}_{}_{}".format(embedding_dim, learning_rate, split, lambda_u, lambda_i)
  checkpoint = torch.load(filename)
  embedding_dim = checkpoint["embedding_dim"]
  learning_rate = checkpoint["learning_rate"]
  split = checkpoint["split"]
  lambda_u = checkpoint["lambda_u"]
  lambda_i = checkpoint["lambda_i"]
  return embedding_dim, learning_rate, split, lambda_u, lambda_i

# load model
def load_model(model, optimizer, epoch, device, embedding_dim, learning_rate, split, lambda_u, lambda_i, dataset, directory):
  filename = directory+"/saved_models_"+dataset+"/saved_model/{}_{}_{}_{}_{}_{}".format(embedding_dim, learning_rate, split, lambda_u, lambda_i, epoch)
  checkpoint = torch.load(filename)
  user_embeddings = Variable(torch.from_numpy(checkpoint["user_embeddings"]).to(device))
  item_embeddings = Variable(torch.from_numpy(checkpoint["item_embeddings"]).to(device))
  train_end_idx = checkpoint["train_end_idx"]
  user_embeddings_time_series = Variable(torch.from_numpy(checkpoint["user_embeddings_time_series"]).to(device))
  item_embeddings_time_series = Variable(torch.from_numpy(checkpoint["item_embeddings_time_series"]).to(device))
  model.load_state_dict(checkpoint["model"])
  optimizer.load_state_dict(checkpoint["optimizer"])
  return model, optimizer, user_embeddings, item_embeddings, user_embeddings_time_series, item_embeddings_time_series, train_end_idx
