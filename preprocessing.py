# nos packages
import numpy as np
import pandas as pd
import torch
import itertools
from collections import defaultdict
import tray2 as t
from tray2 import *

# jodie packages
from sklearn.preprocessing import scale

# preprocessing data
def preprocess(df_to_numpy):

	user_sequence = df_to_numpy[:, 0]
	item_sequence = df_to_numpy[:, 1]
	timestamp_sequence = df_to_numpy[:, 2]
	true_labels = df_to_numpy[:, 3]
	feature_sequence = df_to_numpy[:, 4:]
	
	item_id = {}
	node_item = 0
	delta_i = []
	timestamp_item_current = defaultdict(float)
	
	for idx, item in enumerate(item_sequence):
		if item not in item_id:
			item_id[item] = node_item
			node_item += 1
		timestamp_item = timestamp_sequence[idx]
		delta_i.append(timestamp_item - timestamp_item_current[item])
		timestamp_item_current[item] = timestamp_item
	id_item_sequence = [item_id[item] for item in item_sequence]
	delta_i = scale(np.array(delta_i) + 1)
	num_items = len(item_id)
	
	user_id = {}
	node_user = 0
	delta_u = []
	timestamp_user_current = defaultdict(float)
	previous_item_sequence = []
	latest_item_id_sequence = defaultdict(lambda : num_items)
	for idx, user in enumerate(user_sequence):
		if user not in user_id:
			user_id[user] = node_user
			node_user += 1
		timestamp_user = timestamp_sequence[idx]
		delta_u.append(timestamp_user - timestamp_user_current[user])
		timestamp_user_current[user] = timestamp_user
		previous_item_sequence.append(latest_item_id_sequence[user])
		latest_item_id_sequence[user] = item_id[item_sequence[idx]]
	#num_users = len(user_id)
	id_user_sequence = [user_id[user] for user in user_sequence]
	delta_u = scale(np.array(delta_u) + 1)
		
	return [user_id, id_user_sequence, delta_u, previous_item_sequence, item_id, id_item_sequence, delta_i, timestamp_sequence, feature_sequence, true_labels]
	
# initialize t-batch
def init_tbatch():
	global tbatch_interactionids, tbatch_user, tbatch_item, tbatch_timestamp, tbatch_feature, tbatch_label, tbatch_previous_item
	global tbatch_id_user, tbatch_id_item, tbatch_user_timediffs, tbatch_item_timediffs

	t.tbatch_interactionids = defaultdict(list)
	t.tbatch_user = defaultdict(list)
	t.tbatch_item = defaultdict(list)
	t.tbatch_timestamp = defaultdict(list)
	t.tbatch_feature = defaultdict(list)
	t.tbatch_label = defaultdict(list)
	t.tbatch_previous_item = defaultdict(list)
	t.tbatch_user_timediffs = defaultdict(list)
	t.tbatch_item_timediffs = defaultdict(list)
	
	t.tbatch_id_user = defaultdict(lambda : -1)
	t.tbatch_id_item = defaultdict(lambda : -1)
	
# t-batch
def t_batch(id_user_sequence, id_item_sequence, timestamp_sequence, feature_sequence, y_true_labels, previous_item_sequence, delta_u, delta_i, proportion_train, split = 500):
	
	tbatch_time = None
	
	storage_tbatch_user = {}
	storage_tbatch_item = {}
	storage_tbatch_timestamp = {}
	storage_tbatch_feature = {}
	storage_tbatch_label = {}
	storage_tbatch_user_timediffs = {}
	storage_tbatch_item_timediffs = {}
	storage_tbatch_previous_item = {}
	storage_tbatch_interactionids = {}
	
	tbatch_timespan = (timestamp_sequence[-1] - timestamp_sequence[0]) / split
	id_time_sequence = []
	
	for j in range(proportion_train):
		id_user = id_user_sequence[j]
		id_item = id_item_sequence[j]
		id_time = timestamp_sequence[j]
		id_feature = feature_sequence[j]
		id_label = y_true_labels[j]
		id_delta_u = delta_u[j]
		id_delta_i = delta_i[j]
		id_previous = previous_item_sequence[j]
		
		idx_tbatch = max(t.tbatch_id_user[id_user], t.tbatch_id_item[id_item]) + 1
		
		t.tbatch_id_user[id_user] = idx_tbatch
		t.tbatch_id_item[id_item] = idx_tbatch
		
		t.tbatch_interactionids[idx_tbatch].append(j)
		t.tbatch_user[idx_tbatch].append(id_user)
		t.tbatch_item[idx_tbatch].append(id_item)
		t.tbatch_timestamp[idx_tbatch].append(id_time)
		t.tbatch_feature[idx_tbatch].append(id_feature)
		t.tbatch_label[idx_tbatch].append(id_label)
		t.tbatch_user_timediffs[idx_tbatch].append(id_delta_u)
		t.tbatch_item_timediffs[idx_tbatch].append(id_delta_i)
		t.tbatch_previous_item[idx_tbatch].append(id_previous)
		
		if tbatch_time is None:
			tbatch_time = id_time
		
		if id_time - tbatch_time > tbatch_timespan:
			id_time_sequence.append(id_time)
			tbatch_time = id_time
			
			storage_tbatch_user[id_time] = t.tbatch_user
			storage_tbatch_item[id_time] = t.tbatch_item
			storage_tbatch_timestamp[id_time] = t.tbatch_timestamp
			storage_tbatch_feature[id_time] = t.tbatch_feature
			storage_tbatch_label[id_time] = t.tbatch_label
			storage_tbatch_user_timediffs[id_time] = t.tbatch_user_timediffs
			storage_tbatch_item_timediffs[id_time] = t.tbatch_item_timediffs
			storage_tbatch_previous_item[id_time] = t.tbatch_previous_item
			storage_tbatch_interactionids[id_time] = t.tbatch_interactionids
			init_tbatch()
	
	return storage_tbatch_user, storage_tbatch_item, storage_tbatch_timestamp, storage_tbatch_feature, storage_tbatch_label, storage_tbatch_user_timediffs, storage_tbatch_item_timediffs, storage_tbatch_previous_item, storage_tbatch_interactionids, id_time_sequence
	
# regularization function	
def regularizer(embedding_future, embedding_past, lambda_r):
	return lambda_r * torch.nn.MSELoss()(embedding_future, embedding_past)

# setting on user and item embeddings to the end of the training period
def set_embedding(id_user_sequence, id_item_sequence, proportion_train, embedding_user_timeserie, embedding_item_timeserie, embedding_dynamic_static_user, embedding_dynamic_static_item):
	
  userid2lastidx = {}
  train_end = int(len(id_user_sequence) * proportion_train)

  for i,j in enumerate(id_user_sequence[: train_end]):
    userid2lastidx[j] = i

  itemid2lastidx = {}
  for i,j in enumerate(id_item_sequence[: train_end]):
    itemid2lastidx[j] = i

  try:
    size = embedding_user_timeserie.size(1)

  except:
    size = embedding_user_timeserie.shape[1]

  for j in userid2lastidx:
    embedding_dynamic_static_user[j, :size] = embedding_user_timeserie[userid2lastidx[j]]

  for j in itemid2lastidx:
    embedding_dynamic_static_item[j, :size] = embedding_item_timeserie[itemid2lastidx[j]]
  
  embedding_dynamic_static_user.detach()
  embedding_dynamic_static_item.detach()