from __future__ import absolute_import, division, print_function, unicode_literals
from preprocessing import prepare_data, regression_f_test, recursive_feature_elim, item_selection, select_sorted_items
from sklearn.model_selection import GridSearchCV

import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import json
import datetime
import gc
import math

# current directory
parent_dir = os.path.dirname(os.path.realpath(__file__))

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#TRAIN_SPLIT = 10
tf.random.set_seed(13)
STEP = 1

# =========== UNIVARIATE SINGLE STEP FUNCTIONS =========== 
def univariate_data(dataset, start_index, end_index, history_size, target_size):
	data = []
	labels = []

	start_index = start_index + history_size
	if end_index is None:
		end_index = len(dataset) - target_size

	for i in range(start_index, end_index):
		indices = range(i-history_size, i)
		# Reshape data from (history_size,) to (history_size, 1)
		print(dataset[indices])
		#print()
		data.append(np.reshape(dataset[indices], (history_size,1)))
		labels.append(dataset[i+target_size])
	return np.array(data), np.array(labels)

def univariate_rnn(df, item_to_predict, save_model=True, verbose=1, past_history=5, BATCH_SIZE=32, BUFFER_SIZE=30, \
	EVALUATION_INTERVAL=200, EPOCHS=10, lstm_units=8):
	uni_data = df[item_to_predict]
	uni_data = uni_data.values
	#end=len

	#if past_history > len(uni_data):
	#		past_history=len(uni_data)-1
	split=math.ceil(len(uni_data)/2)
	univariate_past_history = past_history
	univariate_future_target = 0
	x_train_uni, y_train_uni = univariate_data(uni_data, 0, split,
											univariate_past_history,
											univariate_future_target)
	x_val_uni, y_val_uni = univariate_data(uni_data, split, None,
										univariate_past_history,
										univariate_future_target)

	train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
	train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

	val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
	val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

	simple_lstm_model = tf.keras.models.Sequential([
		tf.keras.layers.LSTM(lstm_units, input_shape=x_train_uni.shape[-2:]),
		tf.keras.layers.Dense(1)
	])

	simple_lstm_model.compile(optimizer='adam', loss='mae')

	simple_lstm_history = simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
						steps_per_epoch=EVALUATION_INTERVAL,
						validation_data=val_univariate, validation_steps=50, verbose=verbose)

	if (save_model):
		simple_lstm_model.save(os.path.join(parent_dir, 'models/{}_uni_model.h5'.format(item_to_predict)))

		# open output file for writing
		with open(os.path.join(parent_dir,'models/features/{}_uni_features.txt'.format(item_to_predict)), 'w') as filehandle:
			json.dump(df.columns.values.tolist(), filehandle)
	
	return simple_lstm_history.history

def create_time_steps(length):
	time_steps = []
	for i in range(-length, 0, 1):
		time_steps.append(i)
	return time_steps

def show_plot(plot_data, delta, title):
	labels = ['History', 'True Future', 'Model Prediction']
	marker = ['.-', 'rx', 'go']
	time_steps = create_time_steps(plot_data[0].shape[0])
	if delta:
		future = delta
	else:
		future = 0

	plt.title(title)
	for i, _ in enumerate(plot_data):
		if i:
			plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
		else:
			plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
	plt.legend()
	plt.xlim([time_steps[0], (future+5)*2])
	plt.xlabel('Time-Step')
	return plt

def apply_univariate_test(df, item_to_predict, model, item_std, item_mean, past_history=5, BATCH_SIZE=32):

	uni_data = df[item_to_predict]
	uni_data = uni_data.values
	univariate_past_history = past_history
	univariate_future_target = 0
	split=math.floor((len(uni_data)/2)-1)

	x_val_uni, y_val_uni = univariate_data(uni_data, split, None,
										univariate_past_history,
										univariate_future_target)
	val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
	val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

	#### Unnormalizing the data (so we can see actual prices in GP)
	def unnormalized(val):
		return (val*item_std) + item_mean

	for x, y in val_univariate.take(4):
		plot = show_plot([unnormalized(x[0].numpy()), unnormalized(y[0].numpy()),
						unnormalized(model.predict(x)[0])], 0, 'Simple LSTM model - unnormalized')
		plot.show()

# =========== MULTIVARIATE SINGLE STEP FUNCTIONS =========== 
def multivariate_data(dataset, target, start_index, end_index, history_size,
					  target_size, step, single_step=False):
	data = []
	labels = []

	start_index = start_index + history_size
	if end_index is None:
		end_index = len(dataset) - target_size

	for i in range(start_index, end_index):
		indices = range(i-history_size, i, step)
		data.append(dataset[indices])

		if single_step:
			labels.append(target[i+target_size])
		else:
			labels.append(target[i:i+target_size])


	return np.array(data), np.array(labels)


def plot_train_history(history, title):
	loss = history.history['loss']
	val_loss = history.history['val_loss']

	epochs = range(len(loss))

	plt.figure()

	plt.plot(epochs, loss, 'b', label='Training loss')
	plt.plot(epochs, val_loss, 'r', label='Validation loss')
	plt.title(title)
	plt.legend()

	plt.show()

def multivariate_rnn_single(df, item_to_predict, save_model=True, verbose=1, past_history=5, BATCH_SIZE=32, BUFFER_SIZE=30, \
	EVALUATION_INTERVAL=200, EPOCHS=10, num_dropout=1, lstm_units=32, learning_rate=0.001):
	dataset = df.values
	future_target = 1
	STEP = 1
	split=math.ceil(len(dataset)/2)

	item_to_predict_index = df.columns.get_loc(item_to_predict)

	x_train_single, y_train_single = multivariate_data(dataset, dataset[:, item_to_predict_index], 0,
													split, past_history,
													future_target, STEP,
													single_step=True)
	x_val_single, y_val_single = multivariate_data(dataset, dataset[:, item_to_predict_index],
												split, None, past_history,
												future_target, STEP,
												single_step=True)

	train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
	train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

	val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
	val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

	single_step_model = tf.keras.models.Sequential()
	single_step_model.add(tf.keras.layers.LSTM(lstm_units, input_shape=x_train_single.shape[-2:]))
	single_step_model.add(tf.keras.layers.Dense(1))
	for _ in range(num_dropout):
		single_step_model.add(tf.keras.layers.Dropout(0.2))
		single_step_model.add(tf.keras.layers.Dense(1))

	single_step_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mae') #learning_rate=0.001

	single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
												steps_per_epoch=EVALUATION_INTERVAL,
												validation_data=val_data_single,
												validation_steps=50, verbose=verbose)

	# plot_train_history(single_step_history, 'Single Step Training and validation loss')

	if (save_model):
		# save model to models folder and features to models/features
		single_step_model.save(os.path.join(parent_dir,'models/{}_multiS_model.h5'.format(item_to_predict)))

		with open(os.path.join(parent_dir,'models/features/{}_multiS_features.txt'.format(item_to_predict)), 'w') as filehandle:
			json.dump(df.columns.values.tolist(), filehandle)

	return single_step_history.history

def apply_multivariate_single_step_test(df, item_to_predict, model, item_std, item_mean, past_history=5, BATCH_SIZE=32):
	dataset = df.values
	future_target = 1
	item_to_predict_index = df.columns.get_loc(item_to_predict)
	split=math.ceil(len(dataset)/2)
	x_val_single, y_val_single = multivariate_data(dataset, dataset[:, item_to_predict_index],
												split, None, past_history,
												future_target, STEP,
												single_step=True)
	val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
	val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

	#### Unnormalizing the data (so we can see actual prices in GP)
	def unnormalized(val):
		return (val*item_std) + item_mean

	for x, y in val_data_single.take(3):
		plot = show_plot([unnormalized(x[0][:, item_to_predict_index].numpy()), unnormalized(y[0].numpy()),
							unnormalized(model.predict(x)[0])], 1, 'Single Step Prediction - unnormalized')
		plot.show()

# =========== MULTIVARIATE MULTI STEP FUNCTIONS =========== 
def multi_step_plot(history, true_future, prediction, item_to_predict_index, save_imgs=True, img_title="plot", index=0, item_to_predict=""):

	mode = 0o666
	img_dir = os.path.join(parent_dir,'imgs/{}'.format(item_to_predict))
	if not os.path.exists(item_to_predict): os.makedirs(item_to_predict, mode)

	fig = plt.figure(figsize=(12, 6))
	num_in = create_time_steps(len(history))
	num_out = len(true_future)

	plt.plot(num_in, np.array(history[:, item_to_predict_index]), label='History')
	plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
			label='True Future')
	if prediction.any():
		plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
				label='Predicted Future')
	
	#get the last history value
	#?
	#real perfect profit is last real history minus future real history
	#?
	##true_future - np.roll(true_future, -1)
	#predicted profit 
	#?
	##predicted - np.roll(true_future, -1)

	plt.legend(loc='upper left')
	plt.title(item_to_predict)

	#closest_real_values = buy_avg.iloc[buy_avg.index.get_loc(int(temp_last_row[0]), method='nearest')]
	#real_val = int(closest_real_values[item_predicted])
	#pred_val = int(last_row[prediction_index])
	#data[item_predicted] = [real_val, pred_val, pred_val-real_val]

	if (save_imgs): 
		fig.savefig(os.path.join(item_to_predict,'{}_{}.png'.format(item_to_predict,index)))
	plt.show()

def multivariate_rnn_multi(df, item_to_predict, save_model=True, verbose=1, future_target=5, past_history=5, \
	BATCH_SIZE=32, BUFFER_SIZE=30, EVALUATION_INTERVAL=200, EPOCHS=10, num_dropout=1, lstm_units=64, learning_rate=0.001):
	dataset = df.values
	item_to_predict_index = df.columns.get_loc(item_to_predict)
	split=math.floor((len(dataset)/2)-1)

	x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, item_to_predict_index], 0,
													split, past_history,
													future_target, STEP)
	#if you get errors here , check the object, if it's an Object array the history is larger than the dataset split
	# you need for the split to be less than half - history value you have set.  
	# if you have an index of length 25, and you are splitting in half, so 13 if rounding up
	# you cannot have the history to be more than (25-12-5)
	x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, item_to_predict_index],
												split, None, past_history,
												future_target, STEP)

	train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))

	train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()


	val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))

	val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

	multi_step_model = tf.keras.models.Sequential()
	multi_step_model.add(tf.keras.layers.LSTM(int(lstm_units),
											return_sequences=True,
											input_shape=x_train_multi.shape[-2:]))
	# multi_step_model.add(tf.keras.layers.LSTM(32, return_sequences=True))
	multi_step_model.add(tf.keras.layers.LSTM(int(lstm_units/2), activation='sigmoid'))
	multi_step_model.add(tf.keras.layers.Dense(future_target)) 

	for _ in range(num_dropout):
		multi_step_model.add(tf.keras.layers.Dropout(0.5))
		multi_step_model.add(tf.keras.layers.Dense(future_target))

	# , kernel_regularizer=tf.keras.regularizers.l2(0.04)
	# multi_step_model.add(tf.keras.layers.BatchNormalization())

	multi_step_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mae') # clipvalue=1.0, 

	multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
											steps_per_epoch=EVALUATION_INTERVAL,
											validation_data=val_data_multi,
											validation_steps=50, verbose=verbose)

	if (save_model):
		# save model to models folder and features to models/features
		multi_step_model.save(os.path.join(parent_dir,'models/{}_multiM_model.h5'.format(item_to_predict)))

		with open(os.path.join(parent_dir,'models/features/{}_multiM_features.txt'.format(item_to_predict)), 'w') as filehandle:
			json.dump(df.columns.values.tolist(), filehandle)
	
	return multi_step_history.history

def apply_multivariate_multi_step_test(df, item_to_predict, model, item_std, item_mean, future_target=5, past_history=5, BATCH_SIZE=32):
	dataset = df.values
	item_to_predict_index = df.columns.get_loc(item_to_predict)
	split=math.ceil(len(dataset)/2)
	x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, item_to_predict_index],
												split, None, past_history,
												future_target, STEP)

	val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
	val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

	#### Unnormalizing the data (so we can see actual prices in GP)
	def unnormalized(val):
		return (val*item_std) + item_mean
	
	countindex=0
	for x, y in val_data_multi.take(3):
		multi_step_plot(unnormalized(x[0].numpy()), unnormalized(y[0].numpy()), unnormalized(model.predict(x)[0]), item_to_predict_index,item_to_predict=item_to_predict,index=countindex)
		countindex+=1

# =========== HYPERPARAMETER TUNING FUNCTIONS =========== 
def multivariate_rnn_multi_hyperparameter_tuning(df, item_to_predict, batch_size=[32], buffer_size = [30], \
	epochs = [20], eval_interval = [100], num_dropout_layers = [2],	num_lstm_units = [64], \
		learning = [0.001], past_history = [30]):

	# Write results to file
	current_time = datetime.datetime.utcnow()
	HP_FILE = os.path.join(parent_dir,'data/hp-tuning/{}_MultiM.txt'.format(current_time.strftime("%m-%d-%Y")))

	with open(HP_FILE, 'a') as the_file:
		the_file.write('\nHyperparameter Tuning - item: {}, features: {} - {}\n\n'.format(item_to_predict, \
			len(df.columns)-1, current_time))

	lowest_loss, lowest_std = 100, 100
	best_config = "none"
	for a in batch_size:
		for b in buffer_size:
			for c in epochs:
				for d in eval_interval:
					for e in num_dropout_layers:
						for f in num_lstm_units:
							for g in learning:
								for h in past_history:
									result = multivariate_rnn_multi(df, item_to_predict, save_model=False, verbose=0, \
										BATCH_SIZE=a, BUFFER_SIZE=b, EVALUATION_INTERVAL=d, EPOCHS=c, num_dropout=e, lstm_units=f, learning_rate=g, past_history=h)
									loss_array = np.array(result['val_loss'][-5:])  # make array of last 5 validation loss values
									current_config = "batch-{}_buffer-{}_epoch-{}_eval-{}_drop-{}_lstm-{}_learn-{}_hist-{}".format(a,b,c,d,e,f,g,h)
									mean_loss = np.mean(loss_array)
									std_loss = np.std(loss_array)
									if (mean_loss < lowest_loss):
										lowest_loss = mean_loss
										lowest_std = std_loss
										best_config = current_config
									print("config: {}, mean: {}, std: {}".format(current_config, mean_loss, std_loss))
									with open(HP_FILE, 'a') as the_file:
										the_file.write("config: {}, mean: {}, std: {}\n".format(current_config, mean_loss, std_loss))

	print("BEST CONFIG: {}, mean: {}, std: {}".format(best_config, lowest_loss, lowest_std))
	with open(HP_FILE, 'a') as the_file:
		the_file.write("BEST CONFIG: {}, mean: {}, std: {}\n\n".format(best_config, lowest_loss, lowest_std))

def multivariate_rnn_single_hyperparameter_tuning(df, item_to_predict, batch_size=[32], buffer_size = [30], \
	epochs = [20], eval_interval = [100], num_dropout_layers = [2],	num_lstm_units = [32], \
		learning = [0.001], past_history = [30]):

	# Write results to file
	current_time = datetime.datetime.utcnow()
	HP_FILE = os.path.join(parent_dir,'data/hp-tuning/{}_MultiS.txt'.format(current_time.strftime("%m-%d-%Y")))

	with open(HP_FILE, 'a') as the_file:
		the_file.write('\nHyperparameter Tuning - item: {}, features: {} - {}\n\n'.format(item_to_predict, \
			len(df.columns)-1, current_time))

	lowest_loss, lowest_std = 100, 100
	best_config = "none"
	for a in batch_size:
		for b in buffer_size:
			for c in epochs:
				for d in eval_interval:
					for e in num_dropout_layers:
						for f in num_lstm_units:
							for g in learning:
								for h in past_history:
									result = multivariate_rnn_single(df, item_to_predict, save_model=False, verbose=0, \
										BATCH_SIZE=a, BUFFER_SIZE=b, EVALUATION_INTERVAL=d, EPOCHS=c, num_dropout=e, lstm_units=f, learning_rate=g, past_history=h)
									loss_array = np.array(result['val_loss'][-5:])  # make array of last 5 validation loss values
									current_config = "batch-{}_buffer-{}_epoch-{}_eval-{}_drop-{}_lstm-{}_learn-{}_hist-{}".format(a,b,c,d,e,f,g,h)
									mean_loss = np.mean(loss_array)
									std_loss = np.std(loss_array)
									if (mean_loss < lowest_loss):
										lowest_loss = mean_loss
										lowest_std = std_loss
										best_config = current_config
									print("config: {}, mean: {}, std: {}".format(current_config, mean_loss, std_loss))
									with open(HP_FILE, 'a') as the_file:
										the_file.write("config: {}, mean: {}, std: {}\n".format(current_config, mean_loss, std_loss))

	print("BEST CONFIG: {}, mean: {}, std: {}".format(best_config, lowest_loss, lowest_std))
	with open(HP_FILE, 'a') as the_file:
		the_file.write("BEST CONFIG: {}, mean: {}, std: {}\n\n".format(best_config, lowest_loss, lowest_std))

def univariate_rnn_hyperparameter_tuning(df, item_to_predict, batch_size=[32], buffer_size = [30], \
	epochs = [20], eval_interval = [100],	num_lstm_units = [8], past_history = [30]):

	# Write results to file
	current_time = datetime.datetime.utcnow()
	HP_FILE = os.path.join(parent_dir,'data/hp-tuning/{}_Uni.txt'.format(current_time.strftime("%m-%d-%Y")))

	with open(HP_FILE, 'a') as the_file:
		the_file.write('\nHyperparameter Tuning - item: {}, features: {} - {}\n\n'.format(item_to_predict, \
			len(df.columns)-1, current_time))

	lowest_loss, lowest_std = 100, 100
	best_config = "none"
	for a in batch_size:
		for b in buffer_size:
			for c in epochs:
				for d in eval_interval:
					for f in num_lstm_units:
						for h in past_history:
							result = univariate_rnn(df, item_to_predict, save_model=False, verbose=0, \
								BATCH_SIZE=a, BUFFER_SIZE=b, EVALUATION_INTERVAL=d, EPOCHS=c, lstm_units=f, past_history=h)
							loss_array = np.array(result['val_loss'][-5:])  # make array of last 5 validation loss values
							current_config = "batch-{}_buffer-{}_epoch-{}_eval-{}_lstm-{}_hist-{}".format(a,b,c,d,f,h)
							mean_loss = np.mean(loss_array)
							std_loss = np.std(loss_array)
							if (mean_loss < lowest_loss):
								lowest_loss = mean_loss
								lowest_std = std_loss
								best_config = current_config
							print("config: {}, mean: {}, std: {}".format(current_config, mean_loss, std_loss))
							with open(HP_FILE, 'a') as the_file:
								the_file.write("config: {}, mean: {}, std: {}\n".format(current_config, mean_loss, std_loss))

	print("BEST CONFIG: {}, mean: {}, std: {}".format(best_config, lowest_loss, lowest_std))
	with open(HP_FILE, 'a') as the_file:
		the_file.write("BEST CONFIG: {}, mean: {}, std: {}\n\n".format(best_config, lowest_loss, lowest_std))

def full_hyperparameter_tuning():
	# items_to_predict = ['Old_school_bond', 'Rune_platebody', 'Rune_2h_sword', 'Rune_axe',\
	# 	'Rune_pickaxe', 'Adamant_platebody', 'Amulet_of_power']
	items_to_predict = item_selection()
	items_to_predict = select_sorted_items(items_to_predict)
	min_features = 2
	max_features = 4
	for item_to_predict in items_to_predict:
		for num_features in range(min_features,max_features):
			# SELECT ITEMS
			items_selected = item_selection()

			# FEATURE EXTRACTION
			preprocessed_df = prepare_data(item_to_predict, items_selected)

			# FEATURE SELECTION & NORMALIZATION
			selected_df, pred_std, pred_mean = regression_f_test(preprocessed_df, item_to_predict, number_of_features=num_features)
			# print(selected_df.head())

			# define the grid search parameters
			batch_size = [16, 32, 64, 128]
			buffer_size = [30,50,100]
			epochs = [20,40]
			eval_interval = [100,400]
			num_dropout_layers = [1,2,3]
			num_lstm_units = [32,64,128]
			learning = [0.0001]
			past_history= [30,50]
			
			# multivariate_rnn_multi_hyperparameter_tuning(selected_df, item_to_predict, eval_interval=eval_interval, \
			# 	learning=learning, past_history=past_history, epochs=epochs, num_lstm_units=num_lstm_units, batch_size=batch_size,\
			# 		 buffer_size=buffer_size, num_dropout_layers=num_dropout_layers)
			# multivariate_rnn_single_hyperparameter_tuning(selected_df, item_to_predict, eval_interval=eval_interval, \
			# 	learning=learning, past_history=past_history, epochs=epochs, num_lstm_units=num_lstm_units, batch_size=batch_size,\
			# 		buffer_size=buffer_size, num_dropout_layers=num_dropout_layers)
			# univariate_rnn_hyperparameter_tuning(selected_df, item_to_predict, batch_size = batch_size, epochs= epochs, \
			# 	past_history=past_history, num_lstm_units=num_lstm_units, eval_interval=eval_interval)
			
			multivariate_rnn_single_hyperparameter_tuning(selected_df, item_to_predict, \
				num_lstm_units=[128], past_history=[30], eval_interval=[400], num_dropout_layers=[2], learning = [0.0001])
			# multivariate_rnn_multi_hyperparameter_tuning(selected_df, item_to_predict, \
			# 	num_lstm_units=num_lstm_units, past_history=past_history, eval_interval=eval_interval)
			# univariate_rnn_hyperparameter_tuning(selected_df, item_to_predict, \
			# 	past_history=range(30,50,5), num_lstm_units=[8], eval_interval=eval_interval)

			# univariate_rnn_hyperparameter_tuning(selected_df, item_to_predict)
			
			del selected_df
			del preprocessed_df
			gc.collect()

def main():
	# items_to_predict = item_selection()
	# items_to_predict = select_sorted_items(items_to_predict)
	items_to_predict = ['Mithril_bar','Air_battlestaff','Red_chinchompa','Manta_ray','Saradomin_brew(4)','Anglerfish','Purple_sweets','Anti-venom+(4)','Cactus_spine']
	num_features = 2

	for item_to_predict in items_to_predict:
		# =========== PREPROCESSING =========== 
		# SELECT ITEMS
		items_selected = item_selection()

		print("Learning [{}]".format(item_to_predict))

		# FEATURE EXTRACTION
		preprocessed_df = prepare_data(item_to_predict, items_selected)

		# FEATURE SELECTION & NORMALIZATION
		selected_df, pred_std, pred_mean = regression_f_test(preprocessed_df, item_to_predict, number_of_features=num_features)
		#TRAIN_SPLIT=math.ceil((len(selected_df)/2))
		# print(selected_df.head())
		# print(selected_df.shape)
		# print("columns with nan: {}".format(selected_df.columns[selected_df.isna().any()].tolist()))


		# =========== UNIVARIATE =========== 
		uni_config = {}
		# TRAINING AND SAVING MODEL
		print("On [{}]".format('UNIVARIATE'))
		univariate_rnn(selected_df, item_to_predict)

		# # LOADING AND APPLYING MODEL
		#os.path.join(parent_dir,'models/{}_multiM_model.h5'.format(item_to_predict))
		loaded_model = tf.keras.models.load_model(os.path.join(parent_dir,'models/{}_uni_model.h5'.format(item_to_predict)))
		apply_univariate_test(selected_df, item_to_predict, loaded_model, pred_std, pred_mean)


		# =========== MULTIVARIATE SINGLE STEP ===========
		multiS_config = {'lstm_units':64, 'EVALUATION_INTERVAL':10, 'EPOCHS':10, 'learning_rate':0.0001, 'num_dropout': 2}
		# TRAINING AND SAVING MODEL
		print("On [{}]".format('MULTIVARIATE SINGLE STEP'))
		multivariate_rnn_single(selected_df, item_to_predict, **multiS_config)

		# # LOADING AND APPLYING MODEL
		loaded_model = tf.keras.models.load_model(os.path.join(parent_dir,'models/{}_multiS_model.h5'.format(item_to_predict)))
		apply_multivariate_single_step_test(selected_df, item_to_predict, loaded_model, pred_std, pred_mean)


		# =========== MULTIVARIATE MULTI STEP ===========
		multiM_config = {'lstm_units':128, 'EVALUATION_INTERVAL':10, 'EPOCHS':15, 'learning_rate':0.0001, 'num_dropout': 2}
		# TRAINING AND SAVING MODEL
		print("On [{}]".format('MULTIVARIATE MULTI STEP'))
		multivariate_rnn_multi(selected_df, item_to_predict, **multiM_config)

		# # LOADING AND APPLYING MODEL
		loaded_model = tf.keras.models.load_model(os.path.join(parent_dir,'models/{}_multiM_model.h5'.format(item_to_predict)))
		apply_multivariate_multi_step_test(selected_df, item_to_predict, loaded_model, pred_std, pred_mean)


	# # # =========== HYPERPARAMETER TUNING ===========
	# full_hyperparameter_tuning()
		
if __name__ == "__main__":
	main()