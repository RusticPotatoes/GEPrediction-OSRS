from __future__ import absolute_import, division, print_function, unicode_literals
from preprocessing import prepare_data_from_folder, prepare_data_from_df, regression_f_test, recursive_feature_elim, item_selection, select_sorted_items
from sklearn.model_selection import GridSearchCV
from data_scrapers.classes.wrapper import  PricesAPI

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
import time
import imageio
import socket
import itertools

# current directory
parent_dir = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.join(parent_dir,"models")
features_dir = os.path.join(models_dir,"features")

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#TRAIN_SPLIT = 10
tf.random.set_seed(13)
STEP = 1

# ================== Utility FUNCTIONS ================== 
def getIDFromName(df,name):
	return (df[df['name'] == name].item_id.item())

def getNameFromID(df,id):
	return (df[df['item_id'] == id].name.item())

def gif_from_png_dir(item_to_predict,img_dir):
	images = []
	timestr = time.time()#('%Y%m%d-%H%M%S')
	for file_name in sorted(os.listdir(img_dir)):
		if file_name.endswith('.png'):
			file_path = os.path.join(img_dir, file_name)
			images.append(imageio.imread(file_path))
	gifpath= os.path.join(img_dir,'{}_{}.gif'.format(item_to_predict,timestr))
	imageio.mimsave(gifpath, images, fps=1)

def clear_pngs(img_dir):
	if not os.path.exists(img_dir):
		return
	for file in os.listdir(img_dir):
		if file.endswith('.png'):
			os.remove(os.path.join(img_dir,file)) 

def save_plot_to_png(input_plot, filename, subdir):   			
	mode = 0o666
	img_dir = os.path.join(parent_dir,'imgs/{}'.format(item_to_predict))
	if not os.path.exists(img_dir): os.makedirs(img_dir, mode)

	if subdir is not None:
		if not os.path.exists(os.path.join(img_dir,subdir)): os.makedirs(os.path.join(img_dir,subdir), mode)
		img_dir=os.path.join(img_dir,subdir)

	input_plot.savefig(os.path.join(img_dir,name))#'{}_{}.png'.format(item_to_predict,index))

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

def univariate_rnn(df, item_to_predict, price_type_name="", TRAINING_SIZE=10, save_model=True, verbose=1, past_history=5, BATCH_SIZE=32, BUFFER_SIZE=30, \
	EVALUATION_INTERVAL=200, EPOCHS=10, lstm_units=8):
	uni_data = df[item_to_predict]
	uni_data = uni_data.values
	#end=len

	if past_history > len(uni_data):
			past_history=len(uni_data)-1

	#split=math.floor(len(uni_data)/2)
	univariate_past_history = past_history
	univariate_future_target = 5
 
	print("Dataset: {}, Start index: {}, End Index: {}, History Size: {}, Target Size: {}".format(
    										uni_data.shape,  #dataset
											TRAINING_SIZE, #start_index
    										(len(uni_data) - univariate_future_target), #end_index
    										univariate_past_history, #history_size
											univariate_future_target, #target_size
        )
    )

	x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAINING_SIZE,
											univariate_past_history,
											univariate_future_target)
	x_val_uni, y_val_uni = univariate_data(uni_data, TRAINING_SIZE, None,
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

	simple_lstm_model.compile(optimizer='adam', loss='mae', metrics=["acc"])

	simple_lstm_history = simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
						steps_per_epoch=EVALUATION_INTERVAL,
						validation_data=val_univariate, validation_steps=50, verbose=verbose)

	if (save_model):
		simple_lstm_model.save(os.path.join(models_dir, '{}_{}_uni_model.h5'.format(item_to_predict,price_type_name)))

		# open output file for writing
		with open(os.path.join(features_dir,'{}_{}_uni_features.txt'.format(item_to_predict,price_type_name)), 'w') as filehandle:
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

def apply_univariate_test(df, item_to_predict, model, item_std, item_mean, price_type_name="", TRAINING_SIZE=10, past_history=30, BATCH_SIZE=32, save_img=False):
	name = "uv"
	uni_data = df[item_to_predict]
	uni_data = uni_data.values
	univariate_past_history = past_history
	univariate_future_target = 5
	#split=math.floor((len(uni_data)/2)-1)

	#dataset, start_index, end_index, history_size, target_size
	print("Dataset: {}, Start index: {}, End Index: {}, History Size: {}, Target Size: {}".format(
    										uni_data.shape,  #dataset
											TRAINING_SIZE, #start_index
    										(len(uni_data) - univariate_past_history), #end_index
    										univariate_past_history, #history_size
											univariate_future_target, #target_size
        )
    )

	x_val_uni, y_val_uni = univariate_data(uni_data, TRAINING_SIZE, None,
										univariate_past_history,
										univariate_future_target)
	val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
	val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

	#### Unnormalizing the data (so we can see actual prices in GP)
	def unnormalized(val):
		return (val*item_std) + item_mean
	
	countindex=0
	for x, y in val_univariate.take(4):
		plot = show_plot([unnormalized(x[0].numpy()), unnormalized(y[0].numpy()),
						unnormalized(model.predict(x)[0])], 0, 'Simple LSTM model - unnormalized')
		#plot.show(block=True)
		if (save_img): save_plot_to_png(plot, "{}_{}_{}_{}.png".format(name, item_to_predict, price_type_name, countindex), name)
		countindex+=1

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

	fig = plt.figure()

	plt.plot(epochs, loss, 'b', label='Training loss')
	plt.plot(epochs, val_loss, 'r', label='Validation loss')
	plt.title(title)
	plt.legend()
	return fig
	#plt.show(block=True)

def multivariate_rnn_single(df, item_to_predict, price_type_name="", TRAINING_SIZE=10, save_model=True, verbose=1, past_history=5, BATCH_SIZE=32, BUFFER_SIZE=30, \
	EVALUATION_INTERVAL=200, EPOCHS=10, num_dropout=1, lstm_units=32, learning_rate=0.001, save_img=False):
	name="mv_rnn_s"
	dataset = df.values
	future_target = 5
	STEP = 1
	#split=math.ceil(len(dataset)/2)

	item_to_predict_index = df.columns.get_loc(item_to_predict)

	print("Dataset: {}, Target: {}, Start index: {}, End Index: {}, History Size: {}, Target Size: {}, Step: {}".format(
    										dataset.shape,  #dataset
                                            "(size: {}, index col {})".format(len(dataset[:, item_to_predict_index]),item_to_predict_index), #target
											TRAINING_SIZE, #start_index
    										(len(dataset) - future_target), #end_index
    										past_history, #history_size
											future_target, #target_size
        									STEP #step
        )
    )
  
	x_train_single, y_train_single = multivariate_data(dataset, dataset[:, item_to_predict_index], 0,
													TRAINING_SIZE, past_history,
													future_target, STEP,
													single_step=True)
	x_val_single, y_val_single = multivariate_data(dataset, dataset[:, item_to_predict_index],
												TRAINING_SIZE, None, past_history,
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

	single_step_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mae', metrics=["acc"]) #learning_rate=0.001

	single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
												steps_per_epoch=EVALUATION_INTERVAL,
												validation_data=val_data_single,
												validation_steps=50, verbose=verbose)

	plot = plot_train_history(single_step_history, 'Single Step Training and validation loss')
	if (save_img): save_plot_to_png(plot, "{}_{}_{}.png".format(name, item_to_predict, 0), name)
	# plot_train_history(single_step_history, 'Single Step Training and validation loss')
	

	if (save_model):
		# save model to models folder and features to models/features
		single_step_model.save(os.path.join(models_dir,'{}_{}_multiS_model.h5'.format(item_to_predict,price_type_name)))

		with open(os.path.join(features_dir,'{}_{}_multiS_features.txt'.format(item_to_predict,price_type_name)), 'w') as filehandle:
			json.dump(df.columns.values.tolist(), filehandle)

	return single_step_history.history

def apply_multivariate_single_step_test(df, item_to_predict, model, item_std, item_mean, price_type_name="", TRAINING_SIZE=10, past_history=30, BATCH_SIZE=32, save_img=False):
	name = "mv_s_s"
	dataset = df.values
	#amount of predictions to target
	future_target = 5
	# values of item selected in teh dataset 
	item_to_predict_index = df.columns.get_loc(item_to_predict)

	#start_index = split=math.ceil(len(dataset)/2)

	#print("start index: {}".format(start_index))
 
 	#multivariate_data
 	###dataset, target, start_index, end_index, history_size,target_size, step, single_step=False
	print("Dataset: {}, Target: {}, Start index: {}, End Index: {}, History Size: {}, Target Size: {}, Step: {}".format(
    										dataset.shape,  #dataset
                                            "(size: {}, index col {})".format(len(dataset[:, item_to_predict_index]),item_to_predict_index), #target
											TRAINING_SIZE, #start_index
    										(len(dataset) - future_target), #end_index
    										past_history, #history_size
											future_target, #target_size
        									STEP #step
        )
    )
	#validation data
	x_val_single, y_val_single = multivariate_data(dataset, dataset[:, item_to_predict_index],
												TRAINING_SIZE, None, past_history,
												future_target, STEP,
												single_step=True)
 
	val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
	val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

	#### Unnormalizing the data (so we can see actual prices in GP)
	def unnormalized(val):
		return (val*item_std) + item_mean
	countindex=0
	for x, y in val_data_single.take(3):
		plot = show_plot([unnormalized(x[0][:, item_to_predict_index].numpy()), unnormalized(y[0].numpy()),
							unnormalized(model.predict(x)[0])], 1, 'Single Step Prediction - unnormalized')
		#plot.show(block=True)
		if (save_img): save_plot_to_png(plot, "{}_{}_{}_{}.png".format(name, item_to_predict, price_type_name, countindex), name)
		countindex+=1
	
# =========== MULTIVARIATE MULTI STEP FUNCTIONS =========== 
def multi_step_plot(history, true_future, prediction, item_to_predict_index, img_title="plot", index=0, item_to_predict=""):
	#debug vars
	#print("history: [{}] ".format(history))
	#print("true_future: [{}]".format(true_future))
	#print("prediction: [{}]".format(prediction))
	#print("item_to_predict_index: [{}]".format(item_to_predict_index))
	#print("save_imgs: [{}]".format(save_imgs))
	#print("img_title: [{}]".format(img_title))
	#print("index: [{}]".format(index))
	#print("item_to_predict: [{}]".format(item_to_predict))
			
	#mode = 0o666
	#img_dir = os.path.join(parent_dir,'imgs/{}'.format(item_to_predict))
	#if not os.path.exists(img_dir): os.makedirs(img_dir, mode)

	fig = plt.figure(figsize=(12, 6)) # predefined size of figure in inchies 

	history_size = num_in = create_time_steps(len(history))
	#print(num_in)
	future_size = num_out = len(true_future)
	#print(num_out)


	# this is a slice eg for a dataframe[row slicing, column slicing]
	# selects subset of row columns from the dataframe 
	# eg for a (300,6) , select all 300 and only column 5, dataframe[:, 5]
 	# https://datacarpentry.org/python-ecology-lesson/03-index-slice-subset/index.html
	plt.plot(history_size, np.array(history[:, item_to_predict_index]), label='History')

	#TODO add plot prediction line into figure
	#plt.plot(num_in, np.array(prediction[:, item_to_predict_index]), label='Predicted')


	plt.plot(np.arange(future_size)/STEP, np.array(true_future), 'bo',
			label='True Future')
	if prediction.any():
		plt.plot(np.arange(future_size)/STEP, np.array(prediction), 'ro',
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
	#print(os.path.join(img_dir,'{}_{}.png'.format(item_to_predict,index)))
	#if (save_imgs): 
		#fig.savefig(os.path.join(img_dir,'{}_{}.png'.format(item_to_predict,index)))
		#fig.savefig(os.path.join(img_dir,'{}.png'.format(item_to_predict)))
	#plt.show(block=True)
	return fig

def multivariate_rnn_multi(df, item_to_predict, price_type_name="", TRAINING_SIZE=10, save_model=True, verbose=1, future_target=5, past_history=5, \
	BATCH_SIZE=32, BUFFER_SIZE=30, EVALUATION_INTERVAL=200, EPOCHS=10, num_dropout=1, lstm_units=64, learning_rate=0.001, save_img=False):
	name = "mv_rnn_m"
	dataset = df.values
	item_to_predict_index = df.columns.get_loc(item_to_predict)
	#split=math.floor((len(dataset)/2))

	print("Dataset: {}, Target: {}, Start index: {}, End Index: {}, History Size: {}, Target Size: {}, Step: {}".format(
    										dataset.shape,  #dataset
                                            "(size: {}, index col {})".format(len(dataset[:, item_to_predict_index]),item_to_predict_index), #target
											TRAINING_SIZE, #start_index
    										(len(dataset) - future_target), #end_index
    										past_history, #history_size
											future_target, #target_size
        									STEP #step
        )
    )

	x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, item_to_predict_index], 0,
													TRAINING_SIZE, past_history,
													future_target, STEP)
	#if you get errors here , check the object, if it's an Object array the history is larger than the dataset split
	# you need for the split to be less than half - history value you have set.  
	# if you have an index of length 25, and you are splitting in half, so 13 if rounding up
	# you cannot have the history to be more than (25-12-5)
	x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, item_to_predict_index],
												TRAINING_SIZE, None, past_history,
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

	multi_step_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mae', metrics=["acc"]) # clipvalue=1.0, 

	multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
											steps_per_epoch=EVALUATION_INTERVAL,
											validation_data=val_data_multi,
											validation_steps=50, verbose=verbose)
	
	
	if (save_model):
		# save model to models folder and features to models/features
		multi_step_model.save(os.path.join(models_dir,'{}_{}_multiM_model.h5'.format(item_to_predict,price_type_name)))
		with open(os.path.join(features_dir,'{}_{}_multiM_features.txt'.format(item_to_predict,price_type_name)), 'w') as filehandle:
			json.dump(df.columns.values.tolist(), filehandle)
	
	return multi_step_history.history

def apply_multivariate_multi_step_test(df, item_to_predict, model, item_std, item_mean, price_type_name="", TRAINING_SIZE=10, future_target=5, past_history=30, BATCH_SIZE=32, save_img=False):
	name = "mv_m_s"
	img_dir = os.path.join(parent_dir,'imgs/{}'.format(item_to_predict))
	dataset = df.values
	item_to_predict_index = df.columns.get_loc(item_to_predict)
	#split=math.ceil(len(dataset)/3)
 
	print("Dataset: {}, Target: {}, Start index: {}, End Index: {}, History Size: {}, Target Size: {}, Step: {}".format(
    										dataset.shape,  #dataset
                                            "(size: {}, index col {})".format(len(dataset[:, item_to_predict_index]),item_to_predict_index), #target
											TRAINING_SIZE, #start_index
    										(len(dataset) - future_target), #end_index
    										past_history, #history_size
											future_target, #target_size
        									STEP #step
        )
    )
	x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, item_to_predict_index],
												TRAINING_SIZE, None, past_history,
												future_target, STEP)

	val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
	val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()
	#print((x_val_multi).shape[-1])
	#print(x_val_multi.shape)
	image_number = (x_val_multi).shape[-1]
	#### Unnormalizing the data (so we can see actual prices in GP)
	def unnormalized(val):
		return (val*item_std) + item_mean
	
	countindex=0
	clear_pngs(img_dir)
	print(len(x_val_multi))
	taken= val_data_multi.take(3) #takes 3 from the tensor flow dataset
	for x, y in taken:
		#print(x.shape,y.shape) # this is the size of the X axis as well as 
		history = unnormalized(x[0].numpy())
		#history_old=unnormalized(x[0].numpy())
		future = unnormalized(y[0].numpy())
		#future_old = unnormalized(y[0].numpy())
		prediction = unnormalized(model.predict(x)[0])
		figure = multi_step_plot(history, future, prediction, item_to_predict_index,item_to_predict=item_to_predict,index=countindex)
		if (save_img): save_plot_to_png(figure, "{}_{}_{}_{}.png".format(name, item_to_predict, price_type_name, countindex), name)
		countindex+=1
	#gif_from_png_dir(item_to_predict,img_dir)

# =========== HYPERPARAMETER TUNING FUNCTIONS =========== 
def multivariate_rnn_multi_hyperparameter_tuning(df, item_to_predict, price_type_name="", batch_size=[32], buffer_size = [30], \
	epochs = [20], eval_interval = [100], num_dropout_layers = [2],	num_lstm_units = [64], \
		learning = [0.001], past_history = [30]):

	# Write results to file
	current_time = datetime.datetime.utcnow()
	HP_FILE = os.path.join(parent_dir,'data/hp-tuning/{}_{}_MultiM.txt'.format(current_time.strftime("%m-%d-%Y"),price_type_name))

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

def multivariate_rnn_single_hyperparameter_tuning(df, item_to_predict, price_type_name="", batch_size=[32], buffer_size = [30], \
	epochs = [20], eval_interval = [100], num_dropout_layers = [2],	num_lstm_units = [32], \
		learning = [0.001], past_history = [30]):

	# Write results to file
	current_time = datetime.datetime.utcnow()
	HP_FILE = os.path.join(parent_dir,'data/hp-tuning/{}_{}_MultiS.txt'.format(current_time.strftime("%m-%d-%Y"),price_type_name))

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

def univariate_rnn_hyperparameter_tuning(df, item_to_predict, price_type_name="", batch_size=[32], buffer_size = [30], \
	epochs = [20], eval_interval = [100],	num_lstm_units = [8], past_history = [30]):

	# Write results to file
	current_time = datetime.datetime.utcnow()
	HP_FILE = os.path.join(parent_dir,'data/hp-tuning/{}_{}_Uni.txt'.format(current_time.strftime("%m-%d-%Y"),price_type_name))

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

def full_hyperparameter_tuning(selected_df,items_to_predict,price_type_names, min_features, max_features):
	# items_to_predict = ['Old_school_bond', 'Rune_platebody', 'Rune_2h_sword', 'Rune_axe',\
	# 	'Rune_pickaxe', 'Adamant_platebody', 'Amulet_of_power']
	#items_to_predict #= item_selection()
	#items_to_predict# = select_sorted_items(items_to_predict)
	#price_type_names = ["HighPrice","LowPrice","LowVolumePrice","HighVolumePrice"]
	min_features = 2
	max_features = 5

	for item_to_predict in items_to_predict:
		for price_type_name in price_type_names:
			for num_features in range(min_features,max_features):
				# SELECT ITEMS
				items_selected = item_selection()

				#TODO Stopped here, need to integrate new processing of multi df types
				# FEATURE EXTRACTION
				processed_low_price, processed_high_price, processed_low_volume, processed_high_volume = prepare_data_from_df(item_to_predict, items_selected)

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
	items_to_predict = ["Arcane spirit shield","Inquisitor's mace","Old school bond"]#['Mithril bar','Air battlestaff','Red chinchompa','Manta ray','Saradomin brew(4)','Anglerfish','Purple sweets','Anti-venom+(4)','Cactus spine']
	num_features = 5 # columns for each item minus datetime and name so 7 - 2 = 5
 
 	# SELECT ITEMS
	#items_selected = items_to_predict#item_selection()
 
	for item_to_predict in items_to_predict:

		# =========== PREPROCESSING =========== 
		# FEATURE EXTRACTION
		#preprocessed_df = prepare_data_from_folder(item_to_predict, items_selected)
		#print(preprocessed_df)
		#############################################################
 		#getting live data instead of from csv

		apimapping = PricesAPI("GEPrediction-OSRS","GEPRediction-OSRS")
		mapping_df = apimapping.mapping_df()

		apitimeseries = PricesAPI("GEPrediction-OSRS","GEPRediction-OSRS")
		timeseries_df = apitimeseries.timeseries_df("5m", getIDFromName(mapping_df,item_to_predict.replace("_"," ")))
		timeseries_df['name'] = item_to_predict.replace("_"," ")

		processed_low_price, processed_high_price, processed_low_volume, processed_high_volume = prepare_data_from_df(item_to_predict, data_frame=timeseries_df)
		#print(processed_low_price, processed_high_price, processed_low_volume, processed_high_volume)
  		#############################################################
		
		#low_price = buy_average = avgLowPrice
		#high_price = sell_average = avgHighPrice
		#low_volume= buy_quantity = lowPriceVolume
		#high_volume = sell_quantity = highPriceVolume
		#low_price_df, high_price_df, low_volume_df, high_volume_df
		price_type_names = ["LowPrice","HighPrice","LowVolumePrice","HighVolumePrice"]
		mapping_dfs = [processed_low_price, processed_high_price, processed_low_volume, processed_high_volume]
		
		for (price_type_name,mapping_df) in zip(price_type_names[:2], mapping_dfs[:2]): # get first two ,eg lowprice highprice and dataframes low and high price
			print(price_type_name, mapping_df.shape)
			#print(preprocessed_df)

			#print(items_selected)
			print("Learning [{}]".format(item_to_predict))

			# FEATURE SELECTION & NORMALIZATION
			#selected_df, pred_std, pred_mean = regression_f_test(preprocessed_df, item_to_predict, number_of_features=num_features)
			#############################################################
			#features are any and all columns, eg high, low, high quantity , low quantity ... etc
			num_features = len(mapping_df.columns)-2
			selected_df, pred_std, pred_mean = regression_f_test(mapping_df, item_to_predict, number_of_features=num_features)
			history_var=5#(math.floor(len(timeseries_df)/6)-1)
			training_size=(math.floor(len(timeseries_df)/2))
			#regex everything between qoutes
			#############################################################
			# print(selected_df.head())
			# print(selected_df.shape)
			# print("columns with nan: {}".format(selected_df.columns[selected_df.isna().any()].tolist()))


			# =========== UNIVARIATE =========== 
			uni_config = {}
			# TRAINING AND SAVING MODEL
			print("On [{}]".format('UNIVARIATE'))
			univariate_config = {'lstm_units':8, 'EVALUATION_INTERVAL':500, 'EPOCHS':50, 'past_history':history_var, 'TRAINING_SIZE':training_size, 'price_type_name':price_type_name}
			univariate_rnn_result = univariate_rnn(selected_df, item_to_predict, **univariate_config)

			# # LOADING AND APPLYING MODEL
			#os.path.join(parent_dir,'models/{}_multiM_model.h5'.format(item_to_predict))
			print(os.path.join(models_dir,'{}_{}_uni_model.h5'.format(item_to_predict,price_type_name)))
			loaded_model = tf.keras.models.load_model(os.path.join(models_dir,'{}_{}_uni_model.h5'.format(item_to_predict,price_type_name)))
			univariate_rnn_test = apply_univariate_test(selected_df, item_to_predict, loaded_model, pred_std, pred_mean, past_history=history_var, TRAINING_SIZE=training_size, price_type_name=price_type_name)

			# =========== MULTIVARIATE SINGLE STEP ===========
			multiS_config = {'lstm_units':64, 'EVALUATION_INTERVAL':500, 'EPOCHS':50, 'learning_rate':0.0001, 'num_dropout': 2, 'past_history':history_var, 'TRAINING_SIZE':training_size,'price_type_name':price_type_name}
			# TRAINING AND SAVING MODEL
			print("On [{}]".format('MULTIVARIATE SINGLE STEP'))
			multivariate_rnn_result = multivariate_rnn_single(selected_df, item_to_predict, **multiS_config)

			# # LOADING AND APPLYING MODEL
			multis_model = os.path.join(models_dir,'{}_{}_multiS_model.h5'.format(item_to_predict,price_type_name))
			print(os.path.exists(multis_model))
			loaded_model = tf.keras.models.load_model(multis_model)
			multivariate_rnn_test = apply_multivariate_single_step_test(selected_df, item_to_predict, loaded_model, pred_std, pred_mean, past_history=history_var, TRAINING_SIZE=training_size, price_type_name=price_type_name)


			# =========== MULTIVARIATE MULTI STEP ===========
			multiM_config = {'lstm_units':128, 'EVALUATION_INTERVAL':500, 'EPOCHS':50, 'learning_rate':0.0001, 'num_dropout': 2, 'past_history':history_var, 'TRAINING_SIZE':training_size, 'save_img': False, 'price_type_name':price_type_name}
			# TRAINING AND SAVING MODEL
			print("On [{}]".format('MULTIVARIATE MULTI STEP'))
			multivariate_rnn_multi_result = multivariate_rnn_multi(selected_df, item_to_predict, **multiM_config)

			# # LOADING AND APPLYING MODEL
			loaded_model = tf.keras.models.load_model(os.path.join(models_dir,'{}_{}_multiM_model.h5'.format(item_to_predict,price_type_name)))
			multivariate_rnn_multi_test = apply_multivariate_multi_step_test(selected_df, item_to_predict, loaded_model, pred_std, pred_mean, past_history=history_var, TRAINING_SIZE=training_size, save_img=False, price_type_name=price_type_name)


			# # # =========== HYPERPARAMETER TUNING ===========
			#full_hyperparameter_tuning()
		
if __name__ == "__main__":
	main()