from __future__ import absolute_import, division, print_function, unicode_literals
from preprocessing import prepare_data_from_df, regression_f_test, recursive_feature_elim, item_selection, select_sorted_items
from models import univariate_data, create_time_steps, show_plot, multivariate_data, multi_step_plot
from data_scrapers.classes.wrapper import  PricesAPI
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import json
import csv
import time

parent_dir = os.path.dirname(os.path.realpath(__file__))

models_dir = os.path.join(parent_dir,"models")
features_dir = os.path.join(models_dir,"features")

data_dir = os.path.join(parent_dir,"data")
predict_dir = os.path.join(data_dir,"predictions")

TRAIN_SPLIT = 0
tf.random.set_seed(13)
STEP = 1

def getIDFromName(df,name):
	return (df[df['name'] == name].item_id.item())

def getNameFromID(df,id):
	return (df[df['item_id'] == id].name.item())

labels = ['timestamp', 'uni', 'multiS', 'multiM1', 'multiM2', 'multiM3', 'multiM4', 'multiM5']
def writeToCSV(filename, data, timestamp):
	with open(os.path.join(predict_dir, '{}.csv'.format(filename)), mode='w', newline='') as GE_data:
		GE_writer = csv.writer(GE_data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		GE_writer.writerow(labels)  # write field names

		new_array = [timestamp]
		new_array.extend(data)
		GE_writer.writerow(new_array)

def appendToCSV(filename, data, timestamp):
	with open(os.path.join(predict_dir,'{}.csv'.format(filename)), mode='a', newline='') as GE_data:
		GE_writer = csv.writer(GE_data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

		new_array = [timestamp]
		new_array.extend(data)
		GE_writer.writerow(new_array)

def apply_univariate(df, item_to_predict, model, item_std, item_mean, past_history=5):

	df_newest_values = df.tail(past_history)[item_to_predict].values
	reshaped_values = np.reshape(df_newest_values, (past_history, 1))
	formatted_values = np.array([reshaped_values])

	#### Unnormalizing the data (so we can see actual prices in GP)
	def unnormalized(val):
		return (val*item_std) + item_mean

	result = unnormalized(model.predict(formatted_values))
	return result

def apply_multivariate_single_step(df, item_to_predict, model, item_std, item_mean, past_history=5):

	df_newest_values = df.tail(past_history).values
	formatted_values = np.array([df_newest_values])

	#### Unnormalizing the data (so we can see actual prices in GP)
	def unnormalized(val):
		return (val*item_std) + item_mean

	result = unnormalized(model.predict(formatted_values))
	return result

def apply_multivariate_multi_step(df, item_to_predict, model, item_std, item_mean, future_target=5, past_history=5):
	df_newest_values = df.tail(past_history).values
	formatted_values = np.array([df_newest_values])

	#### Unnormalizing the data (so we can see actual prices in GP)
	def unnormalized(val):
		return (val*item_std) + item_mean

	result = unnormalized(model.predict_on_batch(formatted_values))

	return result

def plot_data(df):
    data = []
    for col in df.columns:
        data.append(go.Scatter(x=df.index,y=df[col],name=col))
    fig = go.Figure(data=data)
    fig.show()

def evaluate_result(model, trainX,trainY, testX, testY, ITEM_TO_PREDICT, verbose=True, batch_size=100):
    # make predictions
    trainPredict = model.predict(trainX, batch_size)
    testPredict = model.predict(testX, )

    true_values = trainY + testY
    train_pred = trainPredict + [np.nan]*(len(true_values)-len(trainPredict))
    test_pred = [np.nan]*(len(true_values)-len(testPredict)) + testPredict
    
    result = pd.DataFrame(columns=['trainPredict','testPredict','True'])
    result['True'] = true_values
    result['trainPredict'] = train_pred
    result['testPredict'] = test_pred

    # folder = f'/content/drive/My Drive/Models/{ITEM_TO_PREDICT}'
    #if not os.path.exists(folder):
    #    os.makedirs(folder)
    #model.save(f'{folder}/{int(time.time())}.h5')
    #if verbose:
    #    plot_data(result)
    return result

def main():
	# Get the seconds since epoch
	current_timestamp = int(time.time())
	print("{} - predicting items".format(current_timestamp))

	# SELECT ITEMS
	model_types = ['uni', 'multiS', 'multiM']
	price_type_names = ["HighPrice","LowPrice","LowVolumePrice","HighVolumePrice"]
	items_to_predict = ["Arcane spirit shield","Inquisitor's mace","Old school bond"]#['Mithril bar','Air battlestaff','Red chinchompa','Manta ray','Saradomin brew(4)','Anglerfish','Purple sweets','Anti-venom+(4)','Cactus spine']
	items_selected = items_to_predict#[:2] #item_selection()
	preprocessed_df = None
	
	for item_to_predict in items_to_predict[:2]:#use[:2] for the first 2 
		# GET LIST OF FEATURES
		for price_type_name in price_type_names[:2]:
			for model_type in model_types:
				model_feature_file= '{}_{}_{}_features.txt'.format(item_to_predict, price_type_name, model_type)
				print(model_feature_file)
				feature_file = os.path.join(features_dir,model_feature_file)
				if not os.path.isfile(feature_file):
					print ("Model for {} hasn't been created, please run models.py first.".format(item_to_predict))
					return
				specific_feature_list = []
				with open(os.path.join(features_dir,'{}_{}_{}_features.txt'.format(item_to_predict, price_type_name, model_type)), 'r') as filehandle:
					specific_feature_list = json.load(filehandle)

		t0 = time.time()
		# FEATURE EXTRACTION

		#############################################################
 		#getting live data instead of from csv

		apimapping = PricesAPI("GEPrediction-OSRS","GEPRediction-OSRS")
		mapping_df = apimapping.mapping_df()

		apitimeseries = PricesAPI("GEPrediction-OSRS","GEPRediction-OSRS")
		timeseries_df = apitimeseries.timeseries_df("5m", getIDFromName(mapping_df,item_to_predict.replace("_"," ")))
		timeseries_df['name'] = item_to_predict.replace("_"," ")
  
		#processed_low_price, processed_high_price, processed_low_volume, processed_high_volume = prepare_data_from_df(item_to_predict, items_selected, data_frame=timeseries_df)
		#print(processed_low_price, processed_high_price, processed_low_volume, processed_high_volume)
  		#############################################################

		processed_low_price, processed_high_price, processed_low_volume, processed_high_volume  = prepare_data_from_df(item_to_predict, items_selected, data_frame=timeseries_df, \
			reused_df=preprocessed_df, specific_features=specific_feature_list)
		mapping_dfs = [processed_low_price, processed_high_price, processed_low_volume, processed_high_volume]

		for (price_type_name,mapping_df) in zip(price_type_names[:2], mapping_dfs[:2]):
			t1 = time.time()
			# FEATURE SELECTION & NORMALIZATION
			#input_df, item_to_predict, number_of_features=7, print_scores=False, specific_features=None
			selected_df, pred_std, pred_mean = regression_f_test(mapping_df, item_to_predict, \
				specific_features=specific_feature_list, number_of_features=len(specific_feature_list)-1)

			t2 = time.time()
			predictions = []
			for model_type in model_types:
				# LOADING AND APPLYING MODEL
				loaded_model = tf.keras.models.load_model(os.path.join(models_dir,'{}_{}_{}_model.h5'.format(item_to_predict,price_type_name, model_type)))
				if (model_type == 'uni'):
					result = apply_univariate(selected_df, item_to_predict, loaded_model, pred_std, pred_mean)[0]
				elif (model_type == 'multiS'):
					result = apply_multivariate_single_step(selected_df, item_to_predict, loaded_model, pred_std, pred_mean)[0]
				elif (model_type == 'multiM'):
					result = apply_multivariate_multi_step(selected_df, item_to_predict, loaded_model, pred_std, pred_mean)[0]
				else:
					print("Unrecognized model type.")
				
				predictions.extend(result)
			tf.keras.backend.clear_session()
			
			t3 = time.time()

			print('TIME LOG - preprocessing: {}, feature selection: {}, prediction: {}, total: {}'.format(t1-t0, t2-t1, t3-t2, t1+t2+t3))

			new_predictions = [int(i) for i in predictions]
			print('item: {}, type: {}, pred: {}'.format(item_to_predict,price_type_name,new_predictions))

			for price_type_name in price_type_names[:2]:
				if os.path.isfile(os.path.join(predict_dir,'{}_{}'.format(item_to_predict,price_type_name))):
					appendToCSV('{}_{}.csv'.format(item_to_predict,price_type_name), new_predictions, current_timestamp)
				else:
					writeToCSV('{}_{}.csv'.format(item_to_predict,price_type_name), new_predictions, current_timestamp)


if __name__ == "__main__":
	main()