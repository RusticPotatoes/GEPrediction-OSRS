from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import os
import pandas as pd
import requests
import json
import socket

from sklearn import datasets
from sklearn.feature_selection import RFE, f_regression, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from data_scrapers.classes.wrapper import PricesAPI

parent_dir = os.path.dirname(os.path.realpath(__file__))

data_dir = os.path.join(parent_dir,"data")
runelite_dir = os.path.join(data_dir,"runelite")

# DATA_FOLDER = "/opt/app/data/workspace/GEPrediction-OSRS/data/osbuddy/excess/"
# DATA_FOLDER = "/opt/app/data/workspace/GEPrediction-OSRS/data/rsbuddy/"
rsAPI = "https://storage.googleapis.com/osb-exchange/summary.json"

def save_member_items():
	r = requests.get(rsAPI)
	json_data = json.loads(r.text)
	non_member_list = []
	member_list = []
	for item in json_data:
		if (json_data[item]["members"] == False): 
			non_member_list.append(json_data[item]["name"].replace(" ", "_"))
		else: 
			member_list.append(json_data[item]["name"].replace(" ", "_"))
	
	# open output file for writing
	with open(os.path.join(data_dir,'non_member_list.txt'), 'w') as filehandle:
		json.dump(non_member_list, filehandle)
	
	# open output file for writing
	with open(os.path.join(data_dir,'member_list.txt', 'w')) as filehandle:
		json.dump(member_list, filehandle)

	# print("member items: {}, non-member items: {}".format(len(member_list), len(non_member_list)))
	
def item_selection(DATA_FOLDER = runelite_dir, drop_percentage=0.99):
	
	#prices_df.pivot()
	low_volume = pd.read_csv(DATA_FOLDER + "low_volume.csv", error_bad_lines=False, warn_bad_lines=False)
	low_volume = low_volume.set_index('timestamp')
	low_volume = low_volume.drop_duplicates()
	df = low_volume#.loc[:, (low_volume==0).mean() < drop_percentage]  # Drop columns with more than 5% 0s

	# open output file for reading
	with open(os.path.join(data_dir,'member_list.txt'), 'r') as filehandle:
		member_list = json.load(filehandle)
	
	#for item_name in member_list:
	#	if (item_name in df.columns.values):
	#		df = df.drop(item_name, axis=1)  # Drop all member only items

	# print(df.shape)
	return df.columns.values

def moving_average_convergence(group, nslow=26, nfast=12):
	emaslow = group.ewm(span=nslow, min_periods=1).mean()
	emafast = group.ewm(span=nfast, min_periods=1).mean()
	result = pd.DataFrame({'MACD': emafast-emaslow, 'emaSlw': emaslow, 'emaFst': emafast})
	result = pd.DataFrame({'MACD': emafast-emaslow})
	return result

def moving_average(group, n=9):
	sma = group.rolling(n).mean()
	sma=sma.rename('SMA')
	return sma

def RSI(group, n=14):
	delta = group.diff()
	dUp, dDown = delta.copy(), delta.copy()
	dUp[dUp < 0] = 0
	dDown[dDown > 0] = 0

	RolUp = dUp.rolling(n).mean()
	RolDown = dDown.rolling(n).mean().abs()
	
	RS = RolUp / RolDown
	rsi= 100.0 - (100.0 / (1.0 + RS))
	rsi=rsi.rename('RSI')
	return rsi

def prepare_data_from_folder(item_to_predict, items_selected, verbose=False, DATA_FOLDER = runelite_dir, reused_df=None, specific_features=None):
	
	# Computational optimization for application (just need to change MACD, RSI or slope)
	if specific_features is not None and reused_df is not None: 
		df = reused_df.copy()
		if ('MACD' in specific_features or 'RSI' in specific_features or 'slope' in specific_features):
			if (verbose): print('REPLACING MACD OR RSI!')
			df = df.drop(['MACD', 'RSI'], axis=1, errors='ignore')

			## Known finance features (MACD, RSI)
			macd = moving_average_convergence(df[item_to_predict])
			rsi = RSI(df[item_to_predict], 10)
			finance_features = pd.concat([macd, rsi], axis=1)

			df = pd.concat([df,finance_features], axis=1)
		
		if ('slope' in specific_features):
			df = df.drop(['slope'], axis=1, errors='ignore')
			if (verbose): print('REPLACING SLOPE!')
			## Differentiated signal
			tmp = df.copy()
			tmp.index = pd.to_datetime(tmp.index)
			slope = pd.Series(np.gradient(tmp[item_to_predict]), df.index, name='slope')
			tmp = pd.concat([tmp, slope], axis=1)

			df = pd.concat([df, slope], axis=1)

		if verbose: print("dropping: {}".format(df.columns[df.isna().any()].tolist()))
		df = df.dropna(axis='columns')
	
		return df

	low_price = pd.read_csv(DATA_FOLDER + "low_price.csv", error_bad_lines=False, warn_bad_lines=False)
	low_price = low_price.set_index('timestamp')
	low_price = low_price.drop_duplicates()
	df = low_price[items_selected].replace(to_replace=0, method='ffill')

	## Known finance features (MACD, RSI)
	macd = moving_average_convergence(df[item_to_predict])
	rsi = RSI(df[item_to_predict], 10)
	finance_features = pd.concat([macd, rsi], axis=1)

	## Fetched API features (buy quantity, sell price average)
	high_price = pd.read_csv(DATA_FOLDER + "high_price.csv", error_bad_lines=False, warn_bad_lines=False)
	high_price = high_price.set_index('timestamp')
	high_price = high_price.drop_duplicates()
	high_price = high_price[items_selected].replace(to_replace=0, method='ffill')
	high_price.columns = [str(col) + '_sa' for col in high_price.columns]

	low_volume = pd.read_csv(DATA_FOLDER + "low_volume.csv", error_bad_lines=False, warn_bad_lines=False)
	low_volume = low_volume.set_index('timestamp')
	low_volume = low_volume.drop_duplicates()
	low_volume = low_volume[items_selected].replace(to_replace=0, method='ffill')
	low_volume.columns = [str(col) + '_bq' for col in low_volume.columns]

	high_volume = pd.read_csv(DATA_FOLDER + "high_volume.csv", error_bad_lines=False, warn_bad_lines=False)
	high_volume = high_volume.set_index('timestamp')
	high_volume = high_volume.drop_duplicates()
	high_volume = high_volume[items_selected].replace(to_replace=0, method='ffill')
	high_volume.columns = [str(col) + '_sq' for col in high_volume.columns]

	## Datetime properties
	df['datetime'] = df.index
	df['datetime'] = pd.to_datetime(df['datetime'],unit='s',origin='unix')
	df['dayofweek'] = df['datetime'].dt.dayofweek
	df['hour'] = df['datetime'].dt.hour

	## Differentiated signal
	tmp = df.copy()
	tmp.index = pd.to_datetime(tmp.index)
	slope = pd.Series(np.gradient(tmp[item_to_predict]), df.index, name='slope')
	tmp = pd.concat([tmp, slope], axis=1)


	## Appending features to main dataframe
	df = pd.concat([df,finance_features, high_price, low_volume, high_volume, slope], axis=1)
	if verbose: print("dropping: {}".format(df.columns[df.isna().any()].tolist()))
	df = df.dropna(axis='columns')

	del low_price, high_price, low_volume, high_volume

	return df

def datetime_df_append(df):
	df['datetime'] = df.index
	df['datetime'] = pd.to_datetime(df['datetime'],unit='s')
	df['dayofweek'] = df['datetime'].dt.dayofweek
	df['hour'] = df['datetime'].dt.hour
	return df

def slope_df(df,item_to_predict):
	tmp = df.copy()
	tmp.index = pd.to_datetime(tmp.index)
	#print(tmp[item_to_predict])
	#print(np.gradient(tmp[item_to_predict]))	
	slope = pd.Series(np.gradient(tmp[item_to_predict]), df.index, name='slope')
	#tmp = pd.concat([tmp, slope], axis=1)
	return slope

def prepare_data_from_df(item_to_predict, verbose=False, data_frame = pd.DataFrame(), reused_df=None, specific_features=None):
	
	# Computational optimization for application (just need to change MACD, RSI or slope)
	if specific_features is not None and reused_df is not None: 
		df = reused_df.copy()
		if ('MACD' in specific_features or 'RSI' in specific_features or 'slope' in specific_features):
			if (verbose): print('REPLACING MACD OR RSI!')
			df = df.drop(['MACD', 'RSI'], axis=1, errors='ignore')

			## Known finance features (MACD, RSI)
			macd = moving_average_convergence(df[item_to_predict])
			rsi = RSI(df[item_to_predict], 10)
			finance_features = pd.concat([macd, rsi], axis=1)

			df = pd.concat([df,finance_features], axis=1)
		
		if ('slope' in specific_features):
			df = df.drop(['slope'], axis=1, errors='ignore')
			if (verbose): print('REPLACING SLOPE!')
			## Differentiated signal
			tmp = df.copy()
			tmp.index = pd.to_datetime(tmp.index)
			slope = pd.Series(np.gradient(tmp[item_to_predict]), df.index, name='slope')
			tmp = pd.concat([tmp, slope], axis=1)

			df = pd.concat([df, slope], axis=1)

		if verbose: print("dropping: {}".format(df.columns[df.isna().any()].tolist()))
		df = df.dropna(axis='columns')
	
		return df
	#low_price = buy_average = avgLowPrice
	#high_price = sell_average = avgHighPrice
	#low_volume= buy_quantity = lowPriceVolume
	#high_volume = sell_quantity = highPriceVolume	
	low_price = data_frame.pivot(index='timestamp',columns='name', values='avgLowPrice')
	#low_price = low_price.drop_duplicates()
	#df = low_price#[item_selected].replace(to_replace=0, method='ffill')
 	#low_price.columns = [str(col) + '_ba' for col in low_price.columns]

	## Fetched API features (buy quantity, sell price average)
	high_price = data_frame.pivot(index='timestamp', columns='name', values='avgHighPrice')
	#high_price = high_price.drop_duplicates()
	#high_price = high_price[items_selected].replace(to_replace=0, method='ffill')
	#high_price.columns = [str(col) + '_sa' for col in high_price.columns]

	low_volume = data_frame.pivot(index='timestamp', columns='name', values='lowPriceVolume')
	#low_volume = low_volume.drop_duplicates()
	#low_volume = low_volume[items_selected].replace(to_replace=0, method='ffill')
	#low_volume.columns = [str(col) + '_bq' for col in low_volume.columns]

	high_volume = data_frame.pivot(index='timestamp', columns='name', values='highPriceVolume')
	#high_volume = high_volume.drop_duplicates()
	#high_volume = high_volume[items_selected].replace(to_replace=0, method='ffill')
	#high_volume.columns = [str(col) + '_sq' for col in high_volume.columns]
 
	## Known finance features (MACD, RSI)
	#macd = moving_average_convergence(df)
	low_price_macd = moving_average_convergence(low_price[item_to_predict])
	high_price_macd = moving_average_convergence(high_price[item_to_predict])
	low_volume_macd = moving_average_convergence(low_volume[item_to_predict])
	high_volume_macd = moving_average_convergence(high_volume[item_to_predict])
	
	#rsi = RSI(df, 10)
	low_price_rsi = RSI(low_price[item_to_predict], 10)
	high_price_rsi = RSI(high_price[item_to_predict], 10)
	low_volume_rsi = RSI(low_volume[item_to_predict], 10)
	high_volume_rsi = RSI(high_volume[item_to_predict], 10)
	
	#finance_features = pd.concat([macd, rsi], axis=1)
	low_price_finance_features = pd.concat([low_price_macd, low_price_rsi.fillna(0)], axis=1)
	high_price_finance_features = pd.concat([high_price_macd, high_price_rsi.fillna(0)], axis=1)
	low_volume_finance_features = pd.concat([low_volume_macd, low_volume_rsi.fillna(0)], axis=1)
	high_volume_finance_features = pd.concat([high_volume_macd, high_volume_rsi.fillna(0)], axis=1)

	## Append datetime properties
	low_price = datetime_df_append(low_price)
	high_price = datetime_df_append(high_price)
	low_volume = datetime_df_append(low_volume)
	high_volume = datetime_df_append(high_volume)

	## Differentiated signal (slope append)
	low_price_slope = slope_df(low_price, item_to_predict)#, '_ba')
	high_price_slope = slope_df(high_price, item_to_predict)#, '_sa')
	low_volume_slope = slope_df(low_volume, item_to_predict)#, '_bq')
	high_volume_slope = slope_df(high_volume, item_to_predict)#, '_sq')

	## Appending features to main dataframe
	low_price_df = pd.concat([low_price, low_price_finance_features, low_price_slope], axis=1)
	high_price_df = pd.concat([high_price, high_price_finance_features, high_price_slope], axis=1)
	low_volume_df = pd.concat([low_volume,low_volume_finance_features, low_volume_slope], axis=1)
	high_volume_df = pd.concat([high_volume,high_volume_finance_features, high_volume_slope], axis=1)
	#if verbose: print("dropping: {}".format(df.columns[df.isna().any()].tolist()))
	#df = df.dropna(axis='columns')

	del low_price, high_price, low_volume, high_volume

	return low_price_df, high_price_df, low_volume_df, high_volume_df
# FEATURE SELECTION FUNCTIONS

def regression_f_test(input_df, item_to_predict, number_of_features=7, print_scores=False, specific_features=None):
	#print(input_df.axes[1])
	features = input_df.drop(['datetime'], axis=1).copy()

	if specific_features is not None: 
		features = features[specific_features]
		# print("SPECIFIC FEATURES USED")
		# print(features.head())

	# normalize dataset
	features_std = features.std()
	features_mean = features.mean()
	dataset=(features-features_mean)/features_std
 
	#print('std', features_std)
	#print('mean', features_mean)
	#print('dataset', dataset)
	
	X = dataset.drop([item_to_predict], axis=1)
	y = dataset[item_to_predict]
	#print('X', X)
	#print('Y', y)
	X = X.dropna(axis='columns')

	#print('New x', X)

	# define feature selection
	fs = SelectKBest(score_func=f_regression, k=number_of_features)
	# apply feature selection
	fs.fit_transform(X, y)

	# Get scores for each of the columns
	scores = fs.scores_
	if print_scores:
		for idx, col in enumerate(X.columns): 
			print("feature: {: >20} \t score: {: >10}".format(col, round(scores[idx],5)))

	# Get columns to keep and create new dataframe with those only
	cols = fs.get_support(indices=True)
	features_df_new = X.iloc[:,cols]
	
	# print('std: {}, mean: {}'.format(features_std[item_to_predict], features_mean[item_to_predict]))
	return pd.concat([features_df_new, y], axis=1), features_std[item_to_predict], features_mean[item_to_predict]

def recursive_feature_elim(input_df, item_to_predict, number_of_features=7):
	features = input_df.drop(['datetime'], axis=1).copy()

	# normalize dataset
	features_std = features.std()
	features_mean = features.mean()
	dataset=(features-features_mean)/features_std

	X = dataset.drop([item_to_predict], axis=1)
	y = dataset[item_to_predict]
	
	X = X.dropna(axis='columns')

	# perform feature selection
	rfe = RFE(RandomForestRegressor(n_estimators=500, random_state=1), number_of_features)
	fit = rfe.fit(X, y)
	# report selected features
	print('Selected Features:')
	names = dataset.drop([item_to_predict], axis=1).columns.values
	selected_features = []
	for i in range(len(fit.support_)):
		if fit.support_[i]:
			selected_features.append(names[i])

	return pd.concat([X[selected_features], y], axis=1), features_std[item_to_predict], features_mean[item_to_predict]

# Unnormalizing the data (so we can see actual prices in GP)
def unnormalized(val, std, mean):
	return (val*std) + mean

def select_sorted_items(items_selected, minimum_price=1000, verbose=False, DATA_FOLDER = runelite_dir):
	
	low_price = pd.read_csv(DATA_FOLDER + "low_price.csv", error_bad_lines=False, warn_bad_lines=False)
	low_price = low_price.set_index('timestamp')
	low_price = low_price.drop_duplicates()
	df = low_price[items_selected].replace(to_replace=0, method='ffill')

	if (verbose):
		pd.set_option('display.max_rows', None)
		print(df.mean().sort_values())

	mean_dict = df.mean().sort_values().to_dict()
	chosen_items = []

	for key in mean_dict:
		if (mean_dict[key] > minimum_price):
			chosen_items.append(key)
	
	if (verbose): print(chosen_items)
	return chosen_items

def main():
	# SAVE ITEM LISTS
	# save_member_items()

	# SELECT ITEMS
	items_selected = item_selection()
	narrowed_items = select_sorted_items(items_selected)
	#print(narrowed_items)
	item_to_predict = 'Arcane_spirit_shield'
	# items_selected = ['Rune_axe', 'Rune_2h_sword', 'Rune_scimitar', 'Rune_chainbody', 'Rune_full_helm', 'Rune_kiteshield']

	# # ADD FEATURES
	# preprocessed_df = prepare_data(item_to_predict, items_selected, verbose=True)
	# print(preprocessed_df.head())
	# print(preprocessed_df.shape)

	# # FEATURE SELECTION
	# # selected_data, pred_std, pred_mean = recursive_feature_elim(preprocessed_df, item_to_predict)
	# selected_data, pred_std, pred_mean = regression_f_test(preprocessed_df, item_to_predict)
	# print(selected_data.head())
	# print(selected_data.shape)
	# # print(unnormalized(selected_data[item_to_predict], pred_std, pred_mean))

if __name__ == "__main__":
	main()