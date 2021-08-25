import requests
import json
import csv
import time
import os.path
import os
import pandas as pd
import socket
from classes.wrapper import PricesAPI
import shutil

#https://github.com/JonasHogman/osrs-prices-api-wrapper

# One dict each for 'low_price', 'low_volume', 'high_price', 'high_volume', 'overall_average', 'overall_quantity'
high = []
low = []
names = ['low_price','low_volume','high_price','high_volume','index']

datestamp=time.time()

#setup paths and variables used below
parent_dir = os.path.dirname(os.path.realpath(__file__))
#os.getcwd()
dataRepo =os.path.join(os.path.dirname(parent_dir), "data/runelite")

# make sure data dir exists for runelite
mode = 0o666
if not os.path.exists(dataRepo):
 os.makedirs(dataRepo, mode)
 print("Data dir created: [{}]".format(dataRepo))

def initialize_data(name, df):
	filepath=os.path.join(dataRepo,'{}.csv'.format(name))
	#df.to_csv(filepath, index = 'timestamp', mode='w')
	if name=='index':
		df.to_csv(filepath, index = 'id', mode='w')
	else:
		df.to_csv(filepath, index = 'timestamp', mode='w')
	
def update_dataframe(name, df):
	filepath=os.path.join(dataRepo,'{}.csv'.format(name))
	if name=='index':
		df.to_csv(filepath, index = 'id', mode='a',header=False)
	else:
		df.to_csv(filepath, index = 'timestamp', mode='a',header=False)

def archive_data(names):
	#print("{} - appending data".format(current_timestamp))
	# check if old folder exists, create if it doesnt 
	backup='old/'
	backupdir = os.path.join(dataRepo, backup) 
	mode = 0o666
	if not os.path.exists(backupdir):
		os.makedirs(backupdir, mode)
		
	for name in names:
	
			#move files to old data folder for backup purposes
		startpath=os.path.join(dataRepo,'{}.csv'.format(name))
		if not os.path.exists(startpath):
			continue
		endpath=os.path.join(backupdir,'{}_{}.csv'.format(datestamp,name))
		#os.rename(startpath,endpath)
		shutil.copy(startpath,endpath)

def doHeadersMatch(name, df):
	doHeadersMatch = False
	filepath=os.path.join(dataRepo,'{}.csv'.format(name))
	print(filepath)
	if not os.path.exists(filepath):
		return False
	HEADERS = list(pd.read_csv(filepath).name)
	print(len(HEADERS), (len(df)))
	if (len(HEADERS)) == (len(df)):
		doHeadersMatch=True
	return doHeadersMatch

def main():

	apimapping = PricesAPI("GEPrediction-OSRS","GEPrediction_Scraper_Rpotato")

	#checks 
	headersMatch = True
	filesfound = []
	filesmissing = []

	#find missing files
	for name in names:
		if not os.path.isfile(os.path.join(dataRepo,'{}.csv'.format(name))):
			#print(os.path.join(dataRepo,'{}.csv'.format(name)))
			filesmissing.append(name)
		else:
			filesfound.append(name)

	#get indexes/mappings
	
	mapping_api = PricesAPI("GEPrediction-OSRS","GEPrediction_Scraper_Rpotato")
	mapping_df = mapping_api.mapping_df()

	#load data from sources 
	prices_api= PricesAPI("GEPrediction-OSRS","GEPrediction_Scraper_Rpotato")
	prices_df = prices_api.prices_df("5m", mapping=True)
	
 # timeseries_api = PricesAPI("GEPrediction-OSRS","GEPrediction_Scraper_Rpotato")
	#timeseries_df= timeseries_api.timeseries_df("5m", '2')

	#buycsv=low sellcsv=high
	headersMatch = doHeadersMatch('index',mapping_df)
	
	#check if new items, reinitialize 
	if not headersMatch:
		archive_data(names)
	
	#check headers for each file that exists, archive and re-initiate if needed
	for found in filesfound:
		df=pd.DataFrame()
		if found == 'low_price':
			values='avgLowPrice'
			df=prices_df.pivot(index='timestamp',columns='name', values='avgLowPrice')
			#stopped here, i think this is importing 2 df's ... debug here 
			
		if found == 'low_volume':
			values ='lowPriceVolume'
			df=prices_df.pivot(index='timestamp', columns='name', values='lowPriceVolume')
			
		if found == 'high_price':
			values ='avgHighPrice'
			df = prices_df.pivot(index='timestamp', columns='name', values='avgHighPrice')
			
		if found == 'high_volume':
			values ='highPriceVolume'
			df =prices_df.pivot(index='timestamp', columns='name', values='highPriceVolume')
				
		if not (found=='index') and not headersMatch: #update trigger
			_filepath=filepath=os.path.join(dataRepo,'{}.csv'.format(found))
			csv_import_df= pd.read_csv(_filepath, index_col='timestamp')
			csv_import_df=csv_import_df.reindex(fill_value=0,columns=(mapping_df.name))
			df = df.reindex(columns=mapping_df.name,fill_value=0)
			csv_import_df=csv_import_df.append(df)
			initialize_data(found,csv_import_df)
			#update_dataframe(found,df)
			
		if (found == 'index') and not headersMatch:
			initialize_data(found,mapping_df)
		
		if not (found == 'index') and headersMatch: #dont update index if headers match
			df = df.reindex(columns=mapping_df.name,fill_value=0)
			update_dataframe(found,df)
	
	for missing in filesmissing:
		df=pd.DataFrame()
		
		if missing == 'low_price':
			values='avgLowPrice'
			df=prices_df.pivot(index='timestamp', columns='name', values='avgLowPrice')
			
		if missing == 'low_volume':
			values ='lowPriceVolume'
			df=prices_df.pivot(index='timestamp', columns='name', values='lowPriceVolume')
			
		if missing == 'high_price':
			values ='avgHighPrice'
			df = prices_df.pivot(index='timestamp', columns='name', values='avgHighPrice')
			
		if missing == 'high_volume':
			values ='highPriceVolume'
			df =prices_df.pivot(index='timestamp', columns='name', values='highPriceVolume')
			
		if (missing == 'index'):
			initialize_data(missing,mapping_df)
		else:
			df = df.reindex(columns=mapping_df.name,fill_value=0)
			initialize_data(missing,df)

if __name__ == "__main__":
	main()
	