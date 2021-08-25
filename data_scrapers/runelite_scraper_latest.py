import requests
import json
import csv
import time
import os.path
import os
import pandas as pd
import socket
from classes.wrapper import PricesAPI

#https://github.com/JonasHogman/osrs-prices-api-wrapper

# One dict each for 'buy_average', 'buy_quantity', 'sell_average', 'sell_quantity', 'overall_average', 'overall_quantity'
high = []
low = []
names = ['latest','index']

datestamp=time.time()

#setup paths and variables used below
parent_dir =	os.path.dirname(os.path.realpath(__file__))
#os.getcwd()
dataRepo =os.path.join(os.path.dirname(parent_dir), "data/runelite")

# make sure data dir exists for runelite
mode = 0o666
if not os.path.exists(dataRepo):
	os.makedirs(dataRepo, mode)
	print("Data dir created: [{}]".format(dataRepo))

def initialize_data(name, df):
		filepath=os.path.join(dataRepo,'{}.csv'.format(name))
		df.to_csv(filepath, index = None, mode='w')
		
def update_dataframe(name, df):
		filepath=os.path.join(dataRepo,'{}.csv'.format(name))
		df_from_csv = pd.read_csv(filepath)
		concat_df = pd.concat([df,df_from_csv]).drop_duplicates().reset_index(drop=True)
		concat_df.to_csv(filepath, index = None, mode='w')

def archive_data(name, df):
		#print("{} - appending data".format(current_timestamp))
		# check if old folder exists, create if it doesnt 
		backup='old/'
		backupdir = os.path.join(dataRepo, backup)	
		mode = 0o666
		if not os.path.exists(backupdir):
				os.makedirs(backupdir, mode)

		files = []

		#move files to old data folder for backup purposes
		startpath=os.path.join(dataRepo,'{}.csv'.format(name))
		endpath=os.path.join(backupdir,'{}_{}.csv'.format(datestamp,name))
		os.rename(startpath,endpath)
		
		#df.to_csv(latest_csv, index = None, mode='w')

def doHeadersMatch(name, df):
		doHeadersMatch = False
		filepath=os.path.join(dataRepo,'{}.csv'.format(name))
		if not os.path.exists(filepath):
				return False
		HEADERS = list(pd.read_csv(filepath, error_bad_lines=False).name)
		#print(len(HEADERS), (len(df)))
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
		apimapping = PricesAPI("GEPrediction-OSRS","GEPrediction_Scraper_Rpotato")
		mapping_df = apimapping.mapping_df()
		#print(len(mapping_df))
		#load data from sources 
		apilatest= PricesAPI("GEPrediction-OSRS","GEPrediction_Scraper_Rpotato")
		latest_df = apilatest.latest_df(mapping=True)
		latest_df['highTime']= pd.to_datetime(latest_df['highTime'],unit='s',origin='unix')
		latest_df['lowTime']= pd.to_datetime(latest_df['lowTime'],unit='s',origin='unix')

		#check headers for each file that exists, archive and re-initiate if needed
		for found in filesfound:
				headersmatch = doHeadersMatch('index',mapping_df)
				if not headersMatch:
						archive_data(found,latest_df)
						initialize_data(found,latest_df)
				elif not (found == 'index'): #dont update index if headers match
						update_dataframe(found,latest_df)
		
		for missing in filesmissing:
				if (missing == 'index'):
						initialize_data(missing,mapping_df)
				else:
				initialize_data(missing,latest_df)

if __name__ == "__main__":
		main()