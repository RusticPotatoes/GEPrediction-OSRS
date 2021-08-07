import requests
import json
import csv
import time
import os.path
import os
import pandas as pd
import socket

#setup paths and variables used below
parent_dir =  os.path.dirname(os.path.realpath(__file__))
#os.getcwd()
datadir =os.path.join(os.path.dirname(parent_dir), "data/runelite")
mode = 0o666
if not os.path.exists(datadir):
  os.makedirs(datadir, mode)
  print("Data dir created: [{}]".format(datadir))

#create ids csv, load json data
wikiurl_mapping = "https://prices.runescape.wiki/api/v1/osrs/mapping"

#make sure you put your own header per tos
hostname= socket.getfqdn()#or use something like 'rusticpotatoes'
wikiheader = {'user-agent':hostname}
wikidata_mapping = requests.get(wikiurl_mapping, headers=wikiheader).json()
df_ids = pd.DataFrame(wikidata_mapping)


#create csv from data

csvfile = os.path.join(datadir, "wikidata_idmappings.csv")
df_ids.to_csv(csvfile, index = None)
#return id from name
def getIDFromName(df, name):
  return df.query(("name == '{}'".format(name))).id.item()

#example
#getIDFromName(df_ids,"Cannonball") # returns '2'
#getNameFromID(df_ids,'2')# returns "Cannonball"


#setup creation of formatted url
#create query for item, valid options are 5m 1hr 6hr
timetoget="1h"
itemid=getIDFromName(df_ids,"Cannonball")

wikiurl_timeseries= "https://prices.runescape.wiki/api/v1/osrs/timeseries?"
timestep = "timestep={}".format(timetoget)
id= "&id={}".format(itemid)
fetchItemTimeSeriesURL="{}{}{}".format(wikiurl_timeseries,timestep,id)

#imports as dict
wikidata_timeseries = requests.get(fetchItemTimeSeriesURL, headers=wikiheader).json()

df_item = pd.DataFrame.from_dict(wikidata_timeseries['data'])

newheaders = df_item.keys().tolist()
newheaders.append("id")

df_item = pd.DataFrame.from_dict(wikidata_timeseries['data'])
df_item = df_item.reindex(columns = newheaders, fill_value=(wikidata_timeseries['itemId']))

#save item info to csv
csvfile = os.path.join(datadir, "wikidata_timeseries.csv")

#TODO: create methods to append csv to file, and create new file on new items like in osbuddy and rsbuddy scrapers 
df_item.to_csv(csvfile, index = None)
