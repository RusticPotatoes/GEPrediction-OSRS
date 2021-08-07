import requests
import json
import csv
import time
import os.path
import os
import pandas as pd

# current directory
parent_dir = os.path.dirname(os.path.realpath(__file__))

rsbuddyAPI = "https://rsbuddy.com/exchange/summary.json"

dataRepo = os.path.join(os.path.dirname(parent_dir), "data/rsbuddy")
names = ["buy_average","buy_quantity","sell_average","sell_quantity","overall_average","overall_quantity"]

# One dict each for 'buy_average', 'buy_quantity', 'sell_average', 'sell_quantity', 'overall_average', 'overall_quantity'
buy_average = []
buy_quantity = []
sell_average = []
sell_quantity = []
overall_average = []
overall_quantity = []

labels = ['timestamp']

def writeToCSV(filename, data, timestamp):
    with open('{}/{}.csv'.format(dataRepo, filename), mode='w', newline='') as GE_data:
        GE_writer = csv.writer(GE_data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        GE_writer.writerow(labels)  # write field names

        new_array = [timestamp]
        new_array.extend(data)
        GE_writer.writerow(new_array)

def appendToCSV(filename, data, timestamp):
    with open('{}/{}.csv'.format(dataRepo, filename), mode='a', newline='') as GE_data:
        GE_writer = csv.writer(GE_data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        new_array = [timestamp]
        new_array.extend(data)
        GE_writer.writerow(new_array)

def initialize_data(json_data, current_timestamp):
    print("{} - initializing data".format(current_timestamp))
    for item in json_data:
        labels.append(json_data[item]["name"].replace(" ", "_"))
        buy_average.append(json_data[item]["buy_average"])
        buy_quantity.append(json_data[item]["buy_quantity"])
        sell_average.append(json_data[item]["sell_average"])
        sell_quantity.append(json_data[item]["sell_quantity"])
        overall_average.append(json_data[item]["overall_average"])
        overall_quantity.append(json_data[item]["overall_quantity"])

    #for name in names:
    #   writeToCSV(name,buy_average, current_timestamp)

    writeToCSV("buy_average", buy_average, current_timestamp)
    writeToCSV("buy_quantity", buy_quantity, current_timestamp)
    writeToCSV("sell_average", sell_average, current_timestamp)
    writeToCSV("sell_quantity", sell_quantity, current_timestamp)
    writeToCSV("overall_average", overall_average, current_timestamp)
    writeToCSV("overall_quantity", overall_quantity, current_timestamp)

def append_data(json_data, current_timestamp):
    print("{} - appending data".format(current_timestamp))

    for item in json_data:
        buy_average.append(json_data[item]["buy_average"])
        buy_quantity.append(json_data[item]["buy_quantity"])
        sell_average.append(json_data[item]["sell_average"])
        sell_quantity.append(json_data[item]["sell_quantity"])
        overall_average.append(json_data[item]["overall_average"])
        overall_quantity.append(json_data[item]["overall_quantity"])

    appendToCSV("buy_average", buy_average, current_timestamp)
    appendToCSV("buy_quantity", buy_quantity, current_timestamp)
    appendToCSV("sell_average", sell_average, current_timestamp)
    appendToCSV("sell_quantity", sell_quantity, current_timestamp)
    appendToCSV("overall_average", overall_average, current_timestamp)
    appendToCSV("overall_quantity", overall_quantity, current_timestamp)

def merge_two_dicts(y, x):
    #"""Given two dictionaries, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

def addColumnsToDF(df, CSV_HEADERS, JSON_HEADERS):
    df.reindex(columns = JSON_HEADERS, fill_value=0)
    columncount=0
    return df

def merge_data(json_data, current_timestamp):
    print("{} - appending data".format(current_timestamp))
    # check if old folder exists, create if it doesnt 
    backup='old/'
    backupdir = os.path.join(dataRepo, backup)  
    mode = 0o666
    if not os.path.exists(backupdir):
        os.makedirs(backupdir, mode)

    files = []

    for name in names:
        files.append('{}.csv'.format(name))
    count = 0

    CSV_HEADERS = list(pd.read_csv('{}/buy_average.csv'.format(dataRepo), error_bad_lines=False).head(0))
    JSON_HEADERS = ['timestamp']

    for key,value in json_data.items():
        JSON_HEADERS.append(value['name'].replace(" ", "_"))       

    #dont need this for now 
    #buy_average_importedcsv = pd.read_csv('{}/{}.csv'.format(dataRepo, "buy_average")).reindex(columns = JSON_HEADERS, fill_value=0)
    #buy_quantity_importedcsv = pd.read_csv('{}/{}.csv'.format(dataRepo, "buy_quantity")).reindex(columns = JSON_HEADERS, fill_value=0)
    #sell_average_importedcsv = pd.read_csv('{}/{}.csv'.format(dataRepo, "sell_average")).reindex(columns = JSON_HEADERS, fill_value=0)
    #sell_quantity_importedcsv = pd.read_csv('{}/{}.csv'.format(dataRepo, "sell_quantity")).reindex(columns = JSON_HEADERS, fill_value=0)
    #overall_average_importedcsv = pd.read_csv('{}/{}.csv'.format(dataRepo, "overall_average")).reindex(columns = JSON_HEADERS, fill_value=0)
    #overall_quantity_importedcsv = pd.read_csv('{}/{}.csv'.format(dataRepo, "overall_quantity")).reindex(columns = JSON_HEADERS, fill_value=0)

    #move files to old data folder for backup purposes
    for file in files:
        os.rename('{}/{}'.format(dataRepo,file), '{}/old/{}_{}'.format(dataRepo,current_timestamp,file))

    buy_average_importedcsv.to_csv('{}/{}.csv'.format(dataRepo, "buy_average"), index=False)
    buy_quantity_importedcsv.to_csv('{}/{}.csv'.format(dataRepo, "buy_quantity"), index=False)
    sell_average_importedcsv.to_csv('{}/{}.csv'.format(dataRepo, "sell_average"), index=False)
    sell_quantity_importedcsv.to_csv('{}/{}.csv'.format(dataRepo, "sell_quantity"), index=False)
    overall_average_importedcsv.to_csv('{}/{}.csv'.format(dataRepo, "overall_average"), index=False)
    overall_quantity_importedcsv.to_csv('{}/{}.csv'.format(dataRepo, "overall_quantity"), index=False) 

def doHeadersMatch(json_data):
    doHeadersMatch = False
    HEADERS = list(pd.read_csv('{}/buy_average.csv'.format(dataRepo), error_bad_lines=False).head(0))
    if (len(HEADERS)) == (len(json_data)+1):
        doHeadersMatch=True
    return doHeadersMatch

def list_diff(list1, list2): 
	return (list(set(list1) - set(list2))) 

def main():
 # Get the seconds since epoch
    current_timestamp = int(time.time())

    headersMatch = True

    filesexist = os.path.isfile('{}/buy_average.csv'.format(dataRepo))

    # load data from sources
    r = requests.get(rsbuddyAPI)
    json_data = json.loads(r.text)

    #if files exist check headers 
    if filesexist:
        headersMatch = doHeadersMatch(json_data)

    #if headers are wrong merge first
    if not headersMatch:
        merge_data(json_data,current_timestamp)    

    #if files exist and they have hte same headers, append
    if filesexist:
        append_data(json_data, current_timestamp)

    #if files dont exist, initialize
    if not filesexist: 
        initialize_data(json_data, current_timestamp)
    

if __name__ == "__main__":
    main()