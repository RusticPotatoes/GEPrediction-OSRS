import requests
import json
import time
import csv
import os.path
import datetime
import os

parent_dir = os.path.dirname(os.path.realpath(__file__))

# rsbuddyAPI = "https://rsbuddy.com/exchange/summary.json"

# fullDict = {}
# labels = ['timestamp']
# allitems = []

# def initialize_fullDict():
#     # Get the seconds since epoch
#     current_timestamp = int(time.time())
#     fullDict[current_timestamp] = []

#     r = requests.get(rsbuddyAPI)
#     json_data = json.loads(r.text)
#     for item in json_data:
#         if (json_data[item]["overall_quantity"] > 5 and json_data[item]["members"] == False):
#             print(json_data[item])
#             allitems.append(item)

#     print(len(allitems))

# initialize_fullDict()

def writeToCSV(filename):
    with open(os.path.join(parent_dir,'data/{}.csv'.format(filename)), mode='w', newline='') as GE_data:
        GE_writer = csv.writer(GE_data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        GE_writer.writerow(['timestamp, random1, random2'])  # write field names

        current_timestamp = int(time.time())
        new_array = [current_timestamp, time.strftime('%Y-%m-%dT%H:%M:%S %Z',time.localtime(time.time())), 25]
        GE_writer.writerow(new_array)

def appendToCSV(filename):
    with open(os.path.join(parent_dir,'data/{}.csv'.format(filename)), mode='a', newline='') as GE_data:
        GE_writer = csv.writer(GE_data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # GE_writer.writerow(['timestamp, random1, random2'])  # write field names
        
        current_timestamp = int(time.time())
        new_array = [current_timestamp, time.strftime('%Y-%m-%dT%H:%M:%S %Z',time.localtime(time.time())), 25]
        GE_writer.writerow(new_array)

# filename = 'test'
# if os.path.isfile('/opt/app/data/workspace/GEPrediction-OSRS/data/{}.csv'.format(filename)):
#     appendToCSV(filename)
# else:
#     writeToCSV(filename)

test_array = [1,2,3,4,5]
print(test_array[-3:])
print(datetime.datetime.utcnow().strftime("%m-%d-%Y_%H-%M"))