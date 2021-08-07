import requests
import json
import csv
import os 

parent_dir = os.path.dirname(os.path.realpath(__file__))

csvName = 'Rune_Data'

# get more item codes here: https://everythingrs.com/tools/osrs/itemlist/238
# itemList = [1521, 1519, 1517, 1515]  # Logs
itemList = [24488,12825,12827,22481,22978,13652,22547,10352,10350,10340,12389,13036,13441,391,385,1513,11998,22246,23348,20065,20062]  # Runes

fullDict = {}
labels = ['timestamp']

# Construct dictionary full of data
for itemID in itemList:
    r = requests.get('http://services.runescape.com/m=itemdb_oldschool/api/graph/{}.json'.format(itemID))
    json_data = json.loads(r.text)
    current_daily_dict = json_data['daily']

    for daily_timestamp in current_daily_dict:
        if (daily_timestamp in fullDict):
            fullDict[daily_timestamp].append(current_daily_dict[daily_timestamp])
        else:
            fullDict[daily_timestamp] = [current_daily_dict[daily_timestamp]]
    
    r2 = requests.get('http://services.runescape.com/m=itemdb_oldschool/api/catalogue/detail.json', params={'item': itemID})
    labels.append(json.loads(r2.text)['item']['name'].replace(" ", "_"))

# print(fullDict)


# Write to CSV file
with open(os.path.join(os.path.dirname(parent_dir), 'data/{}.csv'.format(csvName)), mode='w', newline='') as GE_data:
    GE_writer = csv.writer(GE_data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    GE_writer.writerow(labels)  # write field names

    for daily_timestamp in fullDict:
        new_array = [daily_timestamp]
        new_array.extend(fullDict[daily_timestamp])
        # print(new_array)
        GE_writer.writerow(new_array)