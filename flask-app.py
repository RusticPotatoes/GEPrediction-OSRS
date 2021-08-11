from flask import Flask, render_template, request, jsonify
import pandas as pd
import csv
import json
import os

parent_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(parent_dir,"data")
runelite_dir = os.path.join(data_dir,"runelite")
predict_dir= os.path.join(data_dir,"prediction")

app = Flask(__name__, template_folder='./templates', static_folder='./static')

@app.route('/')
def index():
	# items_predicted = ['Old_school_bond', 'Rune_platebody', 'Adamant_platebody', "Red_spiders'_eggs", 'Ruby_necklace', 'Amulet_of_strength']
	# items_predicted = ["Red_spiders'_eggs", 'Ruby_necklace', 'Amulet_of_strength', "Green_d'hide_vamb", 'Staff_of_fire', \
	# 	'Blue_wizard_robe', 'Adamant_axe', 'Adamant_scimitar', 'Zamorak_monk_top', 'Staff_of_water', 'Staff_of_air', \
	# 		'Adamantite_bar', 'Amulet_of_power', "Green_d'hide_chaps", 'Mithril_platebody', 'Zamorak_monk_bottom', \
	# 			"Green_d'hide_body", 'Rune_axe', 'Adamant_platebody', 'Runite_ore', 'Rune_scimitar', 'Rune_pickaxe', \
	# 				'Rune_full_helm', 'Rune_kiteshield', 'Rune_2h_sword', 'Rune_platelegs', 'Rune_platebody', 'Old_school_bond']
	
	items_predicted = ['Mithril_bar','Air_battlestaff','Red_chinchompa','Manta_ray','Saradomin_brew(4)','Anglerfish','Purple_sweets','Anti-venom+(4)','Cactus_spine']
			
	data = {}
	names = {}
	count = 0 
	
	buy_avg = pd.read_csv(os.path.join(runelite_dir,'buy_average.csv'))
	buy_avg = buy_avg.set_index('timestamp')
	buy_avg = buy_avg.drop_duplicates()
	buy_avg = buy_avg.reset_index()
	buy_avg = buy_avg.replace(to_replace=0, method='ffill')

	for item_predicted in items_predicted:
		df = pd.read_csv(os.path.join(predict_dir,'{}.csv'.format(item_predicted)))

		current_df = buy_avg[['timestamp', item_predicted]]
		current_df = current_df.rename(columns={'timestamp': 'ts', item_predicted: 'real'})

		merged_df = pd.merge_asof(df, current_df, left_on='timestamp', right_on='ts', direction='backward')
		merged_df = merged_df.tail(48)  # Only show the last 48 time steps (24 hours worth of data)
		chart_data = merged_df.to_dict(orient='records')
		data['{}'.format(count)] = chart_data
		names[count] = item_predicted
		count += 1
		# print(data)

	return render_template("index.html", data=data, names=names)

# Webserver's suggestion engine
@app.route('/suggest')
def suggest():
	
	# Prediction index is the type of prediction we want to compare to
	# 1 - univariate
	# 2 - multivar single step
	# 3 to 7 - multivar multi steps 1 to 5 
	prediction_index = request.args.get("pred", default=1, type=int)
	if (prediction_index > 7): return "Incorrect prediction index, please enter number between 1 and 7."

	items_predicted = ['Mithril_bar','Air_battlestaff','Red_chinchompa','Manta_ray','Saradomin_brew(4)','Anglerfish','Purple_sweets','Anti-venom+(4)','Cactus_spine']
				
	data = {}
	
	buy_avg = pd.read_csv(os.path.join(runelite_dir,'buy_average.csv'))
	buy_avg = buy_avg.set_index('timestamp')
	buy_avg = buy_avg.drop_duplicates()
	buy_avg = buy_avg.replace(to_replace=0, method='ffill')

	# Get latest real price values 
	temp_last_row = None
	with open(os.path.join(predict_dir,'{}.csv'.format(items_predicted[0])), mode='r') as infile:
		for row in reversed(list(csv.reader(infile))):
			temp_last_row = row
			break

	buy_avg = buy_avg.loc[~buy_avg.index.duplicated(keep='first')]
	closest_real_values = buy_avg.iloc[buy_avg.index.get_loc(int(temp_last_row[0]), method='nearest')]

	# Get all values predicted
	for item_predicted in items_predicted:
		with open(os.path.join(predict_dir,'{}.csv'.format(item_predicted)), mode='r') as infile:
			last_row = None
			for row in reversed(list(csv.reader(infile))):
				last_row = row
				break

			# Save the real, predicted and profit values
			real_val = int(closest_real_values[item_predicted])
			pred_val = int(last_row[prediction_index])
			data[item_predicted] = [real_val, pred_val, pred_val-real_val]
	
	model_used = ""
	if (prediction_index == 1): model_used = "univariate"
	elif (prediction_index == 2): model_used = "multivariate single step"
	elif (prediction_index > 2): model_used = "multivariate multi step - {} steps forward".format(prediction_index - 2)

	return render_template("suggest.html", data=data, title=model_used)

# A route to return all of the available entries in our catalog.
@app.route('/api', methods=['GET'])
def api_all():
	if 'name' in request.args:
	    name = str(request.args['name'])
	else:
	    return "Error: No name field provided. Please specify an name."

	try:
		with open(os.path.join(predict_dir,'{}.csv'.format(name)), mode='r') as infile:
			
			reader = csv.reader(infile)
			header_row = next(reader)
			print(header_row)
			last_row = []
			for row in reader:
				last_row = row
			mydict = {name:last_row[idx] for idx, name in enumerate(header_row)}
	except EnvironmentError:
		return "File not found. Please specify different name"	

	return jsonify(mydict)

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=80)