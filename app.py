import numpy as np
import pandas as pd
import warnings
import sklearn
import boto3
import os
import pickle
from scipy import sparse, io
from joblib import dump, load
from flask import Flask, request, jsonify, render_template

warnings.filterwarnings("ignore")

def ks_statistics(probabilities, classes) :
  probabilities_df = pd.DataFrame(data=probabilities, columns=classes.classes_)
  probabilities_df = probabilities_df.sort_values('M').reset_index(drop=True)

  probabilities_df['bucket'] = pd.qcut(probabilities_df.M.rank(method='first'), 10, labels=[0,1,2,3,4,5,6,7,8,9])
  grouped = probabilities_df.groupby('bucket', as_index = False)

  # CREATE A SUMMARY DATA FRAME
  agg1 = grouped.min().M

  agg1 = pd.DataFrame(grouped.min().M, columns = ['min_scr'])
  agg1['min_scr'] = grouped.min().M
  agg1['max_scr'] = grouped.max().M
  agg1['male_count'] = grouped.sum().M
  agg1['female_count'] = grouped.sum().F
  agg1['total'] = agg1.male_count + agg1.female_count

  # SORT THE DATA FRAME BY Probability
  agg2 = (agg1.sort_values('min_scr')).reset_index(drop = True)
  agg2['odds'] = (agg2.male_count / agg2.female_count).apply('{0:.2f}'.format)
  agg2['male_ratio'] = (agg2.male_count / agg2.total).apply('{0:.2%}'.format)
  agg2['female_ratio'] = (agg2.female_count / agg2.total).apply('{0:.2%}'.format)

  # CALCULATE KS STATISTIC
  agg2['male_comm'] = ((agg2.male_count / probabilities_df.M.sum()).cumsum()).apply('{0:.2%}'.format)
  agg2['female_comm'] = ((agg2.female_count / probabilities_df.F.sum()).cumsum()).apply('{0:.2%}'.format)
  agg2['ks'] = abs(np.round(((agg2.male_count / probabilities_df.M.sum()).cumsum() - (agg2.female_count / probabilities_df.F.sum()).cumsum()), 4) * 100)

  # DEFINE A FUNCTION TO FLAG MAX KS
  flag = lambda x: '<----' if x == agg2.ks.max() else ''

  # FLAG OUT MAX KS
  agg2['max_ks'] = agg2.ks.apply(flag)

  return agg2

app = Flask(__name__)
"""# **4. Model Deployment**

## **4.1 Load best models from S3**
"""

BUCKET_NAME = "sameer-iitm-bucket-1"

s3 = boto3.resource("s3",
    aws_access_key_id = os.environ['aws_access_key_id'],
	aws_secret_access_key = os.environ['aws_secret_access_key']
)

model_path = 'model/'
best_gender_model = 'xgb_best_gender_event_model.joblib'
best_age_model = 'xgb_grid_age_group_event_model.joblib'

s3.Bucket(BUCKET_NAME).download_file(model_path+best_gender_model, best_gender_model)
s3.Bucket(BUCKET_NAME).download_file(model_path+best_age_model, best_age_model)

xgb_best_gender_model = load(best_gender_model)
xgb_best_age_group_model = load(best_age_model)

"""## **4.2 Best Test data from S3**"""

# Fetch data to be predicted (Test Data)
data_path = 'test_sample/'

test_data_50_sample_path = 'test_data_50_sample.npz'
test_50_device_ids_path = 'test_50_device_ids.csv'

s3.Bucket(BUCKET_NAME).download_file(data_path+test_50_device_ids_path, test_50_device_ids_path)
s3.Bucket(BUCKET_NAME).download_file(data_path+test_data_50_sample_path, test_data_50_sample_path)

test_data_50_sample = sparse.load_npz(test_data_50_sample_path)
test_50_device_id_df = pd.read_csv(test_50_device_ids_path, encoding='utf-8', dtype={'app_id': 'string'} )

# Load Preprossed event data
Xtest_events_path = 'Xtest_events.npz'
s3.Bucket(BUCKET_NAME).download_file(data_path+Xtest_events_path, Xtest_events_path)
Xtest_events = sparse.load_npz(Xtest_events_path)

#Load indexed device ids
test_data_df_path = 'test_data_df.csv'
s3.Bucket(BUCKET_NAME).download_file(data_path+test_data_df_path, test_data_df_path)
test_data_df = pd.read_csv(test_data_df_path, encoding='utf-8')

# Filter device ids w.r.t events
event_test_data = test_data_df[test_data_df['has_events']==1]

#Last line
print("Gender Prediction system is up ....")

def getGenderLabel():
	#Restore encoders
	encoder_folder = 'encoder/'

	#Gender label encoder
	label_gender_encoder_path = 'label_gender.sav'
	s3.Bucket(BUCKET_NAME).download_file(encoder_folder+label_gender_encoder_path, label_gender_encoder_path)
	file = open(label_gender_encoder_path, 'rb')
	label_gender_label_encoder = pickle.load(file)
	return label_gender_label_encoder

@app.route('/')
def home():
    return 'Hello World'

@app.route('/predict_gender_age', methods=['GET'])
def predict_gender():
	# gender probabilities
	xgb_best_gender_model_prediction = xgb_best_gender_model.predict_proba(test_data_50_sample)
	xgb_best_gender_model_prediction_df = pd.DataFrame(data=xgb_best_gender_model_prediction, columns=["M", "F"])

	# Add device ids (matched by indices)
	device_ids_gender_prbabilities_df = pd.concat([test_50_device_id_df, xgb_best_gender_model_prediction_df], axis = 1)

	# age probabilities
	xgb_best_age_group_prediction = xgb_best_age_group_model.predict_proba(test_data_50_sample)
	xgb_best_gender_model_prediction_df = pd.DataFrame(data=xgb_best_age_group_prediction, columns=["0-24", "25-32", "32+"])

	#merge gender and age
	device_ids_gender_prbabilities_df = pd.concat([device_ids_gender_prbabilities_df, xgb_best_gender_model_prediction_df], axis = 1)
	return jsonify(device_ids_gender_prbabilities_df.to_json())  


@app.route('/predict_female', methods=['GET'])
def predect_female_customers():

	label_gender_label_encoder = getGenderLabel()
	
	xgb_best_gender_model_prob = xgb_best_gender_model.predict_proba(Xtest_events)
	ks_table_gender = ks_statistics(xgb_best_gender_model_prob,label_gender_label_encoder)

	#On the basis of this we can filter female(index start with zero), in and above 8 decile
	min_probability_8_decile = ks_table_gender['min_scr'][7]

	#Probability of female and male prediction
	event_test_data['female_probability'] = xgb_best_gender_model_prob[:,0]
	female_prediction = event_test_data[event_test_data['female_probability']>=min_probability_8_decile][['device_id']]

	return jsonify(female_prediction.to_json())  

@app.route('/predict_male', methods=['GET'])
def predect_male_customers():

	label_gender_label_encoder = getGenderLabel()
	
	xgb_best_gender_model_prob = xgb_best_gender_model.predict_proba(Xtest_events)
	ks_table_gender = ks_statistics(xgb_best_gender_model_prob,label_gender_label_encoder)

	#On the basis of this we can filter male(index start with zero), in and lesser 3 decile
	max_probability_3_decile = ks_table_gender['max_scr'][2]
	
	# Probability of female prediction
	event_test_data['male_probability'] = xgb_best_gender_model_prob[:,1]
	male_prediction = event_test_data[event_test_data['male_probability']<=max_probability_3_decile][['device_id']]
	return jsonify(male_prediction.to_json())  

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
