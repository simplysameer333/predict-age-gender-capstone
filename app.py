import numpy as np
import pandas as pd
import warnings
import sklearn
import boto3
import os
from scipy import sparse, io
from joblib import dump, load
from flask import Flask, request, jsonify, render_template

warnings.filterwarnings("ignore")


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
	xgb_best_gender_model_prediction = xgb_best_gender_model.predict(Xtest_events)
	event_test_data['predict_gender'] = xgb_best_gender_model_prediction
	#F=0, M=1
	female_prediction = event_test_data[event_test_data['predict_gender']==0][['device_id']]
	return jsonify(female_prediction.to_json())  

@app.route('/predict_male', methods=['GET'])
def predect_male_customers():
	xgb_best_gender_model_prediction = xgb_best_gender_model.predict(Xtest_events)
	event_test_data['predict_gender'] = xgb_best_gender_model_prediction
	#F=0, M=1
	male_prediction = event_test_data[event_test_data['predict_gender']==1][['device_id']]
	return jsonify(male_prediction.to_json())  

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
