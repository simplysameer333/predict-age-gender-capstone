import numpy as np
import pandas as pd
import warnings
import sklearn
import boto3
import xgboost
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

import os
print(os.environ['HOME'])

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

print(xgb_best_gender_model)
print(xgb_best_age_group_model)

"""## **4.2 Best Test data from S3**"""

data_path = 'test_sample/'

test_data_50_sample_path = 'test_data_50_sample.npz'
test_50_device_ids_path = 'test_50_device_ids.csv'

s3.Bucket(BUCKET_NAME).download_file(data_path+test_50_device_ids_path, test_50_device_ids_path)
s3.Bucket(BUCKET_NAME).download_file(data_path+test_data_50_sample_path, test_data_50_sample_path)

test_data_50_sample = sparse.load_npz(test_data_50_sample_path)
test_50_device_id_df = pd.read_csv(test_50_device_ids_path, encoding='utf-8', dtype={'app_id': 'string'} )


@app.route('/')
def home():
    return 'Hello World'


@app.route('/predict_gender_age', methods=['GET'])
def predict_gender():
	xgb_best_gender_model_prediction = xgb_best_gender_model.predict_proba(test_data_50_sample)
	xgb_best_gender_model_prediction_df = pd.DataFrame(data=xgb_best_gender_model_prediction, columns=["M", "F"])
	device_ids_gender_prbabilities_df = pd.concat([test_50_device_id_df, xgb_best_gender_model_prediction_df], axis = 1)
	xgb_best_age_group_prediction = xgb_best_age_group_model.predict_proba(test_data_50_sample)

	xgb_best_gender_model_prediction_df = pd.DataFrame(data=xgb_best_age_group_prediction, columns=["0-24", "25-32", "32+"])
	device_ids_gender_prbabilities_df = pd.concat([device_ids_gender_prbabilities_df, xgb_best_gender_model_prediction_df], axis = 1)
	return jsonify(device_ids_gender_prbabilities_df.to_json())  

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")