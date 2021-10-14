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

print(xgb_best_gender_model)
print(xgb_best_age_group_model)

"""## **4.2 Best Test data from S3**"""

# Fetch data to be predicted (Test Data)
data_path = 'test_sample/'
test_data_50_sample_path = 'test_data_50_sample.npz'
s3.Bucket(BUCKET_NAME).download_file(data_path+test_data_50_sample_path, test_data_50_sample_path)
test_data_50_sample = sparse.load_npz(test_data_50_sample_path)

@app.route('/')
def home():
    return 'Hello World'

@app.route('/predict_gender_age', methods=['GET'])
def predict_gender():
	# gender probabilities
	xgb_best_gender_prediction = xgb_best_gender_model.predict_proba(test_data_50_sample)
	xgb_best_gender_prediction_df = pd.DataFrame(data=xgb_best_gender_prediction, columns=["M", "F"])
	
	# age probabilities
	xgb_best_age_group_prediction = xgb_best_age_group_model.predict_proba(test_data_50_sample)
	xgb_best_age_group_prediction_df = pd.DataFrame(data=xgb_best_age_group_prediction, columns=["0-24", "25-32", "32+"])

	#merge gender and age
	gender_age_group_prbabilities_df = pd.concat([xgb_best_gender_prediction_df, xgb_best_age_group_prediction_df], axis = 1)
	return jsonify(gender_age_group_prbabilities_df.to_json())  

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
