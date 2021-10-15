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



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
