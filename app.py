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

'''
K-Table to divide probabilities in 10 deciles. Base probabilities are based on 'male'
'''
def ks_statistics(probabilities, classes) :
  probabilities_df = pd.DataFrame(data=probabilities, columns=classes.classes_)
  probabilities_df = probabilities_df.sort_values('M').reset_index(drop=True)

  probabilities_df['bucket'] = pd.qcut(probabilities_df.M.rank(method='first'), 10, labels=[0,1,2,3,4,5,6,7,8,9])
  grouped = probabilities_df.groupby('bucket', as_index = False)

  #