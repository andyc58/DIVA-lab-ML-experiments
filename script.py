import pandas as pd
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from pprint import pprint
import asyncio
from MLExperiment import Experiment
from Evaluator import ModelEvaluator


load_dotenv()



"""
Need 

1. full body, 
2. head, 
3. controller (l + r hand), 
4. head + left + right hand, 
5. left + right foot
"""



data_dir = os.environ.get('DATA_DIR')
RAND_STATE = 42

models = [
          RandomForestClassifier(n_estimators=100, random_state=RAND_STATE, verbose=1),
          KNeighborsClassifier(n_neighbors=5,n_jobs=8), 
          XGBClassifier(random_state=RAND_STATE)
      ]


sessions = ['A', 'E', 'G', 'N']

body_regex_map = {
            "Full": "Head|Hand|Hips|Foot",
            "Headset": "Head",
            "Controller": "Hand",
            "Headset and Controller":"Head|Hand",
            "Foot": "Foot"
            }

results_dir = './results/'
experiment = Experiment(data_dir, results_dir, models, sessions, body_regex_map)
asyncio.run(experiment.run())
experiment.save_results()
