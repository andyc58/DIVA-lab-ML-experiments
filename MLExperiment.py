import pandas as pd
import os
from dotenv import load_dotenv
import numpy as np
import json
from Evaluator import ModelEvaluator 

class ResultBuilder:
    @classmethod
    def from_regex_map(cls, regex_map, data_key):
        results = {}
        for key, v in regex_map.items():
            results[key] = {"regex": v, data_key: {}}
        return results



class DataLoader:
    def __init__(self, data_path, sessions):
        self.data_path = data_path
        self.sessions = sessions
        self.datasets = self._load_all_sessions()
        print("Data Loaded")

    def _load_session(self, session):
        path = os.path.join(self.data_path, f"type_{session}_df.csv")
        df = pd.read_csv(path)

        # Fill missing values in numeric columns
        for col in df.select_dtypes(include=np.number).columns:
            if df[col].isnull().any():
                df.fillna({col: df[col].mean()}, inplace=True)
                
        return df
    
    def _load_all_sessions(self):
        return {session: self._load_session(session) for session in self.sessions}

    def get(self, session):
        return self.datasets[session]

    def get_pair(self, train_session, test_session):
        return self.get(train_session), self.get(test_session)

    def list_sessions(self):
        return list(self.datasets.keys())


class Experiment:
    
    def __init__(self, data_path, results_dir, models, sessions, regex_map) -> None:
        self.split_store_key="splits"
        self.results_dir = results_dir
        self.evaluator = ModelEvaluator(models, results_dir)
        
        print(f"Evaluating {len(self.evaluator.models)} classification models")
        self.data_path = data_path
        self.sessions = sessions
        self.loader = DataLoader(data_path, sessions)
        self.results = self.__generate_pairings(regex_map)
        
        
    def __generate_pairings(self, regex_map):
        results = ResultBuilder.from_regex_map(regex_map, self.split_store_key)
        
        for t1 in range(len(self.sessions)):
            for t2 in range(len(self.sessions)):
                if t1 != t2:
                    for r in results:
                        results[r][self.split_store_key]["|".join((self.sessions[t1], self.sessions[t2]))] = {}
        return results




    async def run(self):
        
        for part in self.results:
            regex = self.results[part]['regex']

            for split in self.results[part][self.split_store_key]:
                sample = f"{part}/{split.replace('|', '_')}"
                train_df, test_df = self.loader.get_pair(*split.split('|'))

                X_train = train_df.filter(regex=regex, axis=1)
                y_train = train_df['ETHNICITY_ENC']
                X_test = test_df.filter(regex=regex, axis=1)
                y_test = test_df['ETHNICITY_ENC']

                model_results = await self.evaluator.evaluate(X_train, X_test, y_train, y_test, sample)
                self.results[part][self.split_store_key][split] = model_results
 
                
    def save_results(self):
        with open(os.path.join(self.results_dir, "results.json"), 'w') as f:
            json.dump(self.results, f)
                
    
        
    
    