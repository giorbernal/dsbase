import numpy as np
import dsbase.ConstantsDSBase as constants

from lightgbm import LGBMClassifier
from sklearn.externals import joblib

description='LightGradientBoostingClassification'

class LightGradientBoostingClassificationDSBaseModel:
    def __init__(self, id, X_train, y_train, X_test, y_test, parameters):
        self.id=id
        if (X_train is not None):
            print("initiating model " + str(self.id) + ". " + description);
        else:
            print("initiating empty model " + str(self.id) + ". " + description);
            return

        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test

        self.model = LGBMClassifier(
            n_estimators=parameters['n_estimators'],
            max_depth=parameters['max_depth'], 
            learning_rate=parameters['learning_rate'],
            objective=parameters['objetive'],
            n_jobs=parameters['n_jobs'],
            num_leaves=parameters['num_leaves'])
        
    def train(self):
        print("training model " + str(self.id) + ". " + description);
        self.model.fit(self.X_train, self.y_train)

    def predict(self, test_data):
        print("predicting model " + str(self.id) + ". " + description);
        return self.model.predict(test_data)
        
    def getTrainScore(self):
        return self.model.score(self.X_train, self.y_train);
    
    def getTestScore(self):
        return self.model.score(self.X_test, self.y_test);
        
    def save(self, folder_path=constants.PERSISTANCE_FOLDER):
        file_path=folder_path + constants.SEP + description + "_" + str(self.id) + constants.EXT
        print("saving model: " + file_path)
        joblib.dump(self.model, file_path)
    
    def load(self, folder_path=constants.PERSISTANCE_FOLDER):
        file_path=folder_path + constants.SEP + description + "_" + str(self.id) + constants.EXT
        print("loading model: " + file_path)
        self.model = joblib.load(file_path)

    def close(self):
        pass

# Params converter function. Reference for every model
def LightGradientBoostingClassificationDSBaseModelParamsToMap(n_estimators=100, max_depth=10,learning_rate=0.1,objetive='binary',n_jobs=1, num_leaves=31):
    params={}
    params['n_estimators']=n_estimators
    params['max_depth']=max_depth
    params['learning_rate']=learning_rate
    params['objetive']=objetive
    params['n_jobs']=n_jobs
    params['num_leaves']=num_leaves
    return params

