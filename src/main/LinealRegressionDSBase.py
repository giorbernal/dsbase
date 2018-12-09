import numpy as np
import ConstantsDSBase as constants

from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib

description='LinearRegression'

class LinealRegressionDSBaseModel:
    def __init__(self, id, X, y, test_perc, parameters, splitter, normalizer):
        self.id=id
        if (X is not None):
            print("initiating model " + str(self.id) + ". " + description);
        else:
            print("initiating empty model " + str(self.id) + ". " + description);
            return

        X_train, X_test, y_train, y_test = splitter(X, y, test_size=test_perc, random_state=42)
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test

        self.model = LinearRegression(normalize=parameters['normalize'])
        
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
def LinealRegressionDSBaseParamsToMap(normalize=False):
    params={}
    params['normalize']=normalize 
    return params

