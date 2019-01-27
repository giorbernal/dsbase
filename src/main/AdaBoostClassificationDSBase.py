import numpy as np
import ConstantsDSBase as constants

from sklearn.ensemble import AdaBoostClassifier 
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

description='AdaBoostClassification'

class AdaBoostClassificationDSBaseModel:
    def __init__(self, id, X, y, test_perc, parameters, splitter, normalizer):
        self.id=id
        if (X is not None):
            print("initiating model " + str(self.id) + ". " + description);
        else:
            print("initiating empty model " + str(self.id) + ". " + description);
            return

        # Normalizing
        self.scalerX=MinMaxScaler()
        X_s = self.scalerX.fit_transform(X)
        self.scalery=MinMaxScaler()
        y_s = self.scalery.fit_transform(y.reshape(-1, 1))

        X_train, X_test, y_train, y_test = splitter(X_s, y_s, test_size=test_perc, random_state=42)
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test

        self.model = AdaBoostClassifier(n_estimators=parameters['n_estimators'], learning_rate=parameters['learning_rate'],
                            random_state=None)
        
    def train(self):
        print("training model " + str(self.id) + ". " + description);
        self.model.fit(self.X_train, self.y_train)

    def predict(self, test_data):
        print("predicting model " + str(self.id) + ". " + description);
        test_data_norm = self.scalerX.transform(test_data)
        return self.model.predict(test_data_norm)
        
    def getTrainScore(self):
        return self.model.score(self.X_train, self.y_train);
    
    def getTestScore(self):
        return self.model.score(self.X_test, self.y_test);
        
    def save(self, folder_path=constants.PERSISTANCE_FOLDER):
        file_path=folder_path + constants.SEP + description + "_" + str(self.id) + constants.EXT
        print("saving model: " + file_path)
        joblib.dump(self.model, file_path)
        scaler_file_path_root=folder_path + constants.SEP + description + "_" + str(self.id)
        joblib.dump(self.scalerX, scaler_file_path_root + "_scalerX" + constants.EXT)
        joblib.dump(self.scalery, scaler_file_path_root + "_scalery" + constants.EXT)
    
    def load(self, folder_path=constants.PERSISTANCE_FOLDER):
        file_path=folder_path + constants.SEP + description + "_" + str(self.id) + constants.EXT
        print("loading model: " + file_path)
        scaler_file_path_root=folder_path + constants.SEP + description + "_" + str(self.id)
        self.model = joblib.load(file_path)
        self.scalerX=joblib.load(scaler_file_path_root + "_scalerX" + constants.EXT)
        self.scalery=joblib.load(scaler_file_path_root + "_scalery" + constants.EXT)


    def close(self):
        pass

# Params converter function. Reference for every model
def AdaBoostClassificationDSBaseModelParamsToMap(n_estimators=50, learning_rate=1.0):
    params={}
    params['n_estimators']=n_estimators
    params['learning_rate']=learning_rate
    return params

