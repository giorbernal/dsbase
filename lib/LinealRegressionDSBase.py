import numpy as np
from sklearn.linear_model import LinearRegression

# BaseModel. Reference for every model
class LinealRegressionDSBaseModel:
    def __init__(self, id, X, y, test_perc, parameters, splitter, normalizer):
        self.id=id
        print("initiating model " + str(self.id) + ". LinearRegression");

        X_train, X_test, y_train, y_test = splitter(X, y, test_size=test_perc, random_state=42)
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test

        self.model = LinearRegression(normalize=parameters['normalize'])
        
    def train(self):
        print("training model " + str(self.id) + ". LinearRegression");
        self.model.fit(self.X_train, self.y_train)

        
    def predict(self, test_data):
        print("predicting model " + str(self.id) + ". LinearRegression");
        return self.model.predict(test_data)
        
    def getTrainScore(self):
        return self.model.score(self.X_train, self.y_train);
    
    def getTestScore(self):
        return self.model.score(self.X_test, self.y_test);
        
    def save(self):
        pass
    
    def load(self, file):
        pass

# Params converter function. Reference for every model
def LinealRegressionDSBaseParamsToMap(normalize=False):
    params={}
    params['normalize']=normalize 
    return params

