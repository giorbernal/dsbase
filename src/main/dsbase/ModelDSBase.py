import numpy as np
import dsbase.ConstantsDSBase as constants

# Module to handle systematic Variance/Bias trade-off

# BaseModel. Reference for every model
class BaseModel:
    def __init__(self, id, X_train, y_train, X_test, y_test, parameters):
        self.id=id
        self.X_train=X_train
        self.y_train=y_train
        print("initiating model " + str(self.id) + ". DO NOTHING. Just information: " + str(X_train.shape));
        
    def train(self):
        print("training model " + str(self.id) + ". DO NOTHING. Just information");
        
    def predict(self, test_data):
        print("predicting data " + str(self.id) + ". DO NOTHING. Just information");
        
    def getTrainScore(self):
        return 1;
    
    def getTestScore(self):
        return 0;
        
    def save(self, folder_path=constants.PERSISTANCE_FOLDER):
        pass
    
    def load(self, folder_path=constants.PERSISTANCE_FOLDER):
        pass
    
# Params converter function. Reference for every model
def basicModelParamsToMap(param1,param2,param3,param4):
    params={}
    params['p1']=param1
    params['p2']=param2
    params['p3']=param3
    params['p4']=param4    
    return params

# Model Wrapper
class ModelDSBaseWrapper:
    def __init__(self, id, X_train, y_train, X_test, y_test, percentiles, model=BaseModel, parameters={}):
        self.id = id
        print("X_train size:" + str(X_train.shape))
        print("y_train size:" + str(y_train.shape))
        print("X_test size:" + str(X_test.shape))
        print("y_test size:" + str(y_test.shape))

        self.models = []

        len=X_train.shape[0]
        i = 0
        for p in percentiles:
            index=int(len*p/100)
            m=model(self.id + str(i),X_train[0:index,:], y_train[0:index], X_test[0:index,:], y_test[0:index], parameters)
            self.models.append(m)
            i=i+1

        self.model=self.models[i-1]

    def close(self):
        for m in self.models:
            m.close()

    def train(self):
        for m in self.models:
            m.train()
    
    def predict(self, test_data):
        return self.model.predict(test_data)
        
    def save(self, folder_path=constants.PERSISTANCE_FOLDER):
        self.model.save(folder_path)

    def load(self, folder_path=constants.PERSISTANCE_FOLDER):
        self.model.load(folder_path)
    
    def getLearningCurves(self):
        self.trainScores = []
        self.testScores = []
        
        for m in self.models:
            self.trainScores.append(m.getTrainScore())
            self.testScores.append(m.getTestScore())
        
        result = np.stack([self.trainScores,self.testScores],axis=0)
        
        return result

    def getScore(self):
        return self.testScores[-1]
