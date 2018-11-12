import numpy as np
import ConstantsDSBase as constants
import tensorflow as tf

# Module to handle systematic Variance/Bias trade-off

# BaseModel. Reference for every model
class BaseModel:
    def __init__(self, id, X, y, test_perc, parameters, splitter, normalizer):
        self.id=id
        self.X=X
        self.y=y
        print("initiating model " + str(self.id) + ". DO NOTHING. Just information: " + str(X.shape));
        
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
    def __init__(self, id, X, y, percentiles, test_perc=0.3, model=BaseModel, parameters={}, splitter=None, normalizer=None):
        self.id = id
        print("X size:" + str(X.shape))
        print("y size:" + str(y.shape))

        self.models = []
        len=X.shape[0]
        i = 0
        for p in percentiles:
            index=int(len*p/100)
            m=model(self.id + str(i),X[0:index,:], y[0:index], test_perc, parameters, splitter, normalizer)
            self.models.append(m)
            i=i+1

        self.model=self.models[i-1]

    # Only for algorithms based on session (i.e: TensorFlow)
    def startSession(self):
        self.sess = tf.Session()
        for model in self.models:
            model.setSession(self.sess)

    def closeSession(self):
        self.sess.close()

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
