import numpy as np

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
        
    def save(self):
        pass
    
    def load(self, file):
        pass
    
# Params converter function. Reference for every model
def basicModelParamsToMap(param1,param2,param3,param4):
    params={}
    params['p1']=param1
    params['p2']=param2
    params['p3']=param3
    params['p4']=param4    
    return params

# Learning Curves Model
class LearningCurvesDSBaseWrapper:
    def __init__(self, X, y, percentiles, test_perc=0.3, model=BaseModel, parameters={}, splitter=None, normalizer=None):
        print("X size:" + str(X.shape))
        print("y size:" + str(y.shape))

        self.models = []
        len=X.shape[0]
        i = 0
        for p in percentiles:
            index=int(len*p/100)
            m=model(i,X[0:index,:], y[0:index], test_perc, parameters, splitter, normalizer)
            self.models.append(m)
            i=i+1

        self.model=self.models[i-1]

    def train(self):
        for m in self.models:
            m.train()
    
    def predict(self, test_data):
        return m.model.predict(test_data)
        
    def save(self):
        self.model.save()

    def load(self, file):
        self.model.load(file)
    
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
