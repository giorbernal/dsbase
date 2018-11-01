import numpy as np

# Module to handle systematic Variance/Bias trade-off

# BaseModel. Reference for every model
class BaseModel:
    def __init__(self, id, data, parameters):
        self.id=id
        self.data=data
        print("initiating model " + str(self.id) + ". DO NOTHING. Just information: " + str(data.shape));
        
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
class LCModel:
    def __init__(self, data, percentiles, model=BaseModel, parameters={}):
        print("data size:" + str(data.shape))
        
        self.models = []
        len=data.shape[0]
        i = 0
        for p in percentiles:
            m=model(i,data[0:int(len*p/100),:], parameters)
            self.models.append(m)
            i=i+1

        self.model=self.models[i-1]

    def train(self):
        for m in self.models:
            m.train()
    
    def predict(self, test_data):
        m.model.predict(test_data)
        
    def save(self):
        self.model.save()

    def load(self, file):
        self.model.load(file)
    
    def getLearningCurves(self):
        trainScores = []
        testScores = []
        
        for m in self.models:
            trainScores.append(m.getTrainScore())
            testScores.append(m.getTestScore())
        
        result = np.stack([trainScores,testScores],axis=0)
        
        return result
