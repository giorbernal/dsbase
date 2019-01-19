# Table of Contents
- [DSBase Library](#dsbase-library)
- [ModelDSBase](#modeldsbase)
  * [Constructor](#constructor)
  * [Methods](#methods)
  * [Example](#example)
- [Regression Models](#regression-models)
    + [Evaluation set](#evaluation-set)
- [Classification Models](#classification-models)
    + [Evaluation set](#evaluation-set-1)
- [Docker](#docker)
***
# DSBase Library
The full name of this project would be "Data Science Base Models and Bootcamp". It is the base of real data science projects.

The project try to unify the interface of every Machine Learning Model, both for classification and regression. The final goal is to wrap them by the ModelDSBase, wich is the called "Model Data Sciece Base". Such model, is designed to systematically determinate whether a model has high Variance (overfitting), or on the other hand, high Bias.

# ModelDSBase
As we were telling previously. This is the entry point of the library. The ModelDSBase package contains the main class of the library, wich is the ModelDSBaseWrapper class. Here we are going to see the different params involved in the constructor, and the different methods of the class.
## Constructor
These are the params of the constructor:
- **Id**: Identifier of the model
- **X**: unlabeled dataset determined by an m x n matrix, being m the number of samples and n the number of features.
- **percentiles**: Data percentiles of data evaluated to check learning curves
- **y**: target of the training. Discrete for classification or continuous for regerssion.
- **test_perc (default: 0.3)**: percentage of data for testing purposes.
- **model**: Model used for training
- **parameters**: object map (specific class really) for the parameters of the model
- **splitter**: Class to perform the train/test splitting
- **normalizer**: Class to perform normalization

## Methods
These are the ModelDSBaseWrapper methods:
- **train()**: To train the model
- **predict(test_data)**: To execute predictions
- **save(folder_path=constants.PERSISTANCE_FOLDER)**: To persist the model
- **load(folder_path=constants.PERSISTANCE_FOLDER)**: To load a previously saved model
- **getLearningCurves()**: To visualize the learning curves
- **getScore()**: It retrieves model performance in a normalized metric between 0 and 1
- **close()**: free resources

## Example
Examples of use can be observed in [src/main/RegressionAirQuality.ipynb](https://github.com/giorbernal/dsbase/blob/master/src/test/RegressionAirQuality.ipynb) and [src/main/ClassificationDeposit.ipynb](https://github.com/giorbernal/dsbase/blob/develop/src/test/ClassificationDeposit.ipynb), Anyway here can see a example:
```Python
from RandomForestClassificationDSBase import RandomForestClassificationDSBaseModel
from RandomForestClassificationDSBase import RandomForestClassificationDSBaseModelParamsToMap
params = RandomForestClassificationDSBaseModelParamsToMap(100,15)
rfc = ModelDSBaseWrapper('RF',X,y,[70,75,80,85,90,95,100],0.3,RandomForestClassificationDSBaseModel,params,splitter=train_test_split)
rfc.train()
lcrfc = rfc.getLearningCurves()
# Print learning curves with this: plt.plot(lcrfc[0,:],'b',lcrfc[1,:],'r')
rfc.getScore()
rfc.close()
# ...
# Now loading model and predict ...
recoveredRfc = RandomForestClassificationDSBaseModel('RF6',None,None,None,None,None,None)
recoveredRfc.load()
predicted=recoveredRfc.predict(X[510:515,:])
```

# Regression Models
These are the available models for regression:
- **Linear Regression**: LinealRegressionDSBase.py - *sklearn* implementation
- **Random Forest Regression**: RandomForestRegressionDSBase.py - *sklearn* implementation
- **Gradient Boosting Regression**: GradientBoostingRegressionDSBase.py - *sklearn* implementation
- **Deep Neural Network Regression**: DNNRegressionDSBase.py - *Tensor Flow* implementation
### Evaluation set
For evaluating and testing this models we have used [this](https://archive.ics.uci.edu/ml/datasets/Air+Quality) dataset

# Classification Models
These are the available models for classification:
- **Support Vector Machines**: SVMClassificationDSBase.py - *sklearn* implementation
- **Random Forest Classifier**: RandomForestClassificationDSBase.py - *sklearn* implementation
- **Ada Boosting Classifier**: AdaBoostClassificationDSBase.py - *sklearn* implementation
- **Deep Neural Network Classifier**: DNNClassificationKerasDSBase.py - *keras* implementation
### Evaluation set
For Evaluating and testing this models we have used [this](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) dataset

# Docker
A docker container is provided to export the computation environment anywhere. To build the image, just execute this:
```
./build.sh
```
Then, you will have to set the following environment variables in your own *run.sh* script, retrieved from the template *run.sh.template*:
- KAGGLE_USER (*Optional*): Kaggle user if you need it
- KAGGLE_KEY (*Optional*): Kaggle key if you need it
- JUPYTER_PASSWORD (*Mandatory*): Password to access to the notebook, check jupyter [documentation](https://jupyter-notebook.readthedocs.io/en/stable/public_server.html) to set this encrypted password.

Run your initialization script:
```
./run.sh
```
After a while (a virtual environment is going to be set in the container), you will be able to access to the notebook by using the usual port 8888.