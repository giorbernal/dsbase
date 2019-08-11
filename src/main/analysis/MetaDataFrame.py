import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

class MetaDataFrame:
    def __init__(self, df_train, df_test, explicit_cat_columns, target):
        self.df_train =df_train
        self.df_test = df_test
        self.target = target
        self.explicit_cat_columns = []
        self.explicit_cat_columns.append(explicit_cat_columns)
        
        df_train_ann = pd.concat([df_train, pd.DataFrame(data=np.ones(df_train.shape[0]), columns=['TT'])], axis=1)
        df_test_ann = pd.concat([df_test, pd.DataFrame(data=np.zeros(df_test.shape[0]), columns=['TT'])], axis=1)    
        self.df = pd.concat([df_train_ann, df_test_ann], axis = 0, sort=True)
    
    def getNullColumns(self, df_nulls, null_th, inplace=False):
        df_perc_nulls = pd.DataFrame(df_nulls.sum()/df_nulls.shape[0],columns=['perc_nulls'])
        nulls_columns = df_perc_nulls[df_perc_nulls['perc_nulls'] > null_th].index
        print('nullable columns:',nulls_columns,'They will be removed if inplace is True:',inplace)
        return (nulls_columns, self.df.drop(nulls_columns, axis=1, inplace=inplace))
    
    def addExplicitCatColumn(self, explicit_cat_columns):
        self.explicit_cat_columns.append(explicit_cat_columns)
        
    def getColumnsByType(self):
        discrete_columns = []
        numeric_columns = ['TT']
        for c in self.df.columns:
            if ( (c in self.explicit_cat_columns) | (self.df[c].dtype == 'object') ):
                discrete_columns.append(c)
            else:
                numeric_columns.append(c)
        print('Discrete columns:',discrete_columns, '. Total:',len(discrete_columns))
        print('Numeric columns:',numeric_columns, '. Total:',len(numeric_columns))
        self.discrete_columns, self.numeric_columns = discrete_columns, numeric_columns
        return (discrete_columns, numeric_columns)

    # Imputation methods For Categorical and Numeric Values. Strategy 1
    # Let's follow two strategies:
    # * **Categorial**: New type NULL
    # * **Numerical**: 0, and create a categorical column    
    def imputeNullData(self, inplace=False):
        M = self.df.shape[0]
        for c in self.df.columns:
            if (M > self.df[c].dropna().shape[0]):
                if (c in self.discrete_columns):
                    print('Categorical column with null values: ',c)
                    self.df[c] = np.where(self.df[c].isna(), 'NULL', self.df[c])
                elif (c in self.numeric_columns):
                    print('Numerical column with null values: ',c)
                    self.df[c + '_isnan'] = self.df[c].apply(lambda x: 1 if (math.isnan(x)) else 0)
                    self.df[c] = np.where(self.df[c].isna(), 0, self.df[c])
                else:
                    print('Unknown column')
        return self.df

    # Categorical columns encoding for Machine Learning purposes
    def categoricalEncoding(self, type, columns):
        if (type == 'one-hot'):
            self.df = pd.get_dummies(data=self.df, columns=columns)
        elif (type == 'frecuency'):
            for c in columns:
                counter = self.df[c].value_counts()/self.df[c].shape[0]
                self.df[c] = self.df[c].map(lambda x: counter[x])
        else:
            print('encode mode not supported')

    def automaticCategoricalEncoding(self, threshool):
        for c in self.discrete_columns:
            counter = self.df[c].value_counts()/self.df[c].shape[0]
            if (counter.shape[0] <= threshool):
                self.categoricalEncoding('one-hot',[c])
            else:
                self.categoricalEncoding('frecuency',[c])

    # Draw Support functions

    def drawNulls(self, df):
        df_nulls = pd.isna(df)
        sns.heatmap(~df_nulls)
        return df_nulls

    # Simple Count Plot
    def drawBasicDiscreteColumn(self, column):
        sns.countplot(data=self.df, x=column,order=self.df[column].value_counts().index)
        
    # Simple Histogram
    def drawBasicNumericColumn(self,column):
        sns.distplot(self.df[column].dropna(), kde=False)
    
    def drawBaseColumns(self, type):
        if (type == 'Categorical'):
            columns = self.discrete_columns
            drawFunction = self.drawBasicDiscreteColumn
        else:
            columns = self.numeric_columns
            drawFunction = self.drawBasicNumericColumn

        N = len(columns)
        plt.figure(figsize=(15,5*N))
        index = 1
        for c in columns:
            plt.subplot(N,1,index)
            drawFunction(c)
            index+=1

    # For Regression cases
    def drawCompareCategoricalColumnWithNumericTarget(self):
        columns = self.discrete_columns
        drawFunction = self.drawBasicDiscreteColumn
        df = self.df[self.df['TT']==1]
        N = len(columns)
        #plt.figure(figsize=(15,5*N))
        plt.figure(figsize=(20,7*N))
        index = 1
        for c in columns:
            plt.subplot(N,2,index)
            drawFunction(c)
            plt.subplot(N,2,index + 1)
            sns.boxplot(data=df, y=self.target, x=c)
            index+=2

    # For Regression cases.
    # WARN: It does not work. ??????
    def drawCompareNumericalColumnWithNumericTarget(self,):
        columns = self.numeric_columns
        df = self.df[self.df['TT']==1]
        N = len(columns)
        #plt.figure(figsize=(15,5*N))
        plt.figure(figsize=(20,7*N))
        index = 1
        for c in columns:
            plt.subplot(N,1,index)
            sns.jointplot(data=df, y=self.target,x=c, kind='scatter')
            index+=1

    # TODO For classification cases
    def drawCompareCategoricalColumnWithCategoricalTarget(self, type):
        pass

    # TODO For classification cases
    def drawCompareNumeticalColumnWithNumericTarget(self, type):
        pass

    # Only after encoding
    def drawMap(self, type):
        corr = self.df.corr()
        plt.figure(figsize=(15,15))
        if (type == 'heat'):
            sns.heatmap(corr)
        elif (type == 'cluster'):
            sns.clustermap(corr)
        else:
            print('Unknown map type')
