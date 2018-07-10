import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Import the dataset
dataset = pd.read_csv('Data.csv')

# Convert column names into lower case
dataset.columns = [x.lower() for x in dataset.columns]

# Get X and y
X = dataset.iloc[:,:-1].values #or iloc[:, :-1]
y = dataset.iloc[:,-1].values              

# Deal with missing data
'''
replace the missing data with the mean
'''
from sklearn.preprocessing import Imputer
# build an instance from the class Imputer, set it up
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
# only pull out the columns that have missing values
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


# Encode the categorical variables
'''
first step: label encoding
to convert string into numeric value
'''
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()    
X[:,0] = labelencoder_X.fit_transform(X[:,0])

'''
second step: dummy encoding
to generate dummy variables
'''
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

'''
since the value of y is already numeric
we only need label encoder
a one-hot encoding of y labels should use a LabelBinarizer instead.
'''
labelencoder_y = LabelEncoder()    
y = labelencoder_y.fit_transform(y)

# Split the dataset into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Feature scaling
'''
importance of feature scaling:
a lot of ML models are based on Euclidean distance
the distance will be dominated by feature that have a larger scale
'''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
