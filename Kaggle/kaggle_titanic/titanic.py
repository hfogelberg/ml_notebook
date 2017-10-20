import numpy as np
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import  confusion_matrix
from keras.models import Sequential
from keras.layers import Dense

ds = pd.read_csv('train.csv')
ds.head()
ds.describe()

print("Missing values")
print(ds.isnull().sum())

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    # Return mean age for different passenger classes,
    # given by the boxplot
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

ds['Age'] = ds[['Age', 'Pclass']].apply(impute_age, axis=1)

ds.drop(['PassengerId', 'Cabin', 'Name', 'Ticket', 'Embarked'], axis=1, inplace=True)

# split between dependent and independent variables
X_train = ds.iloc[:, 1:7].values
y_train = ds.iloc[:, 0]

# encode categorical data (Sex)
le = LabelEncoder()
X_train[:, 1] = le.fit_transform(X_train[:, 1])
ohe = OneHotEncoder(categorical_features=[1])
X_train = ohe.fit_transform(X_train).toarray()
X_train = X_train[:, 1:]

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

# define model as sequense of layers
classifier = Sequential()
classifier.add(Dense(units=4, kernel_initializer='uniform', activation='relu', input_dim=6))
classifier.add(Dense(units=4, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, y_train, batch_size=4, epochs=100)


