import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

diabetes_dataset = pd.read_csv("diabetes_dataset.csv")

def clas(value):
    if value == 'tested_negative':
        return 0
    elif value == 'tested_positive':
        return 1
diabetes_dataset['class'] = diabetes_dataset['class'].apply(clas)

x = diabetes_dataset.drop('class', axis = 1)
y = diabetes_dataset['class']

# Initiatlize the model
logreg = LogisticRegression(solver='liblinear', random_state = 0)

# Fit the model
logreg.fit(x, y)  

pickle.dump(logreg, open('predic_model.pkl', 'wb'))  
