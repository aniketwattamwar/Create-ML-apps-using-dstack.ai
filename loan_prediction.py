import dstack as ds
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 

@ds.cache()
def get_data():
    data = pd.read_csv("D:\\ML_projects\\dstack_proj\\train_ctrUa4K.csv")
    return data

@ds.cache()
def get_testdata():
    return pd.read_csv("D:\\ML_projects\\dstack_proj\\test_lAUu6dG.csv")

def scatter_handler():
    df = get_data()
    return px.scatter(df, x="ApplicantIncome", y="LoanAmount", color="Education")


def bar_handler():
    df = get_data()
    return px.bar(df, x="Gender", y="Loan_Amount_Term", color="Education", barmode="group")

train_data = get_data()
# print(train_data.columns)
y = train_data.iloc[:,-1]

train_data = train_data.drop(['Loan_ID','Loan_Status'],axis=1)
test_data = get_testdata()
test_data = test_data.drop(['Loan_ID'],axis=1)

def encoding(data):
    data = pd.get_dummies(data, columns=["Gender","Married","Education","Self_Employed","Property_Area"],drop_first=True)
    return data


train = encoding(train_data)
test = encoding(test_data)

print("training")
print(train)
print("output")
print(y)
print("testing")
print(test)

from sklearn.ensemble import RandomForestClassifier
random = RandomForestClassifier()
random.fit(train,y) 
y_pred = random.predict(test)

ds.push("Random_Forest", random, "Random Forest Loan Prediction")


# model = ds.pull('/dstack/Random_Forest')
# y_pred = model.predict(test)
print('predictions')
print(y_pred)
 

frame = ds.frame("Loan_Prediction")
frame.add(ds.app(scatter_handler), params={"Visualize": ds.tab()})
frame.add(ds.app(bar_handler), params={"Predict": ds.tab()})

url = frame.push()
print(url)