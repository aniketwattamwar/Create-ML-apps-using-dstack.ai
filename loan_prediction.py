import dstack as ds
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import dstack.controls as ctrl

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
y = train_data.iloc[:,-1]

train_data = train_data.drop(['Loan_ID','Loan_Status'],axis=1)
test_data = get_testdata()
ids = test_data.iloc[:,0]
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
#When running it normmaly without pushing it to dstack it gave the output
# y_pred = random.predict(test)

ds.push("Random_Forest", random, "Random Forest Loan Prediction")

#The below two lines when uncommented gave the error- Comment the line 51 if below two are to be used
model = ds.pull('/dstack/Random_Forest')

# print('predictions')
# print(y_pred)
values = ctrl.ComboBox(data=['Predict'], label="Predictions",require_apply=True)

def get_predicted(values: ctrl.ComboBox):
    y_pred = model.predict(test)
    y_pred = pd.DataFrame(y_pred)
    y_pred = y_pred.rename(columns={0:'Prediction'})
    y_pred['ID'] = ids
    
    return y_pred

 

p_app = ds.app(get_predicted, values = values)

frame = ds.frame("Loan_Prediction")
frame.add(p_app, params={"Predicted": ds.tab()})
frame.add(ds.app(scatter_handler), params={"Visualize": ds.tab()})
frame.add(ds.app(bar_handler), params={"Bar Plot": ds.tab()})

url = frame.push()
print(url)