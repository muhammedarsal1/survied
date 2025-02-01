import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv('passenger.csv')
print(df.head())

print(df.info())

print(df.isnull().sum())

print(df.duplicated().sum())

print(df['Gender'].unique())
print(df['Class'].unique())
print(df['Seat_Type'].unique())
print(df['Fare_Paid'].unique())

df=df.drop(['Passenger_ID'],axis=1)
df=df.drop(['Name'],axis=1)

print(df)

le=LabelEncoder()

df['Gender']=le.fit_transform(df['Gender'])
df['Class']=le.fit_transform(df['Class'])
df['Seat_Type']=le.fit_transform(df['Seat_Type'])


print(df)

x=df[['Gender','Age','Class','Seat_Type','Fare_Paid']]
y=df['Survival_Status']

x_test,x_train,y_test,y_train=train_test_split(x,y,test_size=0.3,random_state=40)

Classifier=DecisionTreeClassifier()
Classifier.fit(x_train,y_train)

pickle.dump(Classifier,open('mpdel.pkl','wb'))